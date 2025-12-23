#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将SMPL骨架投影到所有视角的RGB图像中
支持生成视频输出
"""

import os
import sys
import argparse

# 添加EasyMocap路径
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import json
import numpy as np
import torch
from tqdm import tqdm
from glob import glob


# SMPL骨架连接关系（24个关节）- 官方标准kintree
# 关节顺序: 0-MidHip, 1-LUpLeg, 2-RUpLeg, 3-Spine, 4-LLeg, 5-RLeg, 6-Spine1, 
#          7-LFoot, 8-RFoot, 9-Spine2, 10-LToeBase, 11-RToeBase, 12-Neck, 
#          13-LCollar, 14-RCollar, 15-Head, 16-LShoulder, 17-RShoulder,
#          18-LElbow, 19-RElbow, 20-LWrist, 21-RWrist, 22-LHand, 23-RHand
SMPL_SKELETON = [
    [0, 1],   [0, 2],   [0, 3],    # 骨盆 -> 左大腿, 右大腿, 脊柱
    [1, 4],   [2, 5],              # 大腿 -> 小腿
    [3, 6],                        # 脊柱 -> 脊柱1
    [4, 7],   [5, 8],              # 小腿 -> 脚踝
    [6, 9],                        # 脊柱1 -> 脊柱2
    [7, 10],  [8, 11],             # 脚踝 -> 脚尖
    [9, 12],  [9, 13],  [9, 14],   # 脊柱2 -> 颈部, 左锁骨, 右锁骨
    [12, 15],                      # 颈部 -> 头部
    [13, 16], [14, 17],            # 锁骨 -> 肩膀
    [16, 18], [17, 19],            # 肩膀 -> 肘部
    [18, 20], [19, 21],            # 肘部 -> 手腕
    [20, 22], [21, 23],            # 手腕 -> 手部
]


def load_cameras(intri_path, extri_path, camera_names):
    """加载相机内外参"""
    cameras = {}
    
    intri_fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri_fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    
    for cam_name in camera_names:
        cameras[cam_name] = {
            'K': intri_fs.getNode(f'K_{cam_name}').mat(),
            'dist': intri_fs.getNode(f'dist_{cam_name}').mat().flatten(),
            'R': extri_fs.getNode(f'Rot_{cam_name}').mat(),
            'T': extri_fs.getNode(f'T_{cam_name}').mat()
        }
    
    intri_fs.release()
    extri_fs.release()
    
    return cameras


def load_smpl_model(model_path):
    """加载SMPL模型"""
    from easymocap.bodymodel.smpl import SMPLModel
    
    device = torch.device('cpu')
    smpl = SMPLModel(
        model_path=model_path,
        model_type='smpl',
        device=device,
        NUM_SHAPES=10
    )
    return smpl


def project_points(points_3d, K, R, T, dist):
    """将3D点投影到2D图像平面"""
    points_cam = (R @ points_3d.T + T).T
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    points_2d[:, 0] = points_2d[:, 0] * fx + cx
    points_2d[:, 1] = points_2d[:, 1] * fy + cy
    
    return points_2d, points_cam[:, 2]


def draw_skeleton(img, joints_2d, depth, skeleton_links, joint_color=(0, 255, 0), 
                  bone_color=(255, 0, 0), joint_radius=5, bone_thickness=3):
    """
    在图像上绘制骨架
    joints_2d: (N, 2) - 2D关节位置
    depth: (N,) - 关节深度
    """
    result = img.copy()
    h, w = img.shape[:2]
    
    # 绘制骨骼连接
    for link in skeleton_links:
        j1, j2 = link
        if depth[j1] > 0 and depth[j2] > 0:  # 只绘制在相机前方的关节
            pt1 = joints_2d[j1].astype(np.int32)
            pt2 = joints_2d[j2].astype(np.int32)
            
            # 检查是否在图像范围内
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(result, tuple(pt1), tuple(pt2), bone_color, bone_thickness, cv2.LINE_AA)
    
    # 绘制关节点（在线条之上）
    for i, (joint, d) in enumerate(zip(joints_2d, depth)):
        if d > 0:  # 只绘制在相机前方的关节
            pt = joint.astype(np.int32)
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(result, tuple(pt), joint_radius, joint_color, -1, cv2.LINE_AA)
                # 可选：添加关节索引标签
                # cv2.putText(result, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    return result


def collect_smpl_frames(output_root, frame_start, frame_end):
    """获取SMPL帧列表"""
    smpl_dir = os.path.join(output_root, 'smpl')
    smpl_files = sorted(glob(os.path.join(smpl_dir, '*.json')))
    frames = []
    for path in smpl_files:
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            idx = int(base)
        except ValueError:
            idx = None
        if idx is not None:
            if idx < frame_start or idx > frame_end:
                continue
        frames.append((base, path))
    return frames


def load_camera_images(data_root, cam_name):
    """加载某个相机的所有jpg图片"""
    cam_dir = os.path.join(data_root, 'images', cam_name)
    if not os.path.isdir(cam_dir):
        return [], {}
    images = [
        os.path.join(cam_dir, f)
        for f in sorted(os.listdir(cam_dir))
        if f.lower().endswith('.jpg')
    ]
    name_map = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
    return images, name_map


def choose_image(frame_name, order_idx, images_sorted, name_map):
    """选择与SMPL帧对应的图像"""
    if frame_name in name_map:
        return name_map[frame_name], "name"
    for name, path in name_map.items():
        if name.endswith(frame_name):
            return path, "suffix"
    if order_idx < len(images_sorted):
        return images_sorted[order_idx], "order"
    return None, "missing"


def create_video_from_images(image_dir, output_video_path, fps=30):
    """从图像序列创建视频"""
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    
    if len(images) == 0:
        print(f"   ⚠️  {image_dir} 中没有图像")
        return False
    
    first_img = cv2.imread(os.path.join(image_dir, images[0]))
    if first_img is None:
        return False
    
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    for img_name in tqdm(images, desc=f"生成视频 {os.path.basename(output_video_path)}", leave=False):
        img = cv2.imread(os.path.join(image_dir, img_name))
        if img is not None:
            out.write(img)
    
    out.release()
    print(f"   ✅ 视频已保存: {output_video_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='将SMPL骨架投影到多视角图像')
    parser.add_argument('--data_root', type=str, default='../data/examples/my_multiview')
    parser.add_argument('--output_root', type=str, default='../output/detect_triangulate_fitSMPL')
    parser.add_argument('--smpl_model_path', type=str, default='../data/models/smpl/SMPL_NEUTRAL.pkl')
    parser.add_argument('--cameras', type=str, nargs='+', 
                        default=['cam0', 'cam2', 'cam4', 'cam6', 'cam12'])
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=99999)
    parser.add_argument('--joint_color', type=int, nargs=3, default=[0, 255, 0],
                        help='关节颜色 (B G R)')
    parser.add_argument('--bone_color', type=int, nargs=3, default=[255, 0, 0],
                        help='骨骼颜色 (B G R)')
    parser.add_argument('--joint_radius', type=int, default=5,
                        help='关节圆点半径')
    parser.add_argument('--bone_thickness', type=int, default=3,
                        help='骨骼线条粗细')
    parser.add_argument('--create_video', action='store_true')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='视频帧率（建议与采集帧率一致，默认10fps）')
    parser.add_argument('--output_dir_name', type=str, default='vis_smpl_skeleton')
    
    args = parser.parse_args()
    
    # 创建输出目录
    vis_output_dir = os.path.join(args.output_root, args.output_dir_name)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    for cam_name in args.cameras:
        os.makedirs(os.path.join(vis_output_dir, cam_name), exist_ok=True)
    
    # 加载相机参数
    print("📷 加载相机参数...")
    intri_path = os.path.join(args.data_root, 'intri.yml')
    extri_path = os.path.join(args.data_root, 'extri.yml')
    cameras = load_cameras(intri_path, extri_path, args.cameras)
    print(f"   找到 {len(cameras)} 个相机: {args.cameras}")
    
    # 加载SMPL模型
    print("📦 加载SMPL模型...")
    smpl_model = load_smpl_model(args.smpl_model_path)
    print("   SMPL模型加载完成")
    
    # 预加载每个相机的图片列表
    cam_images = {cam: load_camera_images(args.data_root, cam) for cam in args.cameras}
    for cam, (imgs, _) in cam_images.items():
        print(f"   相机 {cam} 发现 {len(imgs)} 张图片")
    
    # 准备SMPL帧列表
    frames = collect_smpl_frames(args.output_root, args.frame_start, args.frame_end)
    if len(frames) == 0:
        print("❌ 未找到SMPL帧")
        return
    
    print(f"\n🦴 开始投影SMPL骨架到图像")
    print(f"   帧数: {len(frames)}")
    print(f"   视角数: {len(args.cameras)}")
    print(f"   关节颜色: {args.joint_color}, 骨骼颜色: {args.bone_color}\n")
    
    for order_idx, (frame_name, smpl_path) in enumerate(tqdm(frames, desc="处理进度")):
        # 读取SMPL参数
        with open(smpl_path, 'r') as f:
            smpl_data = json.load(f)
        if len(smpl_data) == 0:
            continue
        
        person_data = smpl_data[0]
        
        # 准备SMPL参数
        poses = torch.FloatTensor(person_data['poses']).reshape(1, -1)
        shapes = torch.FloatTensor(person_data['shapes']).reshape(1, -1)
        Rh = torch.FloatTensor(person_data['Rh']).reshape(1, 3)
        Th = torch.FloatTensor(person_data['Th']).reshape(1, 3)
        
        # 前向传播获取关节位置
        with torch.no_grad():
            params_dict = {
                'poses': poses,
                'shapes': shapes,
                'Rh': Rh,
                'Th': Th
            }
            joints = smpl_model(return_verts=False, return_smpl_joints=True, 
                               return_tensor=True, **params_dict)
            joints_3d = joints.cpu().numpy()[0]  # (24, 3)
        
        # 投影到每个视角
        for cam_name in args.cameras:
            images_sorted, name_map = cam_images.get(cam_name, ([], {}))
            img_path, match_mode = choose_image(frame_name, order_idx, images_sorted, name_map)
            if img_path is None:
                continue
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # 获取相机参数
            K = cameras[cam_name]['K']
            R = cameras[cam_name]['R']
            T = cameras[cam_name]['T']
            dist = cameras[cam_name]['dist']
            
            # 投影关节点
            joints_2d, depth = project_points(joints_3d, K, R, T, dist)
            
            # 绘制骨架
            result = draw_skeleton(
                image, joints_2d, depth, SMPL_SKELETON,
                joint_color=tuple(args.joint_color),
                bone_color=tuple(args.bone_color),
                joint_radius=args.joint_radius,
                bone_thickness=args.bone_thickness
            )
            
            # 添加标签
            cv2.putText(result, f'Frame: {frame_name}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result, f'View: {cam_name} ({match_mode})', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(vis_output_dir, cam_name, f'{frame_name}.jpg')
            cv2.imwrite(output_path, result)
    
    print(f"\n✅ 完成！结果保存在: {vis_output_dir}")
    
    # 生成视频
    if args.create_video:
        print(f"\n🎬 开始生成视频 (fps={args.video_fps})...")
        video_dir = os.path.join(vis_output_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        for cam_name in args.cameras:
            cam_img_dir = os.path.join(vis_output_dir, cam_name)
            video_path = os.path.join(video_dir, f'{cam_name}.mp4')
            create_video_from_images(cam_img_dir, video_path, args.video_fps)
        
        print(f"\n✅ 视频生成完成！保存在: {video_dir}")


if __name__ == '__main__':
    main()

