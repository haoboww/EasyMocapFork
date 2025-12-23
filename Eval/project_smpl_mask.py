#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将SMPL结果投影为mask到所有视角的RGB图像中
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


def load_cameras(intri_path, extri_path, camera_names):
    """加载相机内外参"""
    cameras = {}
    
    # 使用OpenCV读取YAML（支持OpenCV特定格式）
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
    """
    将3D点投影到2D图像平面
    points_3d: (N, 3) - 3D点
    K: (3, 3) - 内参矩阵
    R: (3, 3) - 旋转矩阵
    T: (3, 1) - 平移向量
    dist: (5,) - 畸变系数
    """
    # 转换到相机坐标系
    points_cam = (R @ points_3d.T + T).T
    
    # 投影到图像平面
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    
    # 应用内参
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    points_2d[:, 0] = points_2d[:, 0] * fx + cx
    points_2d[:, 1] = points_2d[:, 1] * fy + cy
    
    return points_2d, points_cam[:, 2]


def render_smpl_mask(img_shape, vertices, faces, K, R, T, dist):
    """
    渲染SMPL模型为mask
    返回: 二值mask (0/255)
    """
    h, w = img_shape[:2]
    
    # 投影顶点
    points_2d, depth = project_points(vertices, K, R, T, dist)
    
    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 按深度排序面片（从远到近）
    face_depths = []
    for face in faces:
        avg_depth = np.mean([depth[face[0]], depth[face[1]], depth[face[2]]])
        face_depths.append(avg_depth)
    
    sorted_indices = np.argsort(face_depths)[::-1]
    
    # 绘制面片到mask
    for idx in sorted_indices:
        face = faces[idx]
        # 只绘制深度为正的面片
        if depth[face[0]] > 0 and depth[face[1]] > 0 and depth[face[2]] > 0:
            pts = points_2d[face].astype(np.int32)
            
            # 检查是否在图像范围内
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) & 
                     (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                cv2.fillConvexPoly(mask, pts, 255)
    
    return mask


def apply_mask_to_image(img, mask, mask_color=(0, 255, 0), alpha=0.5, mode='overlay'):
    """
    将mask应用到图像上
    mode: 'overlay' - 半透明叠加, 'binary' - 二值mask, 'colored' - 彩色mask, 'contour' - 轮廓
    """
    if mode == 'binary':
        # 返回二值mask（3通道）
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    elif mode == 'overlay':
        # 半透明叠加
        overlay = img.copy()
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay[mask > 0] = mask_color
        result = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
        return result
    
    elif mode == 'colored':
        # 彩色mask，保留原图背景
        result = img.copy()
        result[mask > 0] = mask_color
        return result
    
    elif mode == 'contour':
        # 只绘制轮廓
        result = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, mask_color, 2)
        return result
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def collect_smpl_frames(output_root, frame_start, frame_end):
    """获取SMPL帧列表，允许根据起止帧裁剪。"""
    smpl_dir = os.path.join(output_root, 'smpl')
    smpl_files = sorted(glob(os.path.join(smpl_dir, '*.json')))
    frames = []
    for path in smpl_files:
        base = os.path.splitext(os.path.basename(path))[0]
        # 尝试按数字帧号过滤；非数字则直接保留
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
    """加载某个相机的所有jpg图片，并返回排序列表与基名索引。"""
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
    """
    选择与SMPL帧对应的图像：
    1) 优先基名精确匹配 (e.g., 000001)
    2) 若无，则尝试基名后缀匹配 (处理某些前缀)
    3) 再无，则按排序顺序回退 (camX_at_timestamp)
    """
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
    # 获取所有图像
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
    
    if len(images) == 0:
        print(f"   ⚠️  {image_dir} 中没有图像，跳过视频生成")
        return False
    
    # 读取第一张图像获取尺寸
    first_img = cv2.imread(os.path.join(image_dir, images[0]))
    if first_img is None:
        print(f"   ⚠️  无法读取图像，跳过视频生成")
        return False
    
    h, w = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # 写入所有帧
    for img_name in tqdm(images, desc=f"生成视频 {os.path.basename(output_video_path)}", leave=False):
        img = cv2.imread(os.path.join(image_dir, img_name))
        if img is not None:
            out.write(img)
    
    out.release()
    print(f"   ✅ 视频已保存: {output_video_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='将SMPL投影为mask到多视角图像')
    parser.add_argument('--data_root', type=str, default='../data/examples/my_multiview',
                        help='数据根目录（包含images、intri.yml、extri.yml）')
    parser.add_argument('--output_root', type=str, default='../output/detect_triangulate_fitSMPL',
                        help='输出根目录（包含smpl文件夹）')
    parser.add_argument('--smpl_model_path', type=str, default='../data/models/smpl/SMPL_NEUTRAL.pkl',
                        help='SMPL模型路径')
    parser.add_argument('--cameras', type=str, nargs='+', 
                        default=['cam0', 'cam2', 'cam4', 'cam6', 'cam12'],
                        help='相机名称列表')
    parser.add_argument('--frame_start', type=int, default=0,
                        help='起始帧')
    parser.add_argument('--frame_end', type=int, default=99999,
                        help='结束帧')
    parser.add_argument('--mask_mode', type=str, default='overlay',
                        choices=['overlay', 'binary', 'colored', 'contour'],
                        help='Mask模式: overlay-半透明叠加, binary-二值mask, colored-彩色mask, contour-轮廓')
    parser.add_argument('--mask_color', type=int, nargs=3, default=[0, 255, 0],
                        help='Mask颜色 (B G R)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='overlay模式的透明度 (0-1)')
    parser.add_argument('--create_video', action='store_true',
                        help='是否生成视频')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='视频帧率（建议与采集帧率一致，默认10fps）')
    parser.add_argument('--output_dir_name', type=str, default='vis_smpl_mask',
                        help='输出目录名称')
    
    args = parser.parse_args()
    
    # 配置参数
    data_root = args.data_root
    output_root = args.output_root
    smpl_model_path = args.smpl_model_path
    camera_names = args.cameras
    frame_start = args.frame_start
    frame_end = args.frame_end
    mask_mode = args.mask_mode
    mask_color = tuple(args.mask_color)
    alpha = args.alpha
    create_video = args.create_video
    video_fps = args.video_fps
    
    # 创建输出目录
    vis_output_dir = os.path.join(output_root, args.output_dir_name)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    for cam_name in camera_names:
        os.makedirs(os.path.join(vis_output_dir, cam_name), exist_ok=True)
    
    # 加载相机参数
    print("📷 加载相机参数...")
    intri_path = os.path.join(data_root, 'intri.yml')
    extri_path = os.path.join(data_root, 'extri.yml')
    cameras = load_cameras(intri_path, extri_path, camera_names)
    print(f"   找到 {len(cameras)} 个相机: {camera_names}")
    
    # 加载SMPL模型
    print("📦 加载SMPL模型...")
    smpl_model = load_smpl_model(smpl_model_path)
    print("   SMPL模型加载完成")
    
    # 预加载每个相机的图片列表
    cam_images = {cam: load_camera_images(data_root, cam) for cam in camera_names}
    for cam, (imgs, _) in cam_images.items():
        print(f"   相机 {cam} 发现 {len(imgs)} 张图片")
    
    # 准备SMPL帧列表
    frames = collect_smpl_frames(output_root, frame_start, frame_end)
    if len(frames) == 0:
        print("❌ 未找到SMPL帧，请检查路径与范围设置")
        return
    
    print(f"\n🎭 开始生成SMPL mask并投影到图像")
    print(f"   帧数: {len(frames)}")
    print(f"   视角数: {len(camera_names)}")
    print(f"   Mask模式: {mask_mode}")
    print(f"   Mask颜色: {mask_color}")
    if mask_mode == 'overlay':
        print(f"   透明度: {alpha}")
    print()
    
    for order_idx, (frame_name, smpl_path) in enumerate(tqdm(frames, desc="处理进度")):
        # 读取SMPL参数
        with open(smpl_path, 'r') as f:
            smpl_data = json.load(f)
        if len(smpl_data) == 0:
            continue
        
        # 提取SMPL参数（假设只有一个人）
        person_data = smpl_data[0]
        
        # 准备SMPL参数
        poses = torch.FloatTensor(person_data['poses']).reshape(1, -1)
        shapes = torch.FloatTensor(person_data['shapes']).reshape(1, -1)
        Rh = torch.FloatTensor(person_data['Rh']).reshape(1, 3)
        Th = torch.FloatTensor(person_data['Th']).reshape(1, 3)
        
        # 前向传播获取顶点
        with torch.no_grad():
            params_dict = {
                'poses': poses,
                'shapes': shapes,
                'Rh': Rh,
                'Th': Th
            }
            vertices = smpl_model.vertices(params_dict, return_tensor=True).cpu().numpy()[0]
        
        faces = smpl_model.faces
        
        # 投影到每个视角
        for cam_name in camera_names:
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
            
            # 渲染SMPL mask
            mask = render_smpl_mask(image.shape, vertices, faces, K, R, T, dist)
            
            # 应用mask到图像
            result = apply_mask_to_image(image, mask, mask_color, alpha, mask_mode)
            
            # 添加帧号标签
            cv2.putText(result, f'Frame: {frame_name}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result, f'View: {cam_name} ({match_mode})', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(vis_output_dir, cam_name, f'{frame_name}.jpg')
            cv2.imwrite(output_path, result)
    
    print(f"\n✅ 完成！结果保存在: {vis_output_dir}")
    print(f"   帧范围: {frame_start} - {frame_end}")
    print(f"   视角数: {len(camera_names)}")
    
    # 生成视频
    if create_video:
        print(f"\n🎬 开始生成视频 (fps={video_fps})...")
        video_dir = os.path.join(vis_output_dir, 'videos')
        os.makedirs(video_dir, exist_ok=True)
        
        for cam_name in camera_names:
            cam_img_dir = os.path.join(vis_output_dir, cam_name)
            video_path = os.path.join(video_dir, f'{cam_name}.mp4')
            create_video_from_images(cam_img_dir, video_path, video_fps)
        
        print(f"\n✅ 视频生成完成！保存在: {video_dir}")


if __name__ == '__main__':
    main()


