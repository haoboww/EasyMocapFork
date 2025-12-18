#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将SMPL结果投影到所有视角的RGB图像中
"""

import os
import sys

# 添加EasyMocap路径
sys.path.insert(0, os.path.dirname(__file__))

# 修复chumpy兼容性问题（必须在导入其他模块之前）
# import fix_chumpy

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


def draw_smpl_wireframe(img, vertices, faces, K, R, T, dist, color=(0, 255, 0), thickness=1):
    """在图像上绘制SMPL线框"""
    # 投影顶点
    points_2d, depth = project_points(vertices, K, R, T, dist)
    
    # 只绘制正面的顶点
    valid_mask = depth > 0
    
    # 绘制边缘（采样部分边以避免太密集）
    for face in faces[::10]:  # 每10个面画一个
        pts = []
        valid = True
        for idx in face:
            if valid_mask[idx]:
                pt = points_2d[idx].astype(np.int32)
                if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                    pts.append(pt)
                else:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid and len(pts) == 3:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)
    
    return img


def draw_smpl_mesh(img, vertices, faces, K, R, T, dist, color=(0, 255, 0), alpha=0.6):
    """在图像上绘制填充的SMPL网格"""
    # 投影顶点
    points_2d, depth = project_points(vertices, K, R, T, dist)
    
    h, w = img.shape[:2]
    overlay = img.copy()
    
    # 按深度排序面片（从远到近）
    face_depths = []
    for face in faces:
        avg_depth = np.mean([depth[face[0]], depth[face[1]], depth[face[2]]])
        face_depths.append(avg_depth)
    
    sorted_indices = np.argsort(face_depths)[::-1]
    
    # 绘制面片
    for idx in sorted_indices:
        face = faces[idx]
        # 只绘制深度为正的面片
        if depth[face[0]] > 0 and depth[face[1]] > 0 and depth[face[2]] > 0:
            pts = points_2d[face].astype(np.int32)
            
            # 检查是否在图像范围内
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) & 
                     (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                cv2.fillConvexPoly(overlay, pts, color)
    
    # 融合
    result = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    return result


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


def main():
    # 配置参数
    data_root = '/home/bupt630/Dabai/AmmWave/EasyMocap/data/examples/my_multiview'
    output_root = '/home/bupt630/Dabai/AmmWave/EasyMocap/output/detect_triangulate_fitSMPL'
    smpl_model_path = '/home/bupt630/Dabai/AmmWave/EasyMocap/data/models/smpl/SMPL_NEUTRAL.pkl'
    
    # camera_names = ['01', '02', '03', '04', '05', '06', '07', '08']
    # camera_names = ['02', '04', '06', '08']
    # camera_names = ['01', '02', '03', '04', '05']
    camera_names = ['cam2', 'cam10', 'cam12', 'cam0']
    # camera_names = ['02','03','07','08']
    frame_start = 0
    frame_end = 99999  # 支持更大范围，实际帧数由数据决定
    # frame_start = 100
    # frame_end = 169  # 包含119
    
    # 创建输出目录
    vis_output_dir = os.path.join(output_root, 'vis_smpl_projection')
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
    print(f"🎨 开始投影SMPL到图像，共 {len(frames)} 帧")
    
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
            
            # 绘制SMPL网格（线框模式）
            result = draw_smpl_wireframe(image, vertices, faces, K, R, T, dist, 
                                        color=(0, 255, 0), thickness=1)
            
            # 也可以使用填充模式（取消注释下面这行并注释上面这行）
            # result = draw_smpl_mesh(image, vertices, faces, K, R, T, dist, 
            #                        color=(0, 255, 0), alpha=0.6)
            
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


if __name__ == '__main__':
    main()

