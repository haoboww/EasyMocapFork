#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将雷达点云投影到多视角RGB图像中（带镜像修正和刚性变换对齐）
支持生成视频输出

修正内容：
1. 镜像修正：点云采集是镜像的，需要翻转
2. 刚性变换：雷达相对于cam12的位置偏移（左13cm，下4cm，前3cm）
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob


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


def radar_spherical_to_cartesian(range_m, az_deg, el_deg):
    """
    将雷达球坐标转换为笛卡尔坐标（雷达坐标系）
    雷达坐标系: X前, Y左, Z上
    """
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    
    x = range_m * np.cos(el_rad) * np.cos(az_rad)
    y = range_m * np.cos(el_rad) * np.sin(az_rad)
    z = range_m * np.sin(el_rad)
    
    return np.stack([x, y, z], axis=1)


def transform_radar_to_camera_with_calibration(
    radar_points, 
    translation_offset=None
):
    """
    将雷达坐标系转换到相机坐标系（带刚性变换）
    
    Args:
        radar_points: (N, 3) 雷达坐标系点云 [X前, Y左, Z上]
        translation_offset: [dx, dy, dz] 雷达相对相机的偏移（相机坐标系）
                           默认 [-0.13, 0.04, 0.03] 表示雷达在相机左13cm、下4cm、前3cm
    
    坐标系转换：
        雷达坐标系: X前, Y左, Z上
        相机坐标系: X右, Y下, Z前
    
    步骤：
        1. 坐标轴转换：
           cam_X = -radar_Y (雷达左 -> 相机右)
           cam_Y = -radar_Z (雷达上 -> 相机下)
           cam_Z =  radar_X (雷达前 -> 相机前)
        2. 刚性变换：加上雷达相对相机的偏移
    """
    radar_points = radar_points.copy()
    
    # 步骤1: 坐标轴转换到相机坐标系
    cam_points = np.zeros_like(radar_points)
    cam_points[:, 0] = -radar_points[:, 1]  # cam_X = -radar_Y
    cam_points[:, 1] = -radar_points[:, 2]  # cam_Y = -radar_Z
    cam_points[:, 2] = radar_points[:, 0]   # cam_Z = radar_X
    
    # 步骤2: 刚性变换（雷达相对相机的偏移）
    if translation_offset is None:
        # 默认偏移：雷达在相机左13cm、下4cm、前3cm
        # 相机坐标系：X右正，Y下正，Z前正
        # 雷达在左边 -> X负，下方 -> Y正，前方 -> Z正
        translation_offset = np.array([-0.13, 0.04, 0.03])
    
    cam_points += translation_offset
    
    return cam_points


def project_points(points_3d, K, R, T, dist):
    """将3D点投影到2D图像平面"""
    points_cam = (R @ points_3d.T + T).T
    points_2d = points_cam[:, :2] / points_cam[:, 2:3]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    points_2d[:, 0] = points_2d[:, 0] * fx + cx
    points_2d[:, 1] = points_2d[:, 1] * fy + cy
    
    return points_2d, points_cam[:, 2]


def load_radar_data(csv_path):
    """加载雷达点云数据（新表头格式）"""
    df = pd.read_csv(csv_path)
    
    # 检查是否为新表头格式
    if 'frame_index' in df.columns:
        # 新表头格式
        df = df.rename(columns={
            'frame_index': 'frame',
            'range': 'range_m',
            'velocity': 'vel_mps',
            'horizontal_angle': 'az_deg',
            'elevation_angle': 'el_deg',
            'power': 'mag'
        })
    elif 'frame' not in df.columns:
        raise ValueError("CSV文件缺少 'frame' 或 'frame_index' 列")
    
    return df


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
    """选择与帧对应的图像"""
    if frame_name in name_map:
        return name_map[frame_name], "name"
    for name, path in name_map.items():
        if name.endswith(frame_name):
            return path, "suffix"
    if order_idx < len(images_sorted):
        return images_sorted[order_idx], "order"
    return None, "missing"


def draw_pointcloud(img, points_2d, depth, power=None, color_mode='depth', point_size=3):
    """
    在图像上绘制点云
    color_mode: 'depth' - 按深度着色, 'power' - 按功率着色, 'fixed' - 固定颜色
    """
    result = img.copy()
    h, w = result.shape[:2]
    
    if len(points_2d) == 0:
        return result
    
    # 只绘制深度为正的点
    valid_mask = depth > 0
    points_2d = points_2d[valid_mask]
    depth = depth[valid_mask]
    if power is not None:
        power = power[valid_mask]
    
    if len(points_2d) == 0:
        return result
    
    # 根据模式选择颜色
    if color_mode == 'depth':
        # 按深度着色（近红远蓝）
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        colors = plt.cm.jet(depth_normalized)[:, :3] * 255
        colors = colors.astype(np.uint8)
    elif color_mode == 'power' and power is not None:
        # 按功率着色（低绿高红）
        power_normalized = (power - power.min()) / (power.max() - power.min() + 1e-6)
        colors = plt.cm.hot(power_normalized)[:, :3] * 255
        colors = colors.astype(np.uint8)
    else:
        # 固定颜色（绿色）
        colors = np.tile([0, 255, 0], (len(points_2d), 1))
    
    # 绘制点
    for (u, v), color in zip(points_2d, colors):
        pt = (int(round(u)), int(round(v)))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            color_bgr = tuple(map(int, color[::-1]))  # RGB -> BGR
            cv2.circle(result, pt, point_size, color_bgr, -1, cv2.LINE_AA)
    
    return result


def create_video_from_images(image_dir, output_video_path, fps=10):
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


# 导入matplotlib用于颜色映射
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='将雷达点云投影到多视角图像（带对齐修正）')
    parser.add_argument('--data_root', type=str, default='../data/examples/my_multiview')
    parser.add_argument('--radar_csv', type=str, required=True,
                        help='雷达点云CSV文件路径')
    parser.add_argument('--cameras', type=str, nargs='+', 
                        default=['cam0', 'cam2', 'cam4', 'cam6', 'cam12'])
    parser.add_argument('--frame_start', type=int, default=0)
    parser.add_argument('--frame_end', type=int, default=99999)
    parser.add_argument('--color_mode', type=str, default='depth',
                        choices=['depth', 'power', 'fixed'],
                        help='点云着色模式: depth-按深度, power-按功率, fixed-固定颜色')
    parser.add_argument('--point_size', type=int, default=4,
                        help='点的大小')
    parser.add_argument('--min_power', type=float, default=0.0,
                        help='最小功率阈值，过滤低功率点')
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help='最大深度（米），过滤过远的点')
    parser.add_argument('--offset_x', type=float, default=-0.13,
                        help='雷达相对相机X偏移（米），负值表示左侧，默认-0.13')
    parser.add_argument('--offset_y', type=float, default=0.04,
                        help='雷达相对相机Y偏移（米），正值表示下方，默认0.04')
    parser.add_argument('--offset_z', type=float, default=0.03,
                        help='雷达相对相机Z偏移（米），正值表示前方，默认0.03')
    parser.add_argument('--create_video', action='store_true')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='视频帧率（建议与采集帧率一致，默认10fps）')
    parser.add_argument('--output_dir_name', type=str, default='vis_pointcloud_aligned')
    
    args = parser.parse_args()
    
    # 创建输出目录
    vis_output_dir = os.path.join(os.path.dirname(args.radar_csv), args.output_dir_name)
    os.makedirs(vis_output_dir, exist_ok=True)
    
    for cam_name in args.cameras:
        os.makedirs(os.path.join(vis_output_dir, cam_name), exist_ok=True)
    
    # 加载相机参数
    print("📷 加载相机参数...")
    intri_path = os.path.join(args.data_root, 'intri.yml')
    extri_path = os.path.join(args.data_root, 'extri.yml')
    cameras = load_cameras(intri_path, extri_path, args.cameras)
    print(f"   找到 {len(cameras)} 个相机: {args.cameras}")
    
    # 加载雷达数据
    print("📡 加载雷达点云数据...")
    radar_df = load_radar_data(args.radar_csv)
    print(f"   总点数: {len(radar_df)}")
    print(f"   帧数: {radar_df['frame'].nunique()}")
    
    # 预加载每个相机的图片列表
    cam_images = {cam: load_camera_images(args.data_root, cam) for cam in args.cameras}
    for cam, (imgs, _) in cam_images.items():
        print(f"   相机 {cam} 发现 {len(imgs)} 张图片")
    
    # 获取要处理的帧列表
    all_frames = sorted(radar_df['frame'].unique())
    frames = [f for f in all_frames if args.frame_start <= f <= args.frame_end]
    
    if len(frames) == 0:
        print("❌ 未找到匹配的帧")
        return
    
    print(f"\n☁️  开始投影点云到图像（带对齐修正）")
    print(f"   帧数: {len(frames)}")
    print(f"   视角数: {len(args.cameras)}")
    print(f"   着色模式: {args.color_mode}")
    print(f"   点大小: {args.point_size}")
    print(f"   最小功率: {args.min_power}")
    print(f"   最大深度: {args.max_depth}m")
    print(f"   刚性偏移: X={args.offset_x}m, Y={args.offset_y}m, Z={args.offset_z}m")
    print(f"              (雷达在相机: {'左' if args.offset_x < 0 else '右'}{abs(args.offset_x)*100:.0f}cm, "
          f"{'下' if args.offset_y > 0 else '上'}{abs(args.offset_y)*100:.0f}cm, "
          f"{'前' if args.offset_z > 0 else '后'}{abs(args.offset_z)*100:.0f}cm)\n")
    
    translation_offset = np.array([args.offset_x, args.offset_y, args.offset_z])
    
    for order_idx, frame_idx in enumerate(tqdm(frames, desc="处理进度")):
        # 获取该帧的雷达数据
        frame_data = radar_df[radar_df['frame'] == frame_idx]
        
        if len(frame_data) == 0:
            continue
        
        # 过滤低功率点
        frame_data = frame_data[frame_data['mag'] >= args.min_power]
        
        if len(frame_data) == 0:
            continue
        
        # 提取雷达数据
        range_m = frame_data['range_m'].values
        az_deg = frame_data['az_deg'].values
        el_deg = frame_data['el_deg'].values
        power = frame_data['mag'].values
        
        # 球坐标转笛卡尔坐标（雷达坐标系）
        radar_points_radar = radar_spherical_to_cartesian(range_m, az_deg, el_deg)
        
        # 雷达坐标系转相机坐标系（带刚性变换）
        radar_points_cam = transform_radar_to_camera_with_calibration(
            radar_points_radar,
            translation_offset=translation_offset
        )
        
        # 投影到每个视角
        for cam_name in args.cameras:
            images_sorted, name_map = cam_images.get(cam_name, ([], {}))
            frame_name = f"{frame_idx:06d}"
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
            
            # 投影点云
            points_2d, depth = project_points(radar_points_cam, K, R, T, dist)
            
            # 过滤：深度为正、不超过最大深度、在图像范围内
            h, w = image.shape[:2]
            valid_mask = (
                (depth > 0) &
                (depth <= args.max_depth) &
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            )
            
            points_2d_valid = points_2d[valid_mask]
            depth_valid = depth[valid_mask]
            power_valid = power[valid_mask]
            
            # 绘制点云
            result = draw_pointcloud(
                image, points_2d_valid, depth_valid, power_valid,
                color_mode=args.color_mode,
                point_size=args.point_size
            )
            
            # 添加标签
            cv2.putText(result, f'Frame: {frame_name}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(result, f'View: {cam_name} | Points: {len(points_2d_valid)} (Aligned)', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存结果
            output_path = os.path.join(vis_output_dir, cam_name, f'{frame_name}.jpg')
            cv2.imwrite(output_path, result)
    
    print(f"\n✅ 完成！结果保存在: {vis_output_dir}")
    print(f"   帧范围: {args.frame_start} - {args.frame_end}")
    print(f"   视角数: {len(args.cameras)}")
    
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

