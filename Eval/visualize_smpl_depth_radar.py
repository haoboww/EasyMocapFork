#!/usr/bin/env python3
"""
可视化 SMPL、深度图和雷达点云的对比。

功能:
- 为每一帧采样前向可见的 SMPL 顶点并投影到图像平面
- 读取 RealSense 深度，在采样点位置取深度并反投影为相机坐标系 3D 点
- 读取雷达点云数据（球坐标），转换为相机坐标系 3D 点并投影
- 生成两类可视化:
    1) 深度图上的 2D 误差分布叠加图 (SMPL vs 深度 vs 雷达)
    2) 3D 散点图 (红: SMPL, 蓝: 深度, 绿: 雷达)
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

# 让 easymocap 模块可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 复用 evaluate_depth.py 和 visualize_smpl_depth.py 中的工具函数
from evaluate_depth import (
    backproject_depth_to_3d,
    get_smpl_vertices,
    load_camera_params,
    load_smpl_model,
    load_smpl_params,
    project_to_image,
    read_depth_image,
    select_front_facing_vertices,
    transform_to_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 SMPL、深度和雷达点云的可视化对比"
    )
    parser.add_argument(
        "--root",
        default="/home/bupt630/Dabai/AmmWave/EasyMocap",
        help="项目根目录",
    )
    parser.add_argument(
        "--smpl-dir",
        default=None,
        help="SMPL 参数 (json) 路径，默认为 {root}/output/detect_triangulate_fitSMPL/smpl",
    )
    parser.add_argument(
        "--depth-dir",
        default=None,
        help="深度图 (png) 路径，默认为 {root}/Evaluation/depth",
    )
    parser.add_argument(
        "--radar-csv",
        default=None,
        help="雷达点云 CSV 文件路径，默认为 {root}/Eval/pointcloud_cfar_fixed.csv",
    )
    parser.add_argument(
        "--intri",
        default=None,
        help="相机内参文件 intri.yml",
    )
    parser.add_argument(
        "--extri",
        default=None,
        help="相机外参文件 extri.yml",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="SMPL 模型文件 (.pkl)",
    )
    parser.add_argument(
        "--cam-id",
        default="cam8",
        help="使用的相机编号",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="只可视化指定帧编号，不指定则批量处理",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="批量模式下处理的帧数上限，-1 表示全部",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="每帧从前向可见面采样的 SMPL 顶点数量",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="深度单位 (米/计数)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="深度上限 (米)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录，默认为 {root}/Eval/vis_smpl_depth_radar",
    )
    parser.add_argument(
        "--radar-min-mag",
        type=float,
        default=0.0,
        help="雷达点最小幅值阈值，低于此值的点将被过滤",
    )
    return parser.parse_args()


def prepare_paths(args: argparse.Namespace) -> argparse.Namespace:
    """补全默认路径并创建输出目录。"""
    if args.smpl_dir is None:
        args.smpl_dir = os.path.join(
            args.root, "output", "detect_triangulate_fitSMPL--eval", "smpl"
        )
    if args.depth_dir is None:
        args.depth_dir = os.path.join(args.root, "Evaluation", "depth")
    if args.radar_csv is None:
        args.radar_csv = os.path.join(args.root, "Eval", "pointcloud_cfar_fixed.csv")
    if args.intri is None:
        args.intri = os.path.join(
            args.root, "data", "examples", "my_multiview", "intri.yml"
        )
    if args.extri is None:
        args.extri = os.path.join(
            args.root, "data", "examples", "my_multiview", "extri.yml"
        )
    if args.model is None:
        args.model = os.path.join(
            args.root,
            "models",
            "pare",
            "data",
            "body_models",
            "smpl",
            "SMPL_NEUTRAL.pkl",
        )
    if args.output is None:
        args.output = os.path.join(args.root, "Eval", "vis_smpl_depth_radar")
    os.makedirs(args.output, exist_ok=True)
    return args


def load_radar_data(csv_path: str) -> pd.DataFrame:
    """加载雷达点云数据"""
    df = pd.read_csv(csv_path)
    return df


def radar_spherical_to_cartesian(
    range_m: np.ndarray, az_deg: np.ndarray, el_deg: np.ndarray
) -> np.ndarray:
    """
    将雷达球坐标转换为笛卡尔坐标（雷达坐标系）
    
    雷达坐标系约定（常见的毫米波雷达）:
    - X轴: 前向（距离方向）
    - Y轴: 左侧（方位角正方向）
    - Z轴: 上方（俯仰角正方向）
    
    参数:
        range_m: 距离 (米)
        az_deg: 方位角 (度), 左正右负
        el_deg: 俯仰角 (度), 上正下负
        
    返回:
        points_3d: (N, 3) 雷达坐标系下的点 [x, y, z]
    """
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    
    # 球坐标转笛卡尔坐标
    x = range_m * np.cos(el_rad) * np.cos(az_rad)  # 前向距离
    y = range_m * np.cos(el_rad) * np.sin(az_rad)  # 左右偏移
    z = range_m * np.sin(el_rad)                     # 上下偏移
    
    return np.stack([x, y, z], axis=1)


def transform_radar_to_camera(radar_points: np.ndarray) -> np.ndarray:
    """
    将雷达坐标系转换到相机坐标系
    
    假设雷达和深度相机在同一位置，但坐标系定义不同：
    - 雷达坐标系: X前, Y左, Z上
    - 相机坐标系: X右, Y下, Z前
    
    转换关系:
    cam_X = radar_Y   (雷达左 -> 相机右，需取反: -Y)
    cam_Y = -radar_Z  (雷达上 -> 相机下，需取反)
    cam_Z = radar_X   (雷达前 -> 相机前)
    
    注：实际应用中可能需要根据具体安装方式调整
    """
    # radar_points: (N, 3) [radar_x, radar_y, radar_z]
    cam_points = np.zeros_like(radar_points)
    cam_points[:, 0] = -radar_points[:, 1]  # cam_X = -radar_Y (雷达左变相机右)
    cam_points[:, 1] = -radar_points[:, 2]  # cam_Y = -radar_Z (雷达上变相机下)
    cam_points[:, 2] = radar_points[:, 0]   # cam_Z = radar_X (雷达前即相机前)
    
    return cam_points


def find_matched_frames(
    smpl_dir: str, depth_dir: str, radar_df: pd.DataFrame, frame: Optional[int], limit: int
) -> List[int]:
    """找到SMPL、深度图和雷达数据都存在的帧"""
    smpl_files = [
        int(os.path.splitext(os.path.basename(f))[0])
        for f in sorted(os.listdir(smpl_dir))
        if f.endswith(".json")
    ]
    depth_files = [
        int(os.path.splitext(os.path.basename(f))[0])
        for f in sorted(os.listdir(depth_dir))
        if f.endswith(".png")
    ]
    radar_frames = radar_df['frame'].unique().tolist()
    
    matched = sorted(list(set(smpl_files) & set(depth_files) & set(radar_frames)))
    
    if frame is not None:
        matched = [f for f in matched if f == frame]
    if limit is not None and limit > 0:
        matched = matched[:limit]
    return matched


def collect_frame_data(
    frame_idx: int,
    smpl_dir: str,
    depth_dir: str,
    radar_df: pd.DataFrame,
    smpl_model,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    dist: np.ndarray,
    num_samples: int,
    depth_scale: float,
    max_depth: float,
    radar_min_mag: float,
) -> Optional[Dict]:
    """
    收集单帧的SMPL、深度和雷达数据
    
    注：max_depth 参数会同时过滤深度点和雷达点，
    只保留深度值在 (0, max_depth] 范围内的点
    """
    smpl_path = os.path.join(smpl_dir, f"{frame_idx:06d}.json")
    depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.png")
    
    if not (os.path.exists(smpl_path) and os.path.exists(depth_path)):
        return None

    # 加载SMPL数据
    params = load_smpl_params(smpl_path)
    if params is None:
        return None

    vertices_world = get_smpl_vertices(smpl_model, params)
    vertices_cam = transform_to_camera(vertices_world, R, T)

    sampled_idx = select_front_facing_vertices(
        vertices_cam, smpl_model, num_samples=num_samples
    )
    sampled_vertices = vertices_cam[sampled_idx]
    smpl_points_2d = project_to_image(sampled_vertices, K, dist)

    # 加载深度数据
    depth_map = read_depth_image(depth_path, depth_scale)
    img_h, img_w = depth_map.shape[:2]

    valid_mask = np.zeros(len(smpl_points_2d), dtype=bool)
    depth_values = np.zeros(len(smpl_points_2d), dtype=np.float32)

    for i, (u, v) in enumerate(smpl_points_2d):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            depth_val = float(depth_map[v_int, u_int])
            if 0 < depth_val <= max_depth:
                valid_mask[i] = True
                depth_values[i] = depth_val

    if np.any(valid_mask):
        depth_points = backproject_depth_to_3d(
            smpl_points_2d[valid_mask], depth_values[valid_mask], K
        )
        smpl_points_valid = sampled_vertices[valid_mask]
        errors = np.linalg.norm(depth_points - smpl_points_valid, axis=1)
    else:
        depth_points = np.array([]).reshape(0, 3)
        smpl_points_valid = np.array([]).reshape(0, 3)
        errors = np.array([])

    # 加载雷达数据
    radar_frame = radar_df[radar_df['frame'] == frame_idx]
    if len(radar_frame) > 0:
        # 过滤低幅值点
        radar_frame = radar_frame[radar_frame['mag'] >= radar_min_mag]
        
        range_m = radar_frame['range_m'].values
        az_deg = radar_frame['az_deg'].values
        el_deg = radar_frame['el_deg'].values
        
        # 球坐标转笛卡尔坐标（雷达坐标系）
        radar_points_radar = radar_spherical_to_cartesian(range_m, az_deg, el_deg)
        
        # 雷达坐标系转相机坐标系
        radar_points_cam = transform_radar_to_camera(radar_points_radar)
        
        # 投影到图像平面
        radar_points_2d = project_to_image(radar_points_cam, K, dist)
        
        # 过滤在图像范围内、深度为正且不超过最大深度的点
        radar_valid = (
            (radar_points_cam[:, 2] > 0) &  # 深度为正
            (radar_points_cam[:, 2] <= max_depth) &  # 深度不超过最大值
            (radar_points_2d[:, 0] >= 0) & (radar_points_2d[:, 0] < img_w) &
            (radar_points_2d[:, 1] >= 0) & (radar_points_2d[:, 1] < img_h)
        )
        
        radar_points_cam = radar_points_cam[radar_valid]
        radar_points_2d = radar_points_2d[radar_valid]
    else:
        radar_points_cam = np.array([]).reshape(0, 3)
        radar_points_2d = np.array([]).reshape(0, 2)

    stats = {
        "frame_idx": frame_idx,
        "num_smpl": int(len(sampled_vertices)),
        "num_depth_valid": int(np.sum(valid_mask)),
        "num_radar": int(len(radar_points_cam)),
        "mean_err": float(np.mean(errors)) if len(errors) > 0 else 0.0,
        "std_err": float(np.std(errors)) if len(errors) > 0 else 0.0,
    }

    return {
        "frame_idx": frame_idx,
        "depth_map": depth_map,
        "smpl_points_2d": smpl_points_2d,
        "smpl_points_3d": sampled_vertices,
        "valid_mask": valid_mask,
        "smpl_points_valid": smpl_points_valid,
        "depth_points": depth_points,
        "radar_points_2d": radar_points_2d,
        "radar_points_3d": radar_points_cam,
        "errors": errors,
        "stats": stats,
    }


def _normalize_depth_for_vis(depth_map: np.ndarray) -> np.ndarray:
    """归一化深度图用于可视化"""
    valid = depth_map[depth_map > 0]
    if valid.size == 0:
        return np.zeros_like(depth_map, dtype=np.float32)
    d_min, d_max = np.percentile(valid, [2, 98])
    if d_max - d_min < 1e-6:
        d_max = d_min + 1e-6
    depth_norm = (depth_map - d_min) / (d_max - d_min)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    return depth_norm.astype(np.float32)


def draw_overlay(
    frame_data: Dict,
    output_path: str,
) -> None:
    """绘制2D叠加图：深度图 + SMPL点 + 深度点 + 雷达点"""
    depth_map = frame_data["depth_map"]
    smpl_points_2d = frame_data["smpl_points_2d"]
    valid_mask = frame_data["valid_mask"]
    radar_points_2d = frame_data["radar_points_2d"]
    stats = frame_data["stats"]

    # 深度图可视化
    depth_norm = _normalize_depth_for_vis(depth_map)
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )

    # 绘制SMPL采样点
    for i, (u, v) in enumerate(smpl_points_2d):
        u_int, v_int = int(round(u)), int(round(v))
        if not (0 <= u_int < depth_color.shape[1] and 0 <= v_int < depth_color.shape[0]):
            continue
        if valid_mask[i]:
            # 有效深度点：红色
            cv2.circle(depth_color, (u_int, v_int), 4, (0, 0, 255), -1)
        else:
            # 无效点：灰色
            cv2.circle(depth_color, (u_int, v_int), 3, (128, 128, 128), 1)

    # 绘制雷达点：绿色
    for u, v in radar_points_2d:
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < depth_color.shape[1] and 0 <= v_int < depth_color.shape[0]:
            cv2.circle(depth_color, (u_int, v_int), 5, (0, 255, 0), 2)
            cv2.circle(depth_color, (u_int, v_int), 2, (0, 255, 0), -1)

    # 添加文字说明
    text_lines = [
        f"Frame: {stats['frame_idx']:06d}",
        f"SMPL: {stats['num_smpl']} pts (Red: valid depth)",
        f"Depth: {stats['num_depth_valid']} valid",
        f"Radar: {stats['num_radar']} pts (Green)",
        f"Mean err: {stats['mean_err']*1000:.1f} mm",
    ]
    y0 = 28
    for line in text_lines:
        cv2.putText(
            depth_color,
            line,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        y0 += 28

    cv2.imwrite(output_path, depth_color)


def save_3d_plot(
    frame_data: Dict,
    output_path: str,
) -> None:
    """保存3D散点图：SMPL vs 深度 vs 雷达"""
    smpl_pts = frame_data["smpl_points_3d"]
    depth_pts = frame_data["depth_points"]
    radar_pts = frame_data["radar_points_3d"]
    stats = frame_data["stats"]

    # 坐标轴旋转：将相机坐标系的z轴（深度）转为x轴，让人物看起来是站立的
    def transform_coords(pts):
        if len(pts) == 0:
            return pts
        return np.stack([
            pts[:, 2],   # X = 原Z（深度）
            pts[:, 0],   # Y = 原X（右）
            -pts[:, 1],  # Z = -原Y（下变上）
        ], axis=1)

    smpl_pts_tf = transform_coords(smpl_pts)
    depth_pts_tf = transform_coords(depth_pts)
    radar_pts_tf = transform_coords(radar_pts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制SMPL点（红色）
    if len(smpl_pts_tf) > 0:
        ax.scatter(
            smpl_pts_tf[:, 0],
            smpl_pts_tf[:, 1],
            smpl_pts_tf[:, 2],
            s=10,
            c="red",
            alpha=0.6,
            label=f"SMPL ({len(smpl_pts_tf)} pts)",
            marker='o'
        )

    # 绘制深度点（蓝色）
    if len(depth_pts_tf) > 0:
        ax.scatter(
            depth_pts_tf[:, 0],
            depth_pts_tf[:, 1],
            depth_pts_tf[:, 2],
            s=10,
            c="deepskyblue",
            alpha=0.6,
            label=f"Depth ({len(depth_pts_tf)} pts)",
            marker='s'
        )

    # 绘制雷达点（绿色）
    if len(radar_pts_tf) > 0:
        ax.scatter(
            radar_pts_tf[:, 0],
            radar_pts_tf[:, 1],
            radar_pts_tf[:, 2],
            s=50,
            c="lime",
            alpha=0.8,
            label=f"Radar ({len(radar_pts_tf)} pts)",
            marker='^',
            edgecolors='darkgreen',
            linewidths=1
        )

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("X (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title(
        f"Frame {stats['frame_idx']:06d} | "
        f"SMPL: {stats['num_smpl']}, Depth: {stats['num_depth_valid']}, Radar: {stats['num_radar']}"
    )
    ax.legend(loc="upper right")
    ax.view_init(elev=15, azim=-70)
    ax.set_box_aspect([1, 1, 1])
    
    # 设置合适的显示范围
    all_pts = []
    if len(smpl_pts_tf) > 0:
        all_pts.append(smpl_pts_tf)
    if len(depth_pts_tf) > 0:
        all_pts.append(depth_pts_tf)
    if len(radar_pts_tf) > 0:
        all_pts.append(radar_pts_tf)
    
    if len(all_pts) > 0:
        all_pts = np.vstack(all_pts)
        x_range = [all_pts[:, 0].min(), all_pts[:, 0].max()]
        y_range = [all_pts[:, 1].min(), all_pts[:, 1].max()]
        z_range = [all_pts[:, 2].min(), all_pts[:, 2].max()]
        
        # 添加一些边距
        margin = 0.2
        ax.set_xlim(x_range[0] - margin, x_range[1] + margin)
        ax.set_ylim(y_range[0] - margin, y_range[1] + margin)
        ax.set_zlim(z_range[0] - margin, z_range[1] + margin)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = prepare_paths(parse_args())

    print("=" * 70)
    print("SMPL vs Depth vs Radar Visualization")
    print("=" * 70)
    print(f"SMPL dir : {args.smpl_dir}")
    print(f"Depth dir: {args.depth_dir}")
    print(f"Radar CSV: {args.radar_csv}")
    print(f"Intri    : {args.intri}")
    print(f"Extri    : {args.extri}")
    print(f"Model    : {args.model}")
    print(f"Camera   : {args.cam_id}")
    print(f"Output   : {args.output}")
    print(f"Samples  : {args.num_samples}")
    print(f"Depth scale: {args.depth_scale} m/count")
    print(f"Max depth: {args.max_depth} m (filters both depth and radar)")
    print(f"Radar min mag: {args.radar_min_mag}")

    print("\nLoading SMPL model ...")
    smpl_model = load_smpl_model(args.model)
    
    print("Loading camera parameters ...")
    K, R, T, dist = load_camera_params(args.intri, args.extri, cam_id=args.cam_id)
    
    print("Loading radar data ...")
    radar_df = load_radar_data(args.radar_csv)
    print(f"  Total radar frames: {len(radar_df['frame'].unique())}")
    print(f"  Total radar points: {len(radar_df)}")

    matched_frames = find_matched_frames(
        args.smpl_dir, args.depth_dir, radar_df, args.frame, args.limit
    )
    
    if len(matched_frames) == 0:
        print("No matched frames found. Please check paths or frame id.")
        return

    print(f"\nFound {len(matched_frames)} frame(s) to visualize: {matched_frames}")

    for frame_idx in matched_frames:
        print(f"\nProcessing frame {frame_idx:06d} ...")
        data = collect_frame_data(
            frame_idx=frame_idx,
            smpl_dir=args.smpl_dir,
            depth_dir=args.depth_dir,
            radar_df=radar_df,
            smpl_model=smpl_model,
            K=K,
            R=R,
            T=T,
            dist=dist,
            num_samples=args.num_samples,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
            radar_min_mag=args.radar_min_mag,
        )
        
        if data is None:
            print("  Skip: files missing.")
            continue

        overlay_path = os.path.join(args.output, f"{frame_idx:06d}_overlay.png")
        plot3d_path = os.path.join(args.output, f"{frame_idx:06d}_3d.png")

        draw_overlay(data, overlay_path)
        save_3d_plot(data, plot3d_path)

        print(
            f"  Saved overlay to {overlay_path}\n"
            f"  Saved 3D plot to {plot3d_path}\n"
            f"  SMPL: {data['stats']['num_smpl']} | "
            f"Depth: {data['stats']['num_depth_valid']} | "
            f"Radar: {data['stats']['num_radar']}"
        )

    print("\n完成！雷达坐标转换说明：")
    print("  假设雷达与深度相机位于同一位置")
    print("  雷达坐标系 -> 相机坐标系:")
    print("    cam_X = -radar_Y (雷达左 -> 相机右)")
    print("    cam_Y = -radar_Z (雷达上 -> 相机下)")
    print("    cam_Z =  radar_X (雷达前 -> 相机前)")
    print("  如果可视化结果不对，可能需要调整 transform_radar_to_camera() 函数中的转换关系")


if __name__ == "__main__":
    main()

