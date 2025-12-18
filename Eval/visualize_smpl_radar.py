#!/usr/bin/env python3
"""
可视化 SMPL 和雷达点云的对比。

功能:
- 为每一帧采样前向可见的 SMPL 顶点并投影到图像平面
- 读取雷达点云数据（球坐标），转换为相机坐标系 3D 点并投影
- 生成两类可视化:
    1) 图像上的 2D 叠加图 (SMPL 点 vs 雷达点)
    2) 3D 散点图 (红: SMPL, 绿: 雷达)
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

# 让 easymocap 模块可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 复用 evaluate_depth.py 中的工具函数
from evaluate_depth import (
    get_smpl_vertices,
    load_camera_params,
    load_smpl_model,
    load_smpl_params,
    project_to_image,
    select_front_facing_vertices,
    transform_to_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 SMPL 和雷达点云的可视化对比"
    )
    parser.add_argument(
        "--root",
        default="/home/bupt630/Dabai/AmmWave/EasyMocap",
        help="项目根目录",
    )
    parser.add_argument(
        "--smpl-dir",
        default=None,
        help="SMPL 参数 (json) 路径",
    )
    parser.add_argument(
        "--radar-csv",
        default=None,
        help="雷达点云 CSV 文件路径",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="背景图像目录（可选），用于2D可视化的背景",
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
        "--max-depth",
        type=float,
        default=10.0,
        help="深度上限 (米)，用于过滤过远的点",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录，默认为 {root}/Eval/vis_smpl_radar",
    )
    parser.add_argument(
        "--radar-min-mag",
        type=float,
        default=0.0,
        help="雷达点最小幅值阈值，低于此值的点将被过滤",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=1280,
        help="图像宽度（用于2D可视化）",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=720,
        help="图像高度（用于2D可视化）",
    )
    return parser.parse_args()


def prepare_paths(args: argparse.Namespace) -> argparse.Namespace:
    """补全默认路径并创建输出目录。"""
    if args.smpl_dir is None:
        args.smpl_dir = os.path.join(
            args.root, "output", "detect_triangulate_fitSMPL", "smpl"
        )
    if args.radar_csv is None:
        args.radar_csv = os.path.join(args.root, "Eval", "pc_cfar.csv")
    if args.image_dir is None:
        args.image_dir = os.path.join(
            args.root, "data", "examples", "my_multiview", "images", args.cam_id
        )
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
        args.output = os.path.join(args.root, "Eval", "vis_smpl_radar")
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
    """
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    
    x = range_m * np.cos(el_rad) * np.cos(az_rad)
    y = range_m * np.cos(el_rad) * np.sin(az_rad)
    z = range_m * np.sin(el_rad)
    
    return np.stack([x, y, z], axis=1)


def transform_radar_to_camera(radar_points: np.ndarray) -> np.ndarray:
    """
    将雷达坐标系转换到相机坐标系
    
    假设雷达和深度相机在同一位置，但坐标系定义不同：
    - 雷达坐标系: X前, Y左, Z上
    - 相机坐标系: X右, Y下, Z前
    """
    cam_points = np.zeros_like(radar_points)
    cam_points[:, 0] = -radar_points[:, 1]  # cam_X = -radar_Y
    cam_points[:, 1] = -radar_points[:, 2]  # cam_Y = -radar_Z
    cam_points[:, 2] = radar_points[:, 0]   # cam_Z = radar_X
    
    return cam_points


def find_matched_frames(
    smpl_dir: str, radar_df: pd.DataFrame, frame: Optional[int], limit: int
) -> List[int]:
    """找到SMPL和雷达数据都存在的帧"""
    smpl_files = [
        int(os.path.splitext(os.path.basename(f))[0])
        for f in sorted(os.listdir(smpl_dir))
        if f.endswith(".json")
    ]
    radar_frames = radar_df['frame'].unique().tolist()
    
    matched = sorted(list(set(smpl_files) & set(radar_frames)))
    
    if frame is not None:
        matched = [f for f in matched if f == frame]
    if limit is not None and limit > 0:
        matched = matched[:limit]
    return matched


def collect_frame_data(
    frame_idx: int,
    smpl_dir: str,
    radar_df: pd.DataFrame,
    image_dir: str,
    smpl_model,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    dist: np.ndarray,
    num_samples: int,
    max_depth: float,
    radar_min_mag: float,
    img_width: int,
    img_height: int,
) -> Optional[Dict]:
    """
    收集单帧的SMPL和雷达数据
    
    注：max_depth 参数会同时过滤SMPL点和雷达点，
    只保留深度值在 (0, max_depth] 范围内的点
    """
    smpl_path = os.path.join(smpl_dir, f"{frame_idx:06d}.json")
    
    if not os.path.exists(smpl_path):
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
    
    # 过滤深度超过max_depth的SMPL点
    smpl_depth_valid = (sampled_vertices[:, 2] > 0) & (sampled_vertices[:, 2] <= max_depth)
    sampled_vertices = sampled_vertices[smpl_depth_valid]
    
    smpl_points_2d = project_to_image(sampled_vertices, K, dist)
    
    # 进一步过滤在图像范围内的点
    smpl_image_valid = (
        (smpl_points_2d[:, 0] >= 0) & (smpl_points_2d[:, 0] < img_width) &
        (smpl_points_2d[:, 1] >= 0) & (smpl_points_2d[:, 1] < img_height)
    )
    sampled_vertices = sampled_vertices[smpl_image_valid]
    smpl_points_2d = smpl_points_2d[smpl_image_valid]

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
            (radar_points_cam[:, 2] > 0) &
            (radar_points_cam[:, 2] <= max_depth) &
            (radar_points_2d[:, 0] >= 0) & (radar_points_2d[:, 0] < img_width) &
            (radar_points_2d[:, 1] >= 0) & (radar_points_2d[:, 1] < img_height)
        )
        
        radar_points_cam = radar_points_cam[radar_valid]
        radar_points_2d = radar_points_2d[radar_valid]
    else:
        radar_points_cam = np.array([]).reshape(0, 3)
        radar_points_2d = np.array([]).reshape(0, 2)

    # 尝试加载背景图像
    background_img = None
    if image_dir and os.path.isdir(image_dir):
        # 尝试多种可能的图像文件名格式
        possible_names = [
            f"{frame_idx:06d}.jpg",
            f"{frame_idx:06d}.png",
        ]
        for name in possible_names:
            img_path = os.path.join(image_dir, name)
            if os.path.exists(img_path):
                background_img = cv2.imread(img_path)
                break

    stats = {
        "frame_idx": frame_idx,
        "num_smpl": int(len(sampled_vertices)),
        "num_radar": int(len(radar_points_cam)),
    }

    return {
        "frame_idx": frame_idx,
        "background_img": background_img,
        "smpl_points_2d": smpl_points_2d,
        "smpl_points_3d": sampled_vertices,
        "radar_points_2d": radar_points_2d,
        "radar_points_3d": radar_points_cam,
        "stats": stats,
        "img_width": img_width,
        "img_height": img_height,
    }


def draw_overlay(
    frame_data: Dict,
    output_path: str,
) -> None:
    """绘制2D叠加图：SMPL点 + 雷达点"""
    background_img = frame_data["background_img"]
    smpl_points_2d = frame_data["smpl_points_2d"]
    radar_points_2d = frame_data["radar_points_2d"]
    stats = frame_data["stats"]
    img_width = frame_data["img_width"]
    img_height = frame_data["img_height"]

    # 创建画布
    if background_img is not None:
        canvas = background_img.copy()
    else:
        # 创建黑色背景
        canvas = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # 绘制SMPL采样点（红色）
    for u, v in smpl_points_2d:
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < canvas.shape[1] and 0 <= v_int < canvas.shape[0]:
            cv2.circle(canvas, (u_int, v_int), 4, (0, 0, 255), -1)

    # 绘制雷达点（绿色，更大更显眼）
    for u, v in radar_points_2d:
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < canvas.shape[1] and 0 <= v_int < canvas.shape[0]:
            cv2.circle(canvas, (u_int, v_int), 6, (0, 255, 0), 2)
            cv2.circle(canvas, (u_int, v_int), 2, (0, 255, 0), -1)

    # 添加文字说明
    text_lines = [
        f"Frame: {stats['frame_idx']:06d}",
        f"SMPL: {stats['num_smpl']} pts (Red)",
        f"Radar: {stats['num_radar']} pts (Green)",
    ]
    y0 = 28
    for line in text_lines:
        cv2.putText(
            canvas,
            line,
            (20, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        y0 += 28

    cv2.imwrite(output_path, canvas)


def save_3d_plot(
    frame_data: Dict,
    output_path: str,
) -> None:
    """保存3D散点图：SMPL vs 雷达"""
    smpl_pts = frame_data["smpl_points_3d"]
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
    radar_pts_tf = transform_coords(radar_pts)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制SMPL点（红色）
    if len(smpl_pts_tf) > 0:
        ax.scatter(
            smpl_pts_tf[:, 0],
            smpl_pts_tf[:, 1],
            smpl_pts_tf[:, 2],
            s=15,
            c="red",
            alpha=0.7,
            label=f"SMPL ({len(smpl_pts_tf)} pts)",
            marker='o'
        )

    # 绘制雷达点（绿色）
    if len(radar_pts_tf) > 0:
        ax.scatter(
            radar_pts_tf[:, 0],
            radar_pts_tf[:, 1],
            radar_pts_tf[:, 2],
            s=60,
            c="lime",
            alpha=0.9,
            label=f"Radar ({len(radar_pts_tf)} pts)",
            marker='^',
            edgecolors='darkgreen',
            linewidths=1.5
        )

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("X (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title(
        f"Frame {stats['frame_idx']:06d} | "
        f"SMPL: {stats['num_smpl']}, Radar: {stats['num_radar']}"
    )
    ax.legend(loc="upper right")
    ax.view_init(elev=15, azim=-70)
    ax.set_box_aspect([1, 1, 1])
    
    # 设置合适的显示范围
    all_pts = []
    if len(smpl_pts_tf) > 0:
        all_pts.append(smpl_pts_tf)
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
    print("SMPL vs Radar Visualization")
    print("=" * 70)
    print(f"SMPL dir : {args.smpl_dir}")
    print(f"Radar CSV: {args.radar_csv}")
    print(f"Image dir: {args.image_dir}")
    print(f"Intri    : {args.intri}")
    print(f"Extri    : {args.extri}")
    print(f"Model    : {args.model}")
    print(f"Camera   : {args.cam_id}")
    print(f"Output   : {args.output}")
    print(f"Samples  : {args.num_samples}")
    print(f"Max depth: {args.max_depth} m (filters both SMPL and radar)")
    print(f"Radar min mag: {args.radar_min_mag}")
    print(f"Image size: {args.img_width}x{args.img_height}")

    print("\nLoading SMPL model ...")
    smpl_model = load_smpl_model(args.model)
    
    print("Loading camera parameters ...")
    K, R, T, dist = load_camera_params(args.intri, args.extri, cam_id=args.cam_id)
    
    print("Loading radar data ...")
    radar_df = load_radar_data(args.radar_csv)
    print(f"  Total radar frames: {len(radar_df['frame'].unique())}")
    print(f"  Total radar points: {len(radar_df)}")

    matched_frames = find_matched_frames(
        args.smpl_dir, radar_df, args.frame, args.limit
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
            radar_df=radar_df,
            image_dir=args.image_dir,
            smpl_model=smpl_model,
            K=K,
            R=R,
            T=T,
            dist=dist,
            num_samples=args.num_samples,
            max_depth=args.max_depth,
            radar_min_mag=args.radar_min_mag,
            img_width=args.img_width,
            img_height=args.img_height,
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
            f"Radar: {data['stats']['num_radar']}"
        )

    print("\n完成！雷达坐标转换说明：")
    print("  假设雷达与相机位于同一位置")
    print("  雷达坐标系 -> 相机坐标系:")
    print("    cam_X = -radar_Y (雷达左 -> 相机右)")
    print("    cam_Y = -radar_Z (雷达上 -> 相机下)")
    print("    cam_Z =  radar_X (雷达前 -> 相机前)")
    print("  如果可视化结果不对，可能需要调整 transform_radar_to_camera() 函数")


if __name__ == "__main__":
    main()

