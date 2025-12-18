#!/usr/bin/env python3
"""
可视化 SMPL 采样点与深度图对应点的对比。

功能:
- 为每一帧采样前向可见的 SMPL 顶点并投影到图像平面
- 读取 RealSense 深度，在采样点位置取深度并反投影为相机坐标系 3D 点
- 生成两类可视化:
    1) 深度图上的 2D 误差分布叠加图 (按误差着色)
    2) 3D 散点/连线图 (红: SMPL, 蓝: 深度，灰线显示偏差)

默认数据路径与 evaluate_depth.py 保持一致，可用参数修改。
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

import matplotlib

matplotlib.use("Agg")  # 使用无窗口后端，便于在服务器上保存图片
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

# 让 easymocap 模块可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 复用 evaluate_depth.py 中的工具函数，避免重复实现
from evaluate_depth import (  # noqa: E402
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
        description="生成 SMPL 点与深度点的可视化对比"
    )
    parser.add_argument(
        "--root",
        default="/home/bupt630/Dabai/AmmWave/EasyMocap",
        help="项目根目录 (默认与 evaluate_depth.py 一致)",
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
        "--intri",
        default=None,
        help="相机内参文件 intri.yml，默认为 {root}/data/examples/my_multiview/intri.yml",
    )
    parser.add_argument(
        "--extri",
        default=None,
        help="相机外参文件 extri.yml，默认为 {root}/data/examples/my_multiview/extri.yml",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="SMPL 模型文件 (.pkl)，默认为 {root}/models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl",
    )
    parser.add_argument(
        "--cam-id",
        default="cam8",
        help="使用的相机编号 (与 intri/extri 中的键一致)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="只可视化指定帧编号 (六位数字)，不指定则批量处理匹配到的帧",
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
        help="深度单位 (米/计数)，RealSense 默认 0.0002，可根据数据调整",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="深度上限 (米)，超过该距离的点视为无效",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出目录，默认为 {root}/Eval/vis_smpl_depth",
    )
    parser.add_argument(
        "--line-count",
        type=int,
        default=300,
        help="3D 连线可视化的抽样线条数量 (过大会变得杂乱)",
    )
    return parser.parse_args()


def prepare_paths(args: argparse.Namespace) -> argparse.Namespace:
    """补全默认路径并创建输出目录。"""
    if args.smpl_dir is None:
        args.smpl_dir = os.path.join(args.root, "output","detect_triangulate_fitSMPL--eval","smpl")
    if args.depth_dir is None:
        args.depth_dir = os.path.join(args.root, "Evaluation", "depth")
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
        args.output = os.path.join(args.root, "Eval", "vis_smpl_depth")
    os.makedirs(args.output, exist_ok=True)
    return args


def find_matched_frames(
    smpl_dir: str, depth_dir: str, frame: Optional[int], limit: int
) -> List[int]:
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
    matched = sorted(list(set(smpl_files) & set(depth_files)))
    if frame is not None:
        matched = [f for f in matched if f == frame]
    if limit is not None and limit > 0:
        matched = matched[:limit]
    return matched


def collect_frame_points(
    frame_idx: int,
    smpl_dir: str,
    depth_dir: str,
    smpl_model,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    dist: np.ndarray,
    num_samples: int,
    depth_scale: float,
    max_depth: float,
) -> Optional[Dict]:
    smpl_path = os.path.join(smpl_dir, f"{frame_idx:06d}.json")
    depth_path = os.path.join(depth_dir, f"{frame_idx:06d}.png")
    if not (os.path.exists(smpl_path) and os.path.exists(depth_path)):
        return None

    params = load_smpl_params(smpl_path)
    if params is None:
        return None

    vertices_world = get_smpl_vertices(smpl_model, params)
    vertices_cam = transform_to_camera(vertices_world, R, T)

    sampled_idx = select_front_facing_vertices(
        vertices_cam, smpl_model, num_samples=num_samples
    )
    sampled_vertices = vertices_cam[sampled_idx]
    points_2d = project_to_image(sampled_vertices, K, dist)

    depth_map = read_depth_image(depth_path, depth_scale)
    img_h, img_w = depth_map.shape[:2]

    valid_mask = np.zeros(len(points_2d), dtype=bool)
    depth_values = np.zeros(len(points_2d), dtype=np.float32)

    for i, (u, v) in enumerate(points_2d):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < img_w and 0 <= v_int < img_h:
            depth_val = float(depth_map[v_int, u_int])
            if 0 < depth_val <= max_depth:
                valid_mask[i] = True
                depth_values[i] = depth_val

    if not np.any(valid_mask):
        return None

    valid_indices = np.where(valid_mask)[0]
    depth_points = backproject_depth_to_3d(
        points_2d[valid_mask], depth_values[valid_mask], K
    )
    smpl_points_valid = sampled_vertices[valid_mask]
    errors = np.linalg.norm(depth_points - smpl_points_valid, axis=1)

    stats = {
        "frame_idx": frame_idx,
        "num_valid": int(len(errors)),
        "mean_err": float(np.mean(errors)),
        "std_err": float(np.std(errors)),
        "p5": float(np.percentile(errors, 5)),
        "p95": float(np.percentile(errors, 95)),
        "max_err": float(np.max(errors)),
    }

    return {
        "frame_idx": frame_idx,
        "depth_map": depth_map,
        "points_2d": points_2d,
        "valid_mask": valid_mask,
        "sampled_vertices": sampled_vertices,
        "smpl_points_valid": smpl_points_valid,
        "depth_points": depth_points,
        "errors": errors,
        "stats": stats,
        "valid_indices": valid_indices,
    }


def _normalize_depth_for_vis(depth_map: np.ndarray) -> np.ndarray:
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
    depth_map = frame_data["depth_map"]
    points_2d = frame_data["points_2d"]
    valid_mask = frame_data["valid_mask"]
    errors = frame_data["errors"]
    valid_indices = frame_data["valid_indices"]
    stats = frame_data["stats"]

    depth_norm = _normalize_depth_for_vis(depth_map)
    depth_color = cv2.applyColorMap(
        (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS
    )

    error_lookup = {idx: err for idx, err in zip(valid_indices, errors)}
    if errors.size > 0:
        error_max = max(float(np.percentile(errors, 95)), 1e-6)
    else:
        error_max = 1.0
    cmap = cm.get_cmap("coolwarm")

    for i, (u, v) in enumerate(points_2d):
        u_int, v_int = int(round(u)), int(round(v))
        if not (0 <= u_int < depth_color.shape[1] and 0 <= v_int < depth_color.shape[0]):
            continue

        if valid_mask[i]:
            err = error_lookup.get(i, 0.0)
            e_norm = np.clip(err / error_max, 0.0, 1.0)
            color = cmap(e_norm)
            cv_color = (
                int(color[2] * 255),
                int(color[1] * 255),
                int(color[0] * 255),
            )
            cv2.circle(depth_color, (u_int, v_int), 3, cv_color, -1)
        else:
            cv2.circle(depth_color, (u_int, v_int), 3, (180, 180, 180), 1)

    text_lines = [
        f"Frame: {stats['frame_idx']:06d}",
        f"Valid pts: {stats['num_valid']}",
        f"Mean err: {stats['mean_err']*1000:.1f} mm",
        f"90% CI: [{stats['p5']*1000:.1f}, {stats['p95']*1000:.1f}] mm",
        f"Max err: {stats['max_err']*1000:.1f} mm",
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
    line_count: int,
) -> None:
    smpl_pts = frame_data["smpl_points_valid"]
    depth_pts = frame_data["depth_points"]
    stats = frame_data["stats"]

    # 坐标轴旋转：将相机坐标系的z轴（深度）转为x轴，让人物看起来是站立的
    # 原坐标系：X右，Y下，Z前（深度）
    # 新坐标系：X=原Z（深度），Y=原X（右），Z=-原Y（下变上）
    def transform_coords(pts):
        return np.stack([
            pts[:, 2],   # X = 原Z（深度）
            pts[:, 0],   # Y = 原X（右）
            -pts[:, 1],  # Z = -原Y（下变上）
        ], axis=1)

    smpl_pts_tf = transform_coords(smpl_pts)
    depth_pts_tf = transform_coords(depth_pts)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        smpl_pts_tf[:, 0],
        smpl_pts_tf[:, 1],
        smpl_pts_tf[:, 2],
        s=4,
        c="red",
        alpha=0.8,
        label="SMPL (camera frame)",
    )
    ax.scatter(
        depth_pts_tf[:, 0],
        depth_pts_tf[:, 1],
        depth_pts_tf[:, 2],
        s=4,
        c="deepskyblue",
        alpha=0.8,
        label="Depth (camera frame)",
    )

    if line_count is not None and line_count > 0:
        step = max(1, len(depth_pts) // line_count)
        for i in range(0, len(depth_pts), step):
            ax.plot(
                [smpl_pts_tf[i, 0], depth_pts_tf[i, 0]],
                [smpl_pts_tf[i, 1], depth_pts_tf[i, 1]],
                [smpl_pts_tf[i, 2], depth_pts_tf[i, 2]],
                c="gray",
                lw=0.6,
                alpha=0.6,
            )

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("X (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title(
        f"Frame {stats['frame_idx']:06d} | mean err {stats['mean_err']*1000:.1f} mm"
    )
    ax.legend(loc="upper right")
    ax.view_init(elev=15, azim=-70)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = prepare_paths(parse_args())

    print("=" * 70)
    print("SMPL vs Depth Visualization")
    print("=" * 70)
    print(f"SMPL dir : {args.smpl_dir}")
    print(f"Depth dir: {args.depth_dir}")
    print(f"Intri    : {args.intri}")
    print(f"Extri    : {args.extri}")
    print(f"Model    : {args.model}")
    print(f"Camera   : {args.cam_id}")
    print(f"Output   : {args.output}")
    print(f"Samples  : {args.num_samples}")
    print(f"Depth scale: {args.depth_scale} m/count")

    print("\nLoading SMPL model ...")
    smpl_model = load_smpl_model(args.model)
    print("Loading camera parameters ...")
    K, R, T, dist = load_camera_params(args.intri, args.extri, cam_id=args.cam_id)

    matched_frames = find_matched_frames(
        args.smpl_dir, args.depth_dir, args.frame, args.limit
    )
    if len(matched_frames) == 0:
        print("No matched frames found. Please check paths or frame id.")
        return

    print(f"Found {len(matched_frames)} frame(s) to visualize: {matched_frames}")

    for frame_idx in matched_frames:
        print(f"\nProcessing frame {frame_idx:06d} ...")
        data = collect_frame_points(
            frame_idx=frame_idx,
            smpl_dir=args.smpl_dir,
            depth_dir=args.depth_dir,
            smpl_model=smpl_model,
            K=K,
            R=R,
            T=T,
            dist=dist,
            num_samples=args.num_samples,
            depth_scale=args.depth_scale,
            max_depth=args.max_depth,
        )
        if data is None:
            print("  Skip: no valid points or files missing.")
            continue

        overlay_path = os.path.join(args.output, f"{frame_idx:06d}_overlay.png")
        plot3d_path = os.path.join(args.output, f"{frame_idx:06d}_3d.png")

        draw_overlay(data, overlay_path)
        save_3d_plot(data, plot3d_path, line_count=args.line_count)

        print(
            f"  Saved overlay to {overlay_path}\n"
            f"  Saved 3D plot to {plot3d_path}\n"
            f"  Mean error: {data['stats']['mean_err']*1000:.2f} mm | "
            f"Valid points: {data['stats']['num_valid']}"
        )

    print("\nDone. 请在输出目录查看可视化结果。")


if __name__ == "__main__":
    main()

