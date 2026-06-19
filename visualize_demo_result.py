#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from plyfile import PlyData, PlyElement

from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RobustVGGT demo outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/demo_result"),
        help="Input directory containing predictions_*.npz, or a specific predictions_*.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where visualization artifacts will be written. Defaults to <input-dir>/visualizations.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride used when subsampling points for the exported point cloud.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Upper bound on exported point count after subsampling.",
    )
    return parser.parse_args()


def resolve_prediction_input(input_path: Path) -> tuple[Path, Path]:
    input_path = input_path.expanduser()

    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"--input must point to a .npz file or a directory, got: {input_path}")
        return input_path, input_path.parent

    preferred = [
        input_path / "predictions_survived.npz",
        input_path / "predictions_first_forward.npz",
    ]
    for path in preferred:
        if path.exists():
            return path, input_path

    checked = "\n".join(str(path) for path in preferred)
    raise FileNotFoundError(
        "No prediction file found.\n"
        f"--input was: {input_path}\n"
        "Expected one of:\n"
        f"{checked}"
    )


def load_predictions(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def log_written(path: Path, label: str) -> None:
    print(f"Wrote {label}: {path}")


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim > 0 and arr.shape[0] == 1:
        return arr[0]
    return arr


def ensure_frame_first(arr: np.ndarray) -> np.ndarray:
    arr = squeeze_batch(arr)
    if arr.ndim == 3:
        return arr[None, ...]
    return arr


def to_uint8_image(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape {rgb.shape}")

    if rgb.shape[0] in (1, 3) and rgb.shape[-1] not in (1, 3):
        rgb = np.transpose(rgb, (1, 2, 0))
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)

    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def percentile_normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(values)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(values)

    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(mask):
        return out

    valid = values[mask]
    lo = np.percentile(valid, 2.0)
    hi = np.percentile(valid, 98.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6

    out[mask] = np.clip((values[mask] - lo) / (hi - lo), 0.0, 1.0)
    return out


def colorize_scalar_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    normalized = percentile_normalize(values, mask)

    anchors = np.array(
        [
            [0.00, 0.07, 0.09, 0.38],
            [0.20, 0.10, 0.42, 0.74],
            [0.40, 0.25, 0.72, 0.54],
            [0.60, 0.83, 0.85, 0.26],
            [0.80, 0.98, 0.58, 0.18],
            [1.00, 0.76, 0.07, 0.05],
        ],
        dtype=np.float32,
    )
    xs = anchors[:, 0]
    colors = anchors[:, 1:]

    flat = normalized.reshape(-1)
    mapped = np.empty((flat.shape[0], 3), dtype=np.float32)
    for channel in range(3):
        mapped[:, channel] = np.interp(flat, xs, colors[:, channel])
    mapped = mapped.reshape(normalized.shape + (3,))

    if mask is not None:
        mapped = np.where(mask[..., None], mapped, 0.0)

    return (mapped * 255.0).round().astype(np.uint8)


def add_title(image: np.ndarray, title: str, index: int) -> Image.Image:
    pil = Image.fromarray(image)
    canvas = Image.new("RGB", (pil.width, pil.height + 26), "white")
    canvas.paste(pil, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"[{index:02d}] {title}", fill="black")
    return canvas


def save_contact_sheet(frames: Iterable[np.ndarray], title: str, out_path: Path, ncols: int = 4) -> None:
    tiles = [add_title(frame, title, idx) for idx, frame in enumerate(frames)]
    if not tiles:
        return

    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    ncols = max(1, min(ncols, len(tiles)))
    nrows = math.ceil(len(tiles) / ncols)
    sheet = Image.new("RGB", (ncols * tile_w, nrows * tile_h), "white")

    for idx, tile in enumerate(tiles):
        x = (idx % ncols) * tile_w
        y = (idx // ncols) * tile_h
        sheet.paste(tile, (x, y))

    sheet.save(out_path)
    log_written(out_path, f"{title} contact sheet")


def camera_centers_from_pose_enc(pose_enc: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    pose_tensor = torch.from_numpy(np.asarray(pose_enc, dtype=np.float32))
    if pose_tensor.ndim == 2:
        pose_tensor = pose_tensor.unsqueeze(0)
    extrinsics, _ = pose_encoding_to_extri_intri(pose_tensor, image_hw)
    extrinsics = extrinsics.squeeze(0).cpu().numpy()

    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)
    return centers.astype(np.float32)


def save_camera_plot(camera_centers: np.ndarray, out_path: Path) -> None:
    size = 960
    margin = 80
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)

    xs = camera_centers[:, 0]
    zs = camera_centers[:, 2]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())

    span_x = max(max_x - min_x, 1e-6)
    span_z = max(max_z - min_z, 1e-6)
    scale = min((size - 2 * margin) / span_x, (size - 2 * margin) / span_z)

    points = []
    for x, z in zip(xs, zs):
        px = margin + (x - min_x) * scale
        py = size - margin - (z - min_z) * scale
        points.append((px, py))

    draw.rectangle((margin, margin, size - margin, size - margin), outline=(180, 180, 180), width=2)
    if len(points) > 1:
        draw.line(points, fill=(40, 110, 220), width=3)

    for idx, (px, py) in enumerate(points):
        radius = 7 if idx == 0 else 5
        color = (220, 60, 60) if idx == 0 else (30, 30, 30)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
        draw.text((px + 8, py - 8), str(idx), fill=(20, 20, 20))

    draw.text((margin, 20), "Camera trajectory (top-down X/Z)", fill="black")
    canvas.save(out_path)
    log_written(out_path, "camera trajectory image")


def save_point_cloud_preview(points: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centered = points - np.median(points, axis=0, keepdims=True)
    dist = np.linalg.norm(centered, axis=1)
    keep = dist <= np.percentile(dist, 98.0)
    points = centered[keep]
    colors = colors[keep]

    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors.astype(np.float32) / 255.0,
        s=0.4,
        linewidths=0,
    )
    ax.set_title("Sparse point cloud preview")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-62)
    ax.set_box_aspect(np.ptp(points, axis=0) + 1e-6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    log_written(out_path, "point cloud preview")


def flatten_depth_frames(depth: np.ndarray) -> np.ndarray:
    depth = ensure_frame_first(depth)
    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim == 4 and depth.shape[-1] == 3:
        depth = np.linalg.norm(depth, axis=-1)
    return depth


def flatten_conf_frames(conf: np.ndarray) -> np.ndarray:
    conf = ensure_frame_first(conf)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    elif conf.ndim == 4 and conf.shape[1] == 1:
        conf = conf[:, 0]
    return conf


def flatten_world_points_frames(world_points: np.ndarray) -> np.ndarray:
    world_points = ensure_frame_first(world_points)
    if world_points.ndim == 5 and world_points.shape[-1] == 3:
        return squeeze_batch(world_points)
    if world_points.ndim == 5 and world_points.shape[2] == 3:
        return np.transpose(squeeze_batch(world_points), (0, 2, 3, 1))
    if world_points.ndim == 4 and world_points.shape[-1] == 3:
        return world_points
    if world_points.ndim == 4 and world_points.shape[1] == 3:
        return np.transpose(world_points, (0, 2, 3, 1))
    raise ValueError(f"Unsupported world_points shape: {world_points.shape}")


def export_sparse_point_cloud(
    depth: np.ndarray | None,
    images: np.ndarray | None,
    pose_enc: np.ndarray,
    world_points: np.ndarray | None,
    out_path: Path,
    stride: int,
    max_points: int,
) -> list[Path]:
    if world_points is not None:
        world = flatten_world_points_frames(world_points)
        world = world[:, ::stride, ::stride]
    else:
        if depth is None:
            return []
        depth = flatten_depth_frames(depth)
        if depth.ndim != 3:
            return []

        image_hw = (int(depth.shape[1]), int(depth.shape[2]))
        pose_tensor = torch.from_numpy(np.asarray(pose_enc, dtype=np.float32))
        if pose_tensor.ndim == 2:
            pose_tensor = pose_tensor.unsqueeze(0)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_tensor, image_hw)
        world = unproject_depth_map_to_point_map(depth[..., None], extrinsics.squeeze(0), intrinsics.squeeze(0))
        world = world[:, ::stride, ::stride]

    if images is not None:
        rgb = ensure_frame_first(images)
        rgb = np.stack([to_uint8_image(frame) for frame in rgb], axis=0)
        rgb = rgb[:, ::stride, ::stride]
    else:
        rgb = None

    points = world.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
    points = points[valid]
    if points.size == 0:
        return []

    if rgb is not None:
        colors = rgb.reshape(-1, 3)[valid]
    else:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)

    if points.shape[0] > max_points:
        step = math.ceil(points.shape[0] / max_points)
        points = points[::step]
        colors = colors[::step]

    vertex = np.empty(
        points.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(out_path)
    log_written(out_path, "sparse point cloud")
    preview_path = out_path.with_name("point_cloud_preview.png")
    save_point_cloud_preview(points, colors, preview_path)
    return [out_path, preview_path]


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists() and input_path == Path("output/demo_result") and Path("demo_result").exists():
        input_path = Path("demo_result")

    prediction_path, result_dir = resolve_prediction_input(input_path)
    output_dir = args.output_dir or (result_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resolved input prediction file: {prediction_path}")
    print(f"Writing outputs under: {output_dir}")

    predictions = load_predictions(prediction_path)

    images = predictions.get("images")
    world_points = predictions.get("world_points")
    if images is not None:
        rgb_frames = [to_uint8_image(frame) for frame in ensure_frame_first(images)]
        save_contact_sheet(rgb_frames, "RGB", output_dir / "rgb_grid.jpg")

    depth = predictions.get("depth")
    if depth is not None:
        depth_frames = flatten_depth_frames(depth)
        if depth_frames.ndim == 3:
            depth_rgb = [colorize_scalar_map(frame, np.isfinite(frame) & (frame > 0)) for frame in depth_frames]
            save_contact_sheet(depth_rgb, "Depth", output_dir / "depth_grid.jpg")

    conf = predictions.get("depth_conf")
    if conf is not None:
        conf_frames = flatten_conf_frames(conf)
        if conf_frames.ndim == 3:
            conf_rgb = [colorize_scalar_map(frame, np.isfinite(frame)) for frame in conf_frames]
            save_contact_sheet(conf_rgb, "Confidence", output_dir / "confidence_grid.jpg")

    pose_enc = predictions.get("pose_enc")
    if pose_enc is not None:
        if depth is not None:
            depth_frames = flatten_depth_frames(depth)
            image_hw = (int(depth_frames.shape[1]), int(depth_frames.shape[2]))
        elif images is not None:
            sample_rgb = to_uint8_image(ensure_frame_first(images)[0])
            image_hw = (sample_rgb.shape[0], sample_rgb.shape[1])
        else:
            image_hw = (518, 518)

        camera_centers = camera_centers_from_pose_enc(squeeze_batch(pose_enc), image_hw)
        save_camera_plot(camera_centers, output_dir / "camera_trajectory_topdown.png")

        if depth is not None:
            written = export_sparse_point_cloud(
                depth=depth,
                images=images,
                pose_enc=squeeze_batch(pose_enc),
                world_points=world_points,
                out_path=output_dir / "sparse_point_cloud.ply",
                stride=max(1, args.stride),
                max_points=max(1, args.max_points),
            )
            if not written:
                print("Skipped sparse point cloud export: no valid 3D points were produced.")

    print("Finished visualization export.")


if __name__ == "__main__":
    main()
