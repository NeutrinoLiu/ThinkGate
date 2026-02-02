"""
AnySplat TUM Novel View Synthesis Evaluation Script

This script evaluates AnySplat's performance on TUM NVS using explicit context/test
lists provided in a JSON index:
1. Context images are specified by "full_view" or "th0p5" lists
2. Test images are specified by the "test_view" list
3. Evaluation: PSNR/SSIM/LPIPS metrics for direct predictions, qualitative video for NVS
"""

import os
from pathlib import Path
import sys
import json
import argparse
import csv

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image, save_interpolated_video
from src.utils.image import process_image

from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

def setup_args():
    """Set up command-line arguments for the eval TUM script."""
    parser = argparse.ArgumentParser(description='Test AnySplat on TUM NVS evaluation')
    parser.add_argument('--tum_index_json', type=str, default="data_index/tum_nvs_updated.json",
                       help='Optional: Path to TUM JSON index (e.g., data_index/tum_nvs_updated.json). '
                            'If provided, runs evaluation for each TUM scene with full_view and th0p5 contexts.')
    parser.add_argument('--tum_root', type=str, default="dataset/tum/tum",
                       help='Root directory for TUM dataset folders')
    parser.add_argument('--output_root', type=str, default="outputs/tum",
                       help='Output root for TUM JSON evaluation runs')
    parser.add_argument('--test_holdout', type=int, default=10,
                       help='Holdout stride for test images: keep 1 of every N test images (default: 10)')
    parser.add_argument('--stride', type=int, default=0,
                       help='Optional context stride for full_view: keep 1 of every N images. '
                            'When set, runs an additional stride_n experiment.')
    parser.add_argument('--time_window_sec', type=float, default=10,
                       help='Limit context/test images to first N seconds based on timestamps in filenames. '
                            'Set to 0 to disable.')
    return parser.parse_args()

def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image, image)
    return psnr, ssim, lpips

def log_evaluation_results(exp_name, num_context, num_test, psnr, ssim, lpips, test_holdout, duration, csv_file="evaluation_results.csv"):
    """Log evaluation results to a CSV file."""
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['experiment_name', 'num_context_views', 'num_test_views', 'psnr', 'ssim', 'lpips', 'test_holdout', 'duration'])

        # Write the results
        writer.writerow([
            exp_name,
            num_context,
            num_test,
            f"{psnr:.2f}",
            f"{ssim:.3f}",
            f"{lpips:.3f}",
            test_holdout,
            duration
        ])

    sort_csv_by_experiment_name(csv_file)

def get_completed_experiments(csv_file="evaluation_results.csv") -> set:
    if not os.path.isfile(csv_file):
        return set()
    completed = set()
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                completed.add(row[0])
    return completed

def sort_csv_by_experiment_name(csv_file: str) -> None:
    if not os.path.isfile(csv_file):
        return
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        return
    data_rows.sort(key=lambda row: row[0] if row else "")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)

def resolve_image_folder(data_dir: str) -> str:
    if os.path.exists(os.path.join(data_dir, "images")):
        return os.path.join(data_dir, "images")
    if os.path.exists(os.path.join(data_dir, "rgb")):
        return os.path.join(data_dir, "rgb")
    return data_dir

def resolve_image_paths(image_folder: str, data_dir: str, filenames: list, label: str) -> list:
    image_paths = []
    missing = []
    for name in filenames:
        if os.path.isabs(name):
            path = name
        elif name.startswith("rgb/") or name.startswith("rgb\\"):
            path = os.path.join(data_dir, name)
        else:
            path = os.path.join(image_folder, name)
        if not os.path.isfile(path):
            missing.append(name)
        else:
            image_paths.append(path)
    if missing:
        sample = ", ".join(missing[:5])
        print(f"Warning: Missing {len(missing)} {label} images. Examples: {sample}")
    return image_paths

def extract_timestamp_seconds(filename: str) -> float:
    basename = os.path.basename(filename)
    stem, _ = os.path.splitext(basename)
    try:
        return float(stem)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp filename: {filename}") from exc

def filter_by_time_window(filenames: list, time_window_sec: float) -> list:
    if time_window_sec <= 0:
        return filenames
    timestamps = [(extract_timestamp_seconds(name), name) for name in filenames]
    start_time = min(ts for ts, _ in timestamps)
    end_time = start_time + time_window_sec
    return [name for ts, name in timestamps if ts <= end_time]

def apply_stride(filenames: list, stride: int) -> list:
    if stride is None or stride <= 1:
        return filenames
    return [name for idx, name in enumerate(filenames) if idx % stride == 0]

def evaluate(
    data_dir: str,
    output_path: str,
    exp_name: str,
    context_filenames: list,
    test_filenames: list,
    test_holdout: int,
    time_window_sec: float,
):
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    os.makedirs(output_path, exist_ok=True)

    image_folder = resolve_image_folder(data_dir)
    if not context_filenames or not test_filenames:
        raise ValueError("context_filenames and test_filenames must be non-empty.")

    if test_holdout <= 0:
        raise ValueError("test_holdout must be a positive integer.")

    context_filenames = filter_by_time_window(context_filenames, time_window_sec)
    test_filenames = filter_by_time_window(test_filenames, time_window_sec)
    if not context_filenames or not test_filenames:
        raise ValueError("Time window filter removed all context or test images; adjust time_window_sec.")

    holdout_test_filenames = [name for idx, name in enumerate(test_filenames) if idx % test_holdout == 0]
    if not holdout_test_filenames:
        raise ValueError("Holdout produced zero test images; adjust test_holdout.")

    ctx_image_paths = resolve_image_paths(image_folder, data_dir, context_filenames, "context")
    tgt_image_paths = resolve_image_paths(image_folder, data_dir, holdout_test_filenames, "test")
    if not ctx_image_paths or not tgt_image_paths:
        raise ValueError("No valid context or test images after removing missing files.")
    ctx_images = torch.stack([process_image(img_path) for img_path in ctx_image_paths], dim=0).unsqueeze(0).to(device)
    tgt_images = torch.stack([process_image(img_path) for img_path in tgt_image_paths], dim=0).unsqueeze(0).to(device)
    ctx_indices = list(range(len(ctx_image_paths)))
    tgt_indices = list(range(len(tgt_image_paths)))
    print(
        f"Using explicit lists: {len(ctx_indices)} context images, "
        f"{len(tgt_indices)} test images (holdout={test_holdout}, window={time_window_sec}s)"
    )
    ctx_images = (ctx_images+1)*0.5
    tgt_images = (tgt_images+1)*0.5
    b, v, _, h, w = tgt_images.shape

    # Run inference: Process context images through AnySplat model
    # 1. Encoder creates 3D Gaussian representation from context images
    # 2. Camera head predicts poses for ALL images (context + test)
    encoder_output = model.encoder(
        ctx_images,
        global_step=0,
        visualization_dump={},
    )
    gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose

    # Prepare all images (context + test) for camera pose prediction
    num_context_view = ctx_images.shape[1]
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)

    # Camera pose estimation: Predict extrinsic/intrinsic parameters for all views
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(vggt_input_image, intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(pred_all_pose_enc, vggt_input_image.shape[-2:])

    extrinsic_padding = torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    # Split predicted poses into context and target (test) views
    pred_all_context_extrinsic, pred_all_target_extrinsic = pred_all_extrinsic[:, :num_context_view], pred_all_extrinsic[:, num_context_view:]
    pred_all_context_intrinsic, pred_all_target_intrinsic = pred_all_intrinsic[:, :num_context_view], pred_all_intrinsic[:, num_context_view:]

    # Scale factor correction: Fix scale ambiguity in camera pose estimation
    # Neural networks can predict camera positions at arbitrary scales
    # Scale factor aligns predicted poses with encoder's reference scale
    scale_factor = pred_context_pose['extrinsic'][:, :, :3, 3].mean() / pred_all_context_extrinsic[:, :, :3, 3].mean()
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
    print("scale_factor:", scale_factor)

    # Generate predictions for test views (direct evaluation)
    # Render images from the EXACT camera positions of held-out test images
    # These predictions directly correspond to available ground truth
    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,  # Camera poses for test views
        pred_all_target_intrinsic.float(),
        torch.ones(1, v, device=device) * 0.01,
        torch.ones(1, v, device=device) * 100,
        (h, w)
        )

    # Generate interpolated novel views (demonstration of NVS capability)
    # Creates smooth camera motion between context views - TRUE novel view synthesis
    # These are completely new viewpoints not in the original dataset
    save_interpolated_video(pred_all_context_extrinsic, pred_all_context_intrinsic, b, h, w, gaussians, output_path, model.decoder)

    # Save evaluation results: Ground truth vs predictions for test views
    save_path = Path(output_path)
    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
        save_image(gt_image, save_path / "gt" / f"{idx:0>6}.jpg")    # Ground truth (real test images)
        save_image(pred_image, save_path / "pred" / f"{idx:0>6}.jpg")  # Model predictions

    # Compute quantitative metrics comparing predictions vs ground truth
    psnr, ssim, lpips = compute_metrics(output.color[0], tgt_images[0])
    psnr_mean, ssim_mean, lpips_mean = psnr.mean(), ssim.mean(), lpips.mean()
    print(f"PSNR: {psnr_mean:.2f}, SSIM: {ssim_mean:.3f}, LPIPS: {lpips_mean:.3f}")

    # Log results to CSV
    log_evaluation_results(exp_name, len(ctx_indices), len(tgt_indices), psnr_mean, ssim_mean, lpips_mean, test_holdout, time_window_sec)

def run_tum_json(tum_index_json: str, tum_root: str, output_root: str, test_holdout: int, time_window_sec: float, stride: int):
    with open(tum_index_json, "r") as f:
        index_data = json.load(f)

    completed_experiments = get_completed_experiments()

    for scene_key, scene_data in index_data.items():
        if not isinstance(scene_data, dict):
            continue

        test_view = scene_data.get("test_view")
        if not test_view:
            print(f"Skipping {scene_key}: missing test_view list")
            continue

        data_dir = os.path.join(tum_root, f"rgbd_dataset_freiburg1_{scene_key}")
        if not os.path.exists(data_dir):
            print(f"Skipping {scene_key}: data directory not found at {data_dir}")
            continue

        for context_key, context_list in scene_data.items():
            if context_key == "test_view":
                continue
            if not context_list:
                print(f"Skipping {scene_key} {context_key}: missing context list")
                continue

            exp_name = f"tum_{scene_key}_{context_key}"
            output_path = os.path.join(output_root, exp_name)
            should_run_base = exp_name not in completed_experiments
            if should_run_base:
                print(f"Received {exp_name}")
                try:
                    evaluate(
                        data_dir=data_dir,
                        output_path=output_path,
                        exp_name=exp_name,
                        context_filenames=context_list,
                        test_filenames=test_view,
                        test_holdout=test_holdout,
                        time_window_sec=time_window_sec,
                    )
                except ValueError as exc:
                    print(f"Skipping {exp_name}: {exc}")
            else:
                print(f"Skipping {exp_name}: already in evaluation_results.csv")

            if context_key == "full_view" and stride and stride > 1:
                stride_context = apply_stride(context_list, stride)
                stride_exp_name = f"tum_{scene_key}_stride_{stride}"
                stride_output_path = os.path.join(output_root, stride_exp_name)
                if stride_exp_name in completed_experiments:
                    print(f"Skipping {stride_exp_name}: already in evaluation_results.csv")
                else:
                    print(f"Received {stride_exp_name}")
                    try:
                        evaluate(
                            data_dir=data_dir,
                            output_path=stride_output_path,
                            exp_name=stride_exp_name,
                            context_filenames=stride_context,
                            test_filenames=test_view,
                            test_holdout=test_holdout,
                            time_window_sec=time_window_sec,
                        )
                    except ValueError as exc:
                        print(f"Skipping {stride_exp_name}: {exc}")

if __name__ == "__main__":
    args = setup_args()
    run_tum_json(
        args.tum_index_json,
        args.tum_root,
        args.output_root,
        args.test_holdout,
        args.time_window_sec,
        args.stride,
    )
