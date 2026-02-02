"""
AnySplat Novel View Synthesis Evaluation Script

This script evaluates AnySplat's performance on novel view synthesis using LLFF holdout strategy:
1. LLFF Holdout: Every Nth image (default N=8) is held out as test view
2. Context images are used to train/create 3D Gaussian representation
3. Optional data_index filtering: If --data_index_file is provided, only images listed in that file
   will be used as context images (test images still follow LLFF holdout)
4. Two types of predictions generated:
   - Direct: Predictions for exact test view camera positions (with ground truth available)
   - Interpolated: Novel views between context cameras (true NVS, no ground truth)
5. Evaluation: PSNR/SSIM/LPIPS metrics for direct predictions, qualitative video for NVS
"""

import os
from pathlib import Path
import sys
import json
import gzip
import argparse
import csv
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from einops import rearrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image, save_interpolated_video
from src.utils.image import process_image

from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

def parse_data_index_file(data_index_path):
    """Parse TUM rgb.txt format data index file.

    Args:
        data_index_path (str): Path to data index file

    Returns:
        list: List of image filenames
    """
    image_filenames = []
    with open(data_index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Format: timestamp filename
                timestamp, filename = parts[0], parts[1]
                image_filenames.append(filename)
    return image_filenames

def setup_args():
    """Set up command-line arguments for the eval NVS script."""
    parser = argparse.ArgumentParser(description='Test AnySplat on NVS evaluation')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically process all data_index files from data_index/ folder')
    parser.add_argument('--data_dir', type=str,
                       help='Path to NVS dataset (required unless --auto is used)')
    parser.add_argument('--data_index_file', type=str, default=None,
                       help='Optional: Path to data_index file (TUM rgb.txt format) to restrict context images. '
                            'If provided, only images listed in this file will be used as context (test images still follow LLFF holdout). '
                            'If not provided, uses standard LLFF holdout for all images.')
    parser.add_argument('--test_data_dir', type=str, default=None,
                       help='Optional: Path to separate directory for test images. If provided, test images will be loaded from this directory '
                            'instead of data_dir. Context images still come from data_dir.')
    parser.add_argument('--llffhold', type=int, default=8,
                       help='LLFF holdout: Use every Nth image as test view. '
                            'Higher values = fewer test views, more context. '
                            'Lower values = more test views, less context.')
    parser.add_argument('--output_path', type=str, default="outputs/nvs", help='Path to output directory')
    return parser.parse_args()

def compute_metrics(pred_image, image):
    psnr = compute_psnr(pred_image, image)
    ssim = compute_ssim(pred_image, image)
    lpips = compute_lpips(pred_image, image)
    return psnr, ssim, lpips

def parse_data_index_filename(filename):
    """Parse data_index filename to construct corresponding data_dir path.

    Example: 'apartment_images-jpeg-1k_10.txt' -> 'dataset/vrnerf/apartment/images-jpeg-1k/10'
    """
    # Remove .txt extension
    name = filename.replace('.txt', '')

    # Split by '_' to get parts
    parts = name.split('_')

    if len(parts) < 3:
        raise ValueError(f"Invalid data_index filename format: {filename}")

    # Extract components
    scene_name = parts[0]  # e.g., 'apartment'
    image_type = '_'.join(parts[1:-1])  # e.g., 'images-jpeg-1k'
    sequence_num = parts[-1]  # e.g., '10'

    # Construct path
    data_dir = f"dataset/vrnerf/{scene_name}/{image_type}/{sequence_num}"
    return data_dir

def get_completed_experiments(csv_file="evaluation_results.csv"):
    """Get set of completed experiment names from CSV file."""
    if not os.path.isfile(csv_file):
        return set()

    completed_experiments = set()
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row:  # Make sure row is not empty
                completed_experiments.add(row[0])  # experiment_name is first column

    return completed_experiments

def log_evaluation_results(exp_name, num_context, num_test, psnr, ssim, lpips, csv_file="evaluation_results.csv"):
    """Log evaluation results to a CSV file."""
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['experiment_name', 'num_context_views', 'num_test_views', 'psnr', 'ssim', 'lpips'])

        # Write the results
        writer.writerow([
            exp_name,
            num_context,
            num_test,
            f"{psnr:.2f}",
            f"{ssim:.3f}",
            f"{lpips:.3f}"
        ])

def evaluate(data_dir: str, data_index_file: str = None, llffhold: int = 8, output_path: str = "outputs/nvs", exp_name_override: str = None, test_data_dir: str = None):
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    os.makedirs(output_path, exist_ok=True)

    # Load images and implement LLFF holdout strategy for test view selection
    # LLFF (Local Light Field Fusion) holdout: Use every Nth image as test view
    # Default llffhold=8 means test views are at indices: 0, 8, 16, 24, etc.
    # Context views are all other indices for model training/inference
    image_folder = data_dir
    if os.path.exists(os.path.join(image_folder, "images")):
        image_folder = os.path.join(image_folder, "images")

    image_names = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not image_names:
        raise ValueError(f"No images found in {image_folder}")

    images = [process_image(img_path) for img_path in image_names]
    
    images = images[:100]
    image_names = image_names[:100]

    # LLFF holdout strategy: Test views are every llffhold-th image (sparse testing)
    # Context views are the remaining images (dense context for reconstruction)
    # Example with llffhold=8: indices [0,8,16,...] = test, others = context
    ctx_indices = [idx for idx, name in enumerate(image_names) if idx % llffhold != 0]  # Context images
    tgt_indices = [idx for idx, name in enumerate(image_names) if idx % llffhold == 0]  # Test images (ground truth)
    
    
    if test_data_dir is not None:
        test_image_names = sorted([os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not test_image_names:
            raise ValueError(f"No images found in {test_data_dir}")
        test_images = [process_image(img_path) for img_path in test_image_names]
        tgt_indices = [idx for idx, name in enumerate(test_image_names) if idx % llffhold == 0]
        print(f"Using separate test directory: {len(ctx_indices)} context images from {data_dir}, {len(test_images)} test images from {test_data_dir}")
    else:
        print(f"Standard LLFF: Using {len(ctx_indices)} context images, {len(tgt_indices)} test images")

    # If data_index_file is provided, restrict context images to those listed in the file
    if data_index_file is not None:
        image_basenames = [os.path.basename(name) for name in image_names]
        context_filenames = parse_data_index_file(data_index_file)
        print(f"Loaded {len(context_filenames)} potential context images from data_index file")

        # Regenerate ctx_indices: indices that are not test indices (i.e., not in tgt_indices)
        # AND have a basename that is in the context_filenames set
        ctx_indices = [
            idx for idx, name in enumerate(image_names)
            if (idx % llffhold != 0) and (os.path.basename(name) in context_filenames)
        ]
        print(f"Using data_index filtering: {len(ctx_indices)} context images, {len(tgt_indices)} test images")
        if len(ctx_indices) == 0:
            raise ValueError("No valid context images found (either no images match data_index file, or all matching images are test images)")
    else:
        print(f"Standard LLFF: Using {len(ctx_indices)} context images, {len(tgt_indices)} test images")
    print(f"context indices: {ctx_indices}")
    print(f"test indices: {tgt_indices}")

    ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
    if test_data_dir is not None:
        tgt_images = torch.stack([test_images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
    else:
        tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
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
    if exp_name_override:
        exp_name = exp_name_override
    else:
        exp_name = os.path.splitext(os.path.basename(data_index_file))[0] if data_index_file else "llff"
    log_evaluation_results(exp_name, len(ctx_indices), len(tgt_indices), psnr_mean, ssim_mean, lpips_mean)

if __name__ == "__main__":
    args = setup_args()

    if args.auto:
        # Auto mode: process all data_index files
        data_index_dir = "data_index"
        if not os.path.exists(data_index_dir):
            print(f"Error: {data_index_dir} directory not found")
            sys.exit(1)

        # Find all .txt files in data_index directory
        data_index_files = [f for f in os.listdir(data_index_dir) if f.endswith('.txt')]
        data_index_files.sort()  # Process in alphabetical order

        if not data_index_files:
            print(f"Error: No .txt files found in {data_index_dir}")
            sys.exit(1)

        print(f"Found {len(data_index_files)} data_index files. Starting automatic evaluation...")
        print("Each data_index file will be evaluated twice: once with full context, once with filtered context.")

        # Get list of already completed experiments
        completed_experiments = get_completed_experiments()
        if completed_experiments:
            print(f"Found {len(completed_experiments)} already completed experiments, will skip them.")

        for filename in tqdm(data_index_files, desc="Running auto evals"):
            data_index_path = os.path.join(data_index_dir, filename)
            base_exp_name = os.path.splitext(filename)[0]
            full_exp_name = f"{base_exp_name}_full"
            filtered_exp_name = f"{base_exp_name}_filtered"

            # Check if both experiments are already completed
            if full_exp_name in completed_experiments and filtered_exp_name in completed_experiments:
                print(f"\nSkipping {filename} (already completed)")
                continue

            print(f"\nProcessing {filename}")

            try:
                # Parse filename to get data_dir
                data_dir = parse_data_index_filename(filename)
                print(f"  Data directory: {data_dir}")

                # Check if data directory exists
                if not os.path.exists(data_dir):
                    print(f"  Warning: Data directory {data_dir} not found, skipping...")
                    continue

                # First evaluation: Full context (no data_index filtering)
                if full_exp_name not in completed_experiments:
                    print(f"  1/2 Full context evaluation...")
                    full_output_path = f"outputs/{full_exp_name}"
                    evaluate(data_dir, None, args.llffhold, full_output_path, exp_name_override=full_exp_name, test_data_dir=args.test_data_dir)
                    print(f"     ✓ Completed full context evaluation for {full_exp_name}")
                else:
                    print(f"  1/2 Full context evaluation... (skipping, already completed)")

                # Second evaluation: Filtered context (with data_index filtering)
                if filtered_exp_name not in completed_experiments:
                    print(f"  2/2 Filtered context evaluation...")
                    filtered_output_path = f"outputs/{filtered_exp_name}"
                    evaluate(data_dir, data_index_path, args.llffhold, filtered_output_path, exp_name_override=filtered_exp_name, test_data_dir=args.test_data_dir)
                    print(f"     ✓ Completed filtered context evaluation for {filtered_exp_name}")
                else:
                    print(f"  2/2 Filtered context evaluation... (skipping, already completed)")

            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                continue

        print("\nAutomatic evaluation completed!")
    else:
        # Manual mode: require data_dir
        if args.data_dir is None:
            print("Error: --data_dir is required when not using --auto mode")
            sys.exit(1)
        evaluate(args.data_dir, args.data_index_file, args.llffhold, args.output_path, test_data_dir=args.test_data_dir)
