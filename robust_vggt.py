#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import einops
import numpy as np
import torch
import torchvision
import gc
from torch import Tensor

from tqdm import tqdm
import os
from PIL import Image

FILE_PATH = Path(__file__).resolve()
REPO_ROOT = FILE_PATH.parents[2]

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert batched SE3 matrices."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    Tin = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    Tin = Tin.clone()
    Tin[..., :3, :3] = Rt
    Tin[..., :3, 3] = t_inv
    return Tin

@dataclass(frozen=True)
class ExperimentConfig:
    image_dir: Path
    preprocess_mode: str = "crop"
    exp_name: str = "demo_result"
    attn_a: float = 0.5
    cos_a: float = 0.5
    rej_thresh: float = 0.4


def safe_empty_cache() -> None:
    """Aggressively free memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def free_cuda(*args):
    for arg in args:
        del arg
    safe_empty_cache()


def extrinsics_to_matrix(extrinsics: Tensor) -> Tensor:
    """Convert [N, 3, 4] extrinsics to homogenous [N, 4, 4] matrices."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics of shape [N,3,4], got {tuple(extrinsics.shape)}")
    n = extrinsics.shape[0]
    device = extrinsics.device
    dtype = extrinsics.dtype
    mats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(n, 4, 4).clone()
    mats[:, :3, :3] = extrinsics[:, :3, :3]
    mats[:, :3, 3] = extrinsics[:, :3, 3]
    return mats


def convert_world_to_cam_to_cam_to_world(extrinsics: Tensor) -> Tensor:
    """Invert world-to-camera extrinsics to obtain camera-to-world transforms."""
    T_w2c = extrinsics_to_matrix(extrinsics)
    return invert_se3(T_w2c)


def list_image_paths(images_dir: Path) -> List[Path]:
    """List image files inside the given directory."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    valid_suffixes = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix in valid_suffixes],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise RuntimeError(f"No image files found in {images_dir}")
    return image_paths


def serialize_paths(value: Any) -> Any:
    """Recursively convert Path objects within nested structures to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: serialize_paths(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_paths(v) for v in value]
    return value


def select_device_and_dtype() -> tuple[str, torch.dtype]:
    """Choose the compute device and mixed-precision dtype."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return "cuda", dtype
    return "cpu", torch.float32


# ANSI color helpers for terminal output
ANSI_GREEN = "\033[92m"
ANSI_RESET = "\033[0m"
def info_print(msg: str) -> None:
    print(f"{ANSI_GREEN}{msg}{ANSI_RESET}")


class RobustVGGTExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device, self.amp_dtype = select_device_and_dtype()
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        try:
            torch.set_grad_enabled(False)
        except Exception:
            pass

    def _forward_once(self, images: Tensor, image_hw: tuple[int, int], device: torch.device) -> Tensor:
        import math
        import numpy as np
        import torch
        import torch.nn.functional as F
        from typing import List, Optional

        non_blocking = device.type == "cuda"
        images_device = images.to(device=device, non_blocking=non_blocking)
        if device.type == "cuda":
            images_device = images_device.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)

        image_level_attn_mask: Optional[Tensor] = None
        rejected_image_indices: List[int] = []
        image_attention_means: List[float] = []
        global_norm_means: List[float] = []

        attn_layers: List[int] = []
        q_out: Dict[int, Tensor] = {}
        k_out: Dict[int, Tensor] = {}
        handles: List[Any] = []

        attn_layers = [23]
        info_print(f"Setting up hooks for layers: {attn_layers}")

        def _make_hook(store_dict, idx):
            def _hook(_module, _inp, out):
                store_dict[idx] = out.detach()
            return _hook

        for i in attn_layers:
            blk = self.model.aggregator.global_blocks[i].attn
            handles.append(blk.q_norm.register_forward_hook(_make_hook(q_out, i)))
            handles.append(blk.k_norm.register_forward_hook(_make_hook(k_out, i)))

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                    predictions, aggregated_tokens_list = self.model(images_device)
            else:
                predictions, aggregated_tokens_list = self.model(images_device)

        prediction_save_path = self.pair_out_dir / f"predictions_first_forward.npz"
        save_predictions = {k: v.float().cpu() for k, v in predictions.items() if torch.is_tensor(v)}
        np.savez(prediction_save_path, **save_predictions)

        '''
        aggregated_tokens_list: features from all layers (list of tensors). 
        '''
        target_layers = [23]
        aggregated_tokens_selected = [aggregated_tokens_list[idx] for idx in target_layers]
        global_tokens = [tokens[..., 1024:] for tokens in aggregated_tokens_selected]

        aggregator = self.model.aggregator
        patch_size = aggregator.patch_size
        patch_start_idx = aggregator.patch_start_idx
        H, W = image_hw
        h_patches = H // patch_size 
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens
            
        cosine_similarities = []
        num_images_total = images.shape[0]
        reject_flags = [False] * num_images_total  
        layer_feat = ref_feat = ref_feat_norm = layer_feat_norm = cos_sim = cos_sim_mean = None
        for layer_idx, feature in enumerate(global_tokens):
            if feature.ndim != 4:
                continue
            B, N, T, C = feature.shape 
            if T == 0 or C == 0:
                continue
            
            feature = feature[:, :, patch_start_idx:, :]
            B, N, T, C = feature.shape
            layer_feat = feature.detach().to(dtype=torch.float32)
            num_samples = B * N
            layer_feat = layer_feat.reshape(num_samples, T, C)
            
            ref_feat = layer_feat[0:1, :, :]  # (1, T, C)
            ref_feat_norm = F.normalize(ref_feat, p=2, dim=-1)  # (1, T, C)
            layer_feat_norm = F.normalize(layer_feat, p=2, dim=-1)  # (B*N, T, C)
            cos_sim = torch.einsum("bic,bjc->bij", layer_feat_norm, ref_feat_norm)  # (B*N, T, T)
            cos_sim_mean = cos_sim.mean(-1).mean(-1)  # (B*N,)
            cosine_similarities.append(cos_sim_mean)  # List of (N,)
        
        global_tokens = None
        aggregated_tokens_selected = None
        safe_empty_cache()
        
        predictions_full = predictions
        try:
            if isinstance(predictions_full, dict):
                for _k, _v in list(predictions_full.items()):
                    if torch.is_tensor(_v) and _v.device.type == "cuda":
                        predictions_full[_k] = _v.detach().cpu()
        except Exception:
            pass
        safe_empty_cache()
        torch.cuda.empty_cache()
                
        global_mean_vals = []
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        try:
            handles.clear()
        except Exception:
            pass

        def _num_input_images(x: Tensor) -> int:
            if x.ndim == 5:   # B, N, C, H, W
                return int(x.shape[1])
            if x.ndim == 4:   # N, C, H, W
                return int(x.shape[0])
            if x.ndim == 3:   # C, H, W
                return 1
            raise ValueError(f"Unsupported images shape: {x.shape}")

        total_imgs = _num_input_images(images)
        num_vis = total_imgs  # Visualize all images
        info_print(f"[INFO] Visualizing self-attention maps for the first {num_vis} input images.")
        batch_size = images.shape[0] if images.ndim == 5 else 1

        if images.ndim == 5:
            base_images = images[0, :num_vis].detach().cpu()         # First num_vis images of the first batch
        elif images.ndim == 4:
            base_images = images[:num_vis].detach().cpu()
        elif images.ndim == 3:
            base_images = images.unsqueeze(0).detach().cpu()
        else:
            raise ValueError(f"Unsupported images shape: {images.shape}")

        aggregator = self.model.aggregator
        patch_size = aggregator.patch_size
        patch_start_idx = aggregator.patch_start_idx
        H, W = image_hw
        h_patches = H // patch_size  
        w_patches = W // patch_size
        num_patch_tokens = h_patches * w_patches
        tokens_per_image = patch_start_idx + num_patch_tokens
        
        global_norms_list=[]

        avg_maps2d_sum = [torch.zeros((h_patches, w_patches), dtype=torch.float32) for _ in range(num_vis)]
        avg_counts = [0 for _ in range(num_vis)]
        reject_flags = [False] * num_vis
        image_attention_means = [float("nan")] * num_vis
        global_norm_means = [float("nan")] * num_vis
        rejection_threshold = self.config.rej_thresh

        first_image_patch_start = patch_start_idx
        first_image_patch_end = first_image_patch_start + num_patch_tokens

        for i in tqdm(attn_layers):
            global_norms_list=[]
            if i not in q_out or i not in k_out:
                continue

            Q = q_out[i]
            K = k_out[i]

            T = int(K.shape[-2])
            num_images_in_seq = T // tokens_per_image
            if num_images_in_seq <= 0:
                continue
            q_first_image = Q[:, :, first_image_patch_start:first_image_patch_end, :]  # (B, H, Nq, D)
            Tk = int(min(num_vis, num_images_in_seq) * tokens_per_image)              # (Tk)
            K_slice = K[:, :, :Tk, :]                                                # (B, H, Tk, D)
            scale = 1.0 / math.sqrt(float(q_first_image.shape[-1]))
            logits = torch.einsum("bhqd,bhtd->bhqt", q_first_image, K_slice) * scale  # (B, H, Nq, Tk)
            probs = torch.softmax(logits, dim=-1)                                      # (B, H, Nq, Tk)
            attn_first_image = probs.mean(dim=1).mean(dim=1)[0]                        # (Tk,)

            import matplotlib.pyplot as plt

            maps_up = []        # Torch tensors upsampled to (H,W) via bilinear interpolation
            maps_2d = []        # Original patch grid attention (h_patches, w_patches)
            valid_indices = []  # Image indices that were actually drawable

            global_min = None
            global_max = None

            for img_idx in range(num_vis):
                start = img_idx * tokens_per_image + patch_start_idx
                end = start + num_patch_tokens
                if start >= attn_first_image.shape[-1]:
                    break
                end = min(end, attn_first_image.shape[-1])

                patch_attn = attn_first_image[start:end]  
                if patch_attn.numel() != num_patch_tokens:
                    continue

                attn2d = patch_attn.view(h_patches, w_patches)  # (h, w)

                attn2d_up = F.interpolate(
                    attn2d.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )[0, 0]  # (H, W)

                maps_2d.append(attn2d)       # (h, w)
                maps_up.append(attn2d_up)    # (H, W)
                valid_indices.append(img_idx)

                vmin = torch.min(attn2d)
                vmax = torch.max(attn2d)
                global_min = vmin if (global_min is None) else torch.minimum(global_min, vmin)
                global_max = vmax if (global_max is None) else torch.maximum(global_max, vmax)

            if len(maps_up) == 0:
                free_cuda(Q, K, q_first_image, K_slice, logits, probs, attn_first_image)
                continue
            
            for idx, img_idx in enumerate(valid_indices):
                avg_maps2d_sum[img_idx] += maps_2d[idx].detach().cpu()
                avg_counts[img_idx] += 1

            eps = 1e-12
            gmin = float(global_min.item())
            gmax = float(global_max.item())
            gden = (gmax - gmin) if (gmax > gmin) else 1.0

            for col_idx, img_idx in enumerate(valid_indices):
                role = "anchor" if img_idx == 0 else "support"

                img_tensor = base_images[img_idx].float()  # (C, H, W)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0.0, 1.0)

                m_up = maps_up[col_idx]
                m_up_np = m_up.detach().cpu().numpy()

                lmin = float(m_up.min().item())
                lmax = float(m_up.max().item())
                lden = (lmax - lmin) if (lmax > lmin) else 1.0
                local_norm = (m_up_np - lmin) / lden

                global_norm = (m_up_np - gmin) / gden
                global_mean_val = float(global_norm.mean())
                global_mean_vals.append(global_mean_val)
                global_norm_means[img_idx] = global_mean_val
                global_norms_list.append(global_norm)
       
            free_cuda(Q, K, q_first_image, K_slice, logits, probs, attn_first_image)
            del Q, K, q_first_image, K_slice, logits, probs, attn_first_image
            try:
                del maps_up, maps_2d, m_up
            except Exception:
                pass
            torch.cuda.empty_cache()
                
        if global_mean_vals and cosine_similarities:
            cos_sim = cosine_similarities[0]  # (N,)
            attn_val = torch.tensor(global_mean_vals, device=cos_sim.device, dtype=cos_sim.dtype)  # (N,)

            cos_sim = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-6)
            attn_val = (attn_val - attn_val.min()) / (attn_val.max() - attn_val.min() + 1e-6)
            
            combined_score = self.config.attn_a * attn_val + self.config.cos_a * cos_sim 
            
            # Reject
            for idx in range(len(combined_score)):
                if idx == 0:
                    continue  # Never reject the reference image
                if combined_score[idx] < self.config.rej_thresh:
                    reject_flags[idx] = True
            info_print(f"[INFO] Rejection threshold: {self.config.rej_thresh:.4f}")
        
        rejected_image_indices = [idx for idx, flag in enumerate(reject_flags) if flag and idx != 0]
        info_print(f"[INFO] Integrated rejection: {rejected_image_indices}")

        del avg_maps2d_sum, avg_counts
        try:
            q_out.clear(); k_out.clear()
        except Exception:
            pass
        del q_out, k_out
        try:
            del base_images
        except Exception:
            pass
        safe_empty_cache()
            
        aggregated_tokens_list = None
        aggregated_tokens_selected = None
        global_tokens = None
        cosine_similarities = []
        safe_empty_cache()
            
        if len(rejected_image_indices) > 0:
            if images.ndim == 5:  # (B, N, C, H, W)
                total_N = int(images.shape[1])
                B_dim = int(images.shape[0])
            elif images.ndim == 4:  # (N, C, H, W)
                total_N = int(images.shape[0])
                B_dim = 1
            elif images.ndim == 3:  # (C, H, W)
                total_N = 1
                B_dim = 1
            else:
                raise ValueError(f"Unsupported images shape: {images.shape}")

            survivors = [i for i in range(total_N) if i not in rejected_image_indices]

            if len(survivors) == 0:
                info_print("[INFO] All images rejected by attention-vis; skipping second forward.")
            elif len(survivors) == total_N:
                pass
            else:
                try:
                    del image_level_attn_mask
                except Exception:
                    pass
                try:
                    del images_device
                except Exception:
                    pass
                safe_empty_cache()

                if images.ndim == 5:
                    images_subset_cpu = images[:, survivors, ...]
                elif images.ndim == 4:
                    images_subset_cpu = images[survivors, ...]
                elif images.ndim == 3:
                    images_subset_cpu = images.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported images shape: {images.shape}")

                images_subset = images_subset_cpu.to(device=device, dtype=self.amp_dtype, non_blocking=non_blocking)
        
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            predictions_survive, _ = self.model(images_subset)
                    else:
                        predictions_survive, _ = self.model(images_subset)
                
                prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
                save_predictions = {k: v.float().cpu() for k, v in predictions_survive.items() if torch.is_tensor(v)}
                np.savez(prediction_save_path, **save_predictions)

                try:
                    if isinstance(predictions_survive, dict):
                        for _k, _v in list(predictions_survive.items()):
                            if torch.is_tensor(_v) and _v.device.type == "cuda":
                                predictions_survive[_k] = _v.detach().cpu()
                    
                        predictions = predictions_survive
                except Exception:
                    print("[WARN] Failed to move survivor predictions to CPU")
                    pass
                try:
                    del images_subset, predictions_survive
                except Exception:
                    pass
                safe_empty_cache()
        else:
            prediction_save_path = self.pair_out_dir / f"predictions_survived.npz"
            save_predictions = {k: v.float().cpu() for k, v in predictions_full.items() if torch.is_tensor(v)}
            np.savez(prediction_save_path, **save_predictions)
            predictions = predictions_full

        pose_enc = predictions["pose_enc"]
        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, image_hw)
        extrinsics = extrinsics.squeeze(0) if extrinsics.ndim == 4 else extrinsics
        Twc = convert_world_to_cam_to_cam_to_world(extrinsics)
        result = Twc.detach().cpu().float()
        conf = predictions["depth_conf"].detach().cpu().float()
        depth = predictions["depth"].detach().cpu().float()

        del predictions
        try:
            del images_device
        except Exception:
            pass
        safe_empty_cache()

        return result, depth, conf, rejected_image_indices

    def run_demo(self) -> None:
        self.pair_out_dir = Path(self.config.exp_name)
        self.pair_out_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list_image_paths(self.config.image_dir)
        info_print(f"[INFO] Found {len(image_paths)} images in {self.config.image_dir}")

        images_tensor = load_and_preprocess_images(
            [str(p) for p in image_paths],
            mode=self.config.preprocess_mode,
        )
        
        try:
            import matplotlib.pyplot as plt
            from torchvision.utils import make_grid
            images_cpu = images_tensor.detach().cpu()
            grid = make_grid(images_cpu, nrow=8, padding=2, pad_value=1.0)
            grid_np = grid.permute(1, 2, 0).numpy()
            plt.imsave(self.pair_out_dir / "preprocessed_grid.png", np.clip(grid_np, 0.0, 1.0))
        except Exception:
            pass

        if torch.device(self.device).type == "cuda":
            images_tensor = images_tensor.pin_memory()

        image_hw = tuple(int(dim) for dim in images_tensor.shape[-2:])
        
        pred_twcs, depth, conf, rejected_indices = self._forward_once(images_tensor, image_hw, torch.device(self.device))


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run VGGT demo on a directory of images.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Directory containing images to process.",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["crop", "pad"],
        default="crop",
        help="Image preprocessing mode.",
    )
    parser.add_argument("--exp-name", type=str, default="demo_result", help="Experiment name for output directory.")
    parser.add_argument("--attn_a", type=float, default=0.5, help="Attention weight.")
    parser.add_argument("--cos_a", type=float, default=0.5, help="Cosine similarity weight.")
    parser.add_argument("--rej-thresh", type=float, default=0.4, help="Rejection threshold.")

    args = parser.parse_args()

    return ExperimentConfig(
        image_dir=args.image_dir,
        preprocess_mode=args.preprocess_mode,
        exp_name=args.exp_name,
        attn_a=args.attn_a,
        cos_a=args.cos_a,
        rej_thresh=args.rej_thresh,
    )


def main() -> None:
    config = parse_args()
    experiment = RobustVGGTExperiment(config)
    experiment.run_demo()


if __name__ == "__main__":
    main()
