import torch
import torch.nn as nn
import os
import sys
import numpy as np
import comfy.model_management
from comfy.model_patcher import ModelPatcher

# ----------------------------------------------------------------------------
# HSWQ Optimizer Logic (Embedded & Enhanced for Debugging)
# ----------------------------------------------------------------------------
class FP8E4M3Quantizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._positive_grid = None
        self.max_representable = 448.0
        self._build_fp8_grid()
    
    def _build_fp8_grid(self):
        try:
            # Generate all byte patterns
            all_bytes = torch.arange(256, dtype=torch.uint8, device=self.device)
            if hasattr(torch, "float8_e4m3fn"):
                fp8_vals = all_bytes.view(torch.float8_e4m3fn)
                f32_vals = fp8_vals.float()
                valid_mask = ~f32_vals.isnan()
                valid_vals = f32_vals[valid_mask]
                pos_vals = valid_vals[valid_vals >= 0]
                unique_vals = pos_vals.unique().sort().values
                self._positive_grid = unique_vals
                self.max_representable = self._positive_grid.max().item()
            else:
                print("[HSWQ] Error: torch.float8_e4m3fn not supported. Quantization will fail.")
                self._positive_grid = torch.linspace(0, 448, 256, device=self.device) # Dummy fallback
        except Exception as e:
            print(f"[HSWQ] Warning: FP8 grid build failed ({e}).")

    def quantize_dequantize(self, values: torch.Tensor, amax: float) -> torch.Tensor:
        """
        Simulation of quantization error for MSE search.
        Compatible Mode (scaled=False): Clip -> Round to nearest grid
        """
        if amax <= 0: return torch.zeros_like(values)
        
        # 1. Clip
        clipped = values.clamp(-amax, amax)
        clipped = clipped.clamp(-self.max_representable, self.max_representable)
        
        # 2. Round to nearest (Simulation)
        if hasattr(torch, "float8_e4m3fn"):
            x_fp8 = clipped.to(torch.float8_e4m3fn)
            x_dq = x_fp8.float()
            return x_dq
        else:
            return clipped

class WeightedHistogram:
    def __init__(self, bins: int = 2048, device: str = "cuda"):
        self.bins = bins
        self.device = device
        self.histogram = None
        self.max_val = 0.0
        self.total_weight = 0.0
        
    def build(self, weight: torch.Tensor, importance: torch.Tensor = None):
        weight = weight.detach().float().to(self.device)
        w_abs = weight.abs()
        self.max_val = w_abs.max().item()
        if self.max_val == 0: self.max_val = 1e-7

        # Expand Importance
        imp_expanded = torch.ones_like(weight)
        
        if importance is not None:
            importance = importance.float().to(self.device)
            if importance.dim() == 0:
                importance = importance.view(1)
            
            # Padding check
            if weight.dim() == 4: # Conv2d
                in_c = weight.shape[1]
                if importance.numel() < in_c: 
                     padding = torch.ones(in_c - importance.numel(), device=self.device)
                     importance = torch.cat([importance, padding])
                imp = importance[:in_c].view(1, -1, 1, 1)
                imp_expanded = imp.expand_as(weight)
            
            elif weight.dim() == 2: # Linear
                in_c = weight.shape[1]
                if importance.numel() < in_c:
                     padding = torch.ones(in_c - importance.numel(), device=self.device)
                     importance = torch.cat([importance, padding])
                imp = importance[:in_c].view(1, -1)
                imp_expanded = imp.expand_as(weight)
        
        bin_width = self.max_val / self.bins
        bin_indices = (w_abs / bin_width).long().clamp(0, self.bins - 1)
        
        self.histogram = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.histogram.scatter_add_(0, bin_indices.view(-1), imp_expanded.reshape(-1).double())
        
        self.total_weight = self.histogram.sum().item()
        if self.total_weight > 0:
            self.histogram /= self.total_weight
    
    def get_bin_centers(self) -> torch.Tensor:
        bin_width = self.max_val / self.bins
        return torch.linspace(0.5 * bin_width, self.max_val - 0.5 * bin_width, self.bins, device=self.device, dtype=torch.float64)

class MSEOptimizer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.quantizer = FP8E4M3Quantizer(device)
    
    def compute_mse(self, hist, centers, amax):
        dequant = self.quantizer.quantize_dequantize(centers.float(), amax).double()
        err = (dequant - centers) ** 2
        return (hist * err).sum().item()
    
    def find_optimal_amax(self, weighted_hist: WeightedHistogram, candidates=100, iterations=3):
        if weighted_hist.histogram is None or weighted_hist.max_val <= 0:
            return weighted_hist.max_val, 0.0
            
        hist = weighted_hist.histogram
        centers = weighted_hist.get_bin_centers()
        max_val = weighted_hist.max_val
        
        low = max_val * 0.1
        high = max_val * 1.0
        best_amax = max_val
        min_mse = float('inf')
        
        for _ in range(iterations):
            test_amaxes = torch.linspace(low, high, candidates, device=self.device)
            for amax_t in test_amaxes:
                amax = amax_t.item()
                mse = self.compute_mse(hist, centers, amax)
                if mse < min_mse:
                    min_mse = mse
                    best_amax = amax
            
            width = (high - low) / candidates * 4
            low = max(0.01, best_amax - width)
            high = min(max_val, best_amax + width)
            
        return best_amax, min_mse

# ----------------------------------------------------------------------------
# ComfyUI Node
# ----------------------------------------------------------------------------
class HSWQFP8ConverterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "hswq_stats_path": ("STRING", {"default": "output/hswq_stats/sdxl_calib_session_01.pt"}),
                "keep_ratio": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05, "label": "FP16 Keep Ratio"}),
                "log_level": (["Basic", "Verbose", "Debug"], {"default": "Basic"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "convert"
    CATEGORY = "Quantization"

    def convert(self, model, hswq_stats_path, keep_ratio, log_level):
        print(f"### [HSWQ] Starting FP8 Conversion (Log: {log_level}) ###")
        print(f"[HSWQ] Loading stats from {hswq_stats_path}")
        
        if not os.path.exists(hswq_stats_path):
            print(f"[HSWQ] Error: Stats file not found at {hswq_stats_path}")
            return (model,)

        try:
            session_data = torch.load(hswq_stats_path, map_location="cpu")
        except Exception as e:
            print(f"[HSWQ] Error loading stats: {e}")
            return (model,)

        layers_data = session_data.get("layers", {})
        if not layers_data:
            print("[HSWQ] Error: No layer data found in stats file.")
            return (model,)

        # --- 1. Sensitivity Analysis ---
        print("[HSWQ] Analyzing sensitivity...")
        sensitivities = []
        for name, stats in layers_data.items():
            count = stats["out_count"]
            if count > 0:
                mean = stats["output_sum"] / count
                sq_mean = stats["output_sq_sum"] / count
                variance = sq_mean - (mean ** 2)
                if variance < 0: variance = 0
                sensitivities.append((name, variance))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        total_layers = len(sensitivities)
        num_keep = int(total_layers * keep_ratio)
        keep_names = set(x[0] for x in sensitivities[:num_keep])
        
        print(f"[HSWQ] Total stats layers: {total_layers}")
        print(f"[HSWQ] Keep Top {num_keep} layers (FP16). Threshold Var: {sensitivities[num_keep-1][1]:.6f}" if num_keep > 0 else "[HSWQ] Keep 0 layers.")
        
        # --- 2. Setup Quantization ---
        work_model = model.clone()
        if isinstance(work_model, ModelPatcher):
            diffusion_model = work_model.model.diffusion_model
        else:
            diffusion_model = work_model.diffusion_model

        device = comfy.model_management.get_torch_device()
        optimizer_backend = MSEOptimizer(device=device)

        converted_count = 0
        skipped_sensitive_count = 0
        skipped_no_stats_count = 0
        error_accum = 0.0

        print("[HSWQ] Starting quantization loop...")
        
        # Check Stats vs Model mismatch
        all_model_layers = []
        for name, module in diffusion_model.named_modules():
             if isinstance(module, (nn.Linear, nn.Conv2d)):
                 all_model_layers.append(name)
        
        stats_keys = set(layers_data.keys())
        model_keys = set(all_model_layers)
        missing_in_stats = model_keys - stats_keys
        
        if log_level != "Basic":
            if missing_in_stats:
                print(f"[HSWQ] WARNING: {len(missing_in_stats)} layers exist in model but miss stats (will be skipped).")
        
        # Loop
        for name, module in diffusion_model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            # Check weights
            if not hasattr(module, "weight") or module.weight is None:
                continue

            # Skip Logic
            if name not in layers_data:
                skipped_no_stats_count += 1
                if log_level == "Debug": print(f"[Skip] No Stats: {name}")
                continue
            
            if name in keep_names:
                skipped_sensitive_count += 1
                if log_level == "Debug": print(f"[Keep] Sensitive: {name}")
                continue

            # Quantization
            try:
                stats = layers_data[name]
                in_count = stats["in_count"]
                if in_count == 0:
                    print(f"[HSWQ] Warning: {name} has stats entry but 0 count.")
                    continue

                weight = module.weight.data
                imp_sum = stats["input_imp_sum"]
                
                # Check shapes for Importance
                if weight.shape[1] != imp_sum.shape[0]:
                    if log_level in ["Verbose", "Debug"]:
                        print(f"[HSWQ] Shape mismatch for {name}: W{weight.shape} vs Imp{imp_sum.shape}. Using ones.")
                    importance = None
                else:
                    importance = imp_sum.float() / in_count

                # Optimizer Logic
                wh = WeightedHistogram(bins=2048, device=device)
                wh.build(weight, importance)
                
                optimal_amax, mse_score = optimizer_backend.find_optimal_amax(wh, candidates=50, iterations=3)
                
                # Apply Quantization
                w_dev = weight.to(device)
                w_clamped = torch.clamp(w_dev, -optimal_amax, optimal_amax)
                w_fp8 = w_clamped.to(torch.float8_e4m3fn)
                
                # VERIFICATION: Check if valid (FIXED: Cast to float first)
                w_check = w_fp8.float()
                if torch.isnan(w_check).any() or torch.isinf(w_check).any():
                    print(f"[HSWQ] CRITICAL: NaNs/Inf detected in quantized {name}. Reverting to FP16.")
                    del w_check
                    continue
                del w_check

                # Assign back
                module.weight.data = w_fp8.to(weight.device)
                
                converted_count += 1
                error_accum += mse_score
                
                if log_level == "Debug" or (log_level == "Verbose" and converted_count % 50 == 0):
                    max_w = weight.abs().max().item()
                    ratio = optimal_amax / max_w if max_w > 0 else 0
                    print(f" [Quant] {name}: Max={max_w:.4f} -> Amax={optimal_amax:.4f} (Ratio:{ratio:.2f}) MSE={mse_score:.2e}")

                del wh, w_dev, w_clamped, w_fp8

            except Exception as e:
                print(f"[HSWQ] Failed to quantize {name}: {e}")

        print("------------------------------------------------")
        print(f"### [HSWQ] Quantization Finished ###")
        print(f"  Converted to FP8: {converted_count}")
        print(f"  Kept in FP16 (Sensitive): {skipped_sensitive_count}")
        print(f"  Skipped (No Stats): {skipped_no_stats_count}")
        print(f"  Total Layers Processed: {converted_count + skipped_sensitive_count + skipped_no_stats_count}")
        if converted_count > 0:
            print(f"  Avg Optimization MSE: {error_accum/converted_count:.2e}")
        print("------------------------------------------------")

        return (work_model,)

NODE_CLASS_MAPPINGS = {
    "HSWQFP8ConverterNode": HSWQFP8ConverterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HSWQFP8ConverterNode": "HSWQ FP8 Converter (Debug Enhanced)"
}