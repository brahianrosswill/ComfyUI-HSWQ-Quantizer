import torch
import torch.nn as nn
import os
import time
import threading
import comfy.model_management
import folder_paths

# ----------------------------------------------------------------------------
# グローバルセッション管理
# ----------------------------------------------------------------------------
# セッションデータとロックを管理
_SESSIONS = {}
_SESSION_LOCKS = {}
_GLOBAL_LOCK = threading.Lock()

def _get_lock(session_key):
    with _GLOBAL_LOCK:
        if session_key not in _SESSION_LOCKS:
            _SESSION_LOCKS[session_key] = threading.Lock()
        return _SESSION_LOCKS[session_key]

def _atomic_torch_save(obj, path: str):
    """書き込み中の破損を防ぐAtomic Save"""
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)
    except Exception as e:
        print(f"[HSWQCollector] Save failed: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)

def _get_session(save_folder_name, file_prefix, session_id):
    """セッションの取得・初期化・ロード"""
    key = f"{save_folder_name}::{file_prefix}::{session_id}"
    lock = _get_lock(key)

    output_dir = folder_paths.get_output_directory()
    full_output_path = os.path.join(output_dir, save_folder_name)
    os.makedirs(full_output_path, exist_ok=True)
    
    ckpt_path = os.path.join(full_output_path, f"{file_prefix}_{session_id}.pt")

    with lock:
        # 1. メモリ上のキャッシュを確認
        if key in _SESSIONS:
            return _SESSIONS[key], ckpt_path, lock

        # 2. ディスクから復元
        if os.path.exists(ckpt_path):
            try:
                print(f"[HSWQCollector] Loading session from {ckpt_path}")
                data = torch.load(ckpt_path, map_location="cpu")
                
                # 互換性チェック (簡易)
                if data.get("meta", {}).get("type") != "hswq_dual_monitor":
                    print("[HSWQCollector] Warning: File type mismatch. Starting new session.")
                else:
                    _SESSIONS[key] = data
                    return data, ckpt_path, lock
            except Exception as e:
                print(f"[HSWQCollector] Error loading checkpoint: {e}")

        # 3. 新規作成
        print(f"[HSWQCollector] Starting new session: {session_id}")
        session_data = {
            "meta": {
                "type": "hswq_dual_monitor",
                "created_at": time.strftime("%Y%m%d_%H%M%S"),
                "total_steps": 0,
            },
            "layers": {} 
            # 構造: 
            # "layers": {
            #    "layer_name": {
            #       "output_sum": float,      # for Sensitivity (Mean)
            #       "output_sq_sum": float,   # for Sensitivity (Variance)
            #       "out_count": int,
            #       "input_imp_sum": Tensor,  # for Importance (Channel Mean)
            #       "in_count": int
            #    }
            # }
        }
        _SESSIONS[key] = session_data
        return session_data, ckpt_path, lock

# ----------------------------------------------------------------------------
# 集計バックエンド (HSWQ DualMonitor)
# ----------------------------------------------------------------------------
class HSWQStatsCollectorBackend:
    def __init__(self, session, lock, device):
        self.session = session
        self.lock = lock
        self.device = device

    def hook_fn(self, module, input_t, output_t, name):
        # Input: tuple(Tensor, ...), Output: Tensor
        inp = input_t[0] if isinstance(input_t, tuple) else input_t
        out = output_t
        
        if not isinstance(inp, torch.Tensor) or not isinstance(out, torch.Tensor):
            return

        # --- 1. Output Sensitivity Calculation (Variance) ---
        # Variance = E[X^2] - (E[X])^2
        # ここでは sum と sum_sq を集める
        
        # Detach and cast to float32 for stability
        out_f32 = out.detach().float()
        
        # Batch mean (Global scalar for the batch)
        # HSWQスクリプトの実装に合わせ、バッチ全体の平均・二乗平均をとる
        batch_mean = out_f32.mean().item()
        batch_sq_mean = (out_f32 ** 2).mean().item()
        
        # --- 2. Input Importance Calculation (Channel Mean Abs) ---
        inp_detached = inp.detach().float()
        
        # 形状に応じたチャネル平均の計算
        # Conv2d: (B, C, H, W) -> reduce (0, 2, 3)
        # Linear/Transformer: (B, T, C) -> reduce (0, 1)
        if inp_detached.dim() == 4:
            current_imp = inp_detached.abs().mean(dim=(0, 2, 3))
        elif inp_detached.dim() == 3:
            current_imp = inp_detached.abs().mean(dim=(0, 1))
        elif inp_detached.dim() == 2:
             current_imp = inp_detached.abs().mean(dim=0)
        else:
            # Fallback for unexpected shapes
            current_imp = inp_detached.abs().mean()
            # Scalar to 1D tensor
            current_imp = current_imp.view(1)

        # CPUへ移動して集計 (VRAM節約)
        current_imp_cpu = current_imp.cpu()
        
        # --- Update Session (Thread-Safe) ---
        with self.lock:
            layers = self.session["layers"]
            if name not in layers:
                layers[name] = {
                    "output_sum": 0.0,
                    "output_sq_sum": 0.0,
                    "out_count": 0,
                    "input_imp_sum": torch.zeros_like(current_imp_cpu),
                    "in_count": 0
                }
            
            l_stats = layers[name]
            l_stats["output_sum"] += batch_mean
            l_stats["output_sq_sum"] += batch_sq_mean
            l_stats["out_count"] += 1
            
            # Input importance accumulator
            # 形状が合うか確認 (初回作成時と異なるサイズが来ない前提だがガードする)
            if l_stats["input_imp_sum"].shape == current_imp_cpu.shape:
                l_stats["input_imp_sum"] += current_imp_cpu
                l_stats["in_count"] += 1

# ----------------------------------------------------------------------------
# ComfyUI ノード定義
# ----------------------------------------------------------------------------
class SDXLHSWQCalibrationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "save_folder_name": ("STRING", {"default": "hswq_stats"}), 
                "file_prefix": ("STRING", {"default": "sdxl_calib"}),
                "session_id": ("STRING", {"default": "session_01"}),
                "target_layer": (["all_linear_conv", "attn_ffn", "unet_blocks"],),
                "save_every_steps": ("INT", {"default": 50, "min": 1, "max": 10000, "label": "Save Every N Steps"}),
                "reset_session": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "collect"
    CATEGORY = "Quantization"

    def collect(self, model, save_folder_name, file_prefix, session_id, target_layer, save_every_steps, reset_session):
        m = model.clone()
        device = comfy.model_management.get_torch_device()
        
        # セッション取得
        session, ckpt_path, lock = _get_session(save_folder_name, file_prefix, session_id)
        
        if reset_session:
            with lock:
                print(f"[HSWQCollector] Resetting session {session_id}")
                session["layers"] = {}
                session["meta"]["total_steps"] = 0
                if os.path.exists(ckpt_path):
                    try: 
                        os.remove(ckpt_path)
                    except: pass

        def stats_wrapper(model_function, params):
            # バックエンド初期化
            collector = HSWQStatsCollectorBackend(session, lock, device)
            
            def create_hook(name):
                return lambda module, i, o: collector.hook_fn(module, i, o, name)

            diffusion_model = m.model.diffusion_model
            hooks_list = []
            
            # ターゲット層のフック登録
            for name, module in diffusion_model.named_modules():
                should_hook = False
                # HSWQではConv2dとLinearの両方が対象
                is_target_type = isinstance(module, (nn.Linear, nn.Conv2d))
                
                if not is_target_type:
                    continue

                if target_layer == "all_linear_conv":
                    should_hook = True
                elif target_layer == "attn_ffn":
                    if "attn" in name or "ff" in name:
                        should_hook = True
                elif target_layer == "unet_blocks":
                    if "output_blocks" in name or "input_blocks" in name or "middle_block" in name:
                         should_hook = True

                if should_hook:
                    h = module.register_forward_hook(create_hook(name))
                    hooks_list.append(h)

            try:
                input_x = params.get("input")
                timestep = params.get("timestep")
                c = params.get("c")
                
                # 実行
                out = model_function(input_x, timestep, **c)
                
                # ステップカウントと保存チェック
                do_save = False
                with lock:
                    session["meta"]["total_steps"] += 1
                    current_steps = session["meta"]["total_steps"]
                    if current_steps % save_every_steps == 0:
                        do_save = True
                
                if do_save:
                    # ディスク書き込みは時間がかかる可能性があるため、ロック内で行うか、
                    # あるいはデータのコピーを取ってから保存する。
                    # ここでは安全のためロック内でコピーを作成し、ロック外で保存する
                    save_data = None
                    with lock:
                        # Deep copy is safest but slow. 
                        # Tensorは参照共有で問題ない(加算時に新tensorになるため)が、辞書構造はコピーする
                        save_data = {
                            "meta": session["meta"].copy(),
                            "layers": session["layers"].copy() 
                        }
                    
                    # 保存実行
                    if save_data:
                        _atomic_torch_save(save_data, ckpt_path)
                        # print(f"[HSWQCollector] Saved stats at step {current_steps}")

            finally:
                for h in hooks_list:
                    h.remove()
            
            return out

        m.set_model_unet_function_wrapper(stats_wrapper)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "SDXLHSWQCalibrationNode": SDXLHSWQCalibrationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXLHSWQCalibrationNode": "SDXL HSWQ Calibration (DualMonitor)"
}