"""
In-Memory LoRA Dynamic Hot-Scaler.
Enables sub-millisecond hyperparameter sweeps without disk reload.
"""
import torch
from peft.tuners.lora.layer import LoraLayer

class LoRAHotScaler:
    @staticmethod
    def set_active_scaling(model, adapter_name: str, scaling_factor: float):
        """
        Dynamically adjusts the effective scaling of a LoRA adapter in VRAM.
        """
        modified_count = 0
        for module in model.modules():
            if isinstance(module, LoraLayer):
                if adapter_name in module.scaling:
                    module.scaling[adapter_name] = scaling_factor
                    modified_count += 1
        return modified_count
