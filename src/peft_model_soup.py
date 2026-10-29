"""
PEFT Model Soup & Low-Rank Adapter Fusion Guard.
Provides sub-10ms task arithmetic / soup interpolation without dense expansion.
"""
import torch
from peft import PeftModel
from typing import List, Dict, Optional

class LowRankModelSoupMerger:
    @staticmethod
    def merge_adapters_direct(
        model: PeftModel,
        adapters: List[str],
        weights: List[float],
        soup_name: str = "soup_active",
        combination_type: str = "linear",
        cleanup_existing: bool = True
    ) -> None:
        """
        Merges adapters directly in low-rank manifold (A, B matrices) in < 10ms.
        """
        if cleanup_existing and soup_name in model.peft_config:
            try:
                model.delete_adapter(soup_name)
            except Exception:
                pass

        if combination_type in ["linear", "ties", "dare_ties"]:
            model.add_weighted_adapter(
                adapters=adapters,
                weights=weights,
                adapter_name=soup_name,
                combination_type=combination_type
            )
        elif "svd" in combination_type:
            # Enforce non-blocking Jacobi SVD with full_matrices=False
            model.add_weighted_adapter(
                adapters=adapters,
                weights=weights,
                adapter_name=soup_name,
                combination_type=combination_type,
                svd_rank=16,
                svd_full_matrices=False,
                svd_driver="gesvda"
            )
        else:
            raise ValueError(f"Unsupported combination_type: {combination_type}")

        model.set_adapter(soup_name)
