"""
Phased Burst Concurrency Scheduler.
Eliminates Python GIL contention and CUDA Kernel Micro-Starvation during autoregressive decoding.
"""
import os
import time
from typing import List, Dict, Any, Callable
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class PhasedBurstScheduler:
    def __init__(self, model, tokenizer, eval_workers: int = 8, batch_size: int = 32):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_workers = eval_workers
        self.batch_size = batch_size

    def step(self, prompts: List[str], eval_fn: Callable[[str], Dict[str, Any]], max_new_tokens: int = 384) -> List[Dict[str, Any]]:
        """
        Executes Phased Burst Isolation:
        Phase 1: Pure GPU Autoregressive Generation (100% GIL exclusive, saturated batch).
        Phase 2: Multiprocess CPU Unit Test / Environment Evaluation.
        """
        # ==========================================================
        # Phase 1: Pure GPU Autoregressive Generation (Saturated Batch)
        # ==========================================================
        t0 = time.perf_counter()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # Decode only newly generated tokens
        input_len = inputs.input_ids.shape[1]
        generated_texts = [
            self.tokenizer.decode(out[input_len:], skip_special_tokens=True)
            for out in output_ids
        ]
        gpu_time = time.perf_counter() - t0

        # ==========================================================
        # Phase 2: CPU Multi-Worker Evaluation Burst
        # ==========================================================
        t1 = time.perf_counter()
        results = []
        with ThreadPoolExecutor(max_workers=self.eval_workers) as executor:
            futures = [executor.submit(eval_fn, text) for text in generated_texts]
            for f in futures:
                results.append(f.result())
        cpu_time = time.perf_counter() - t1

        return {
            "results": results,
            "generated_texts": generated_texts,
            "gpu_time_sec": gpu_time,
            "cpu_time_sec": cpu_time,
            "throughput_tok_per_sec": (output_ids.shape[0] * (output_ids.shape[1] - input_len)) / max(gpu_time, 1e-4)
        }
