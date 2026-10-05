from __future__ import annotations

import importlib.util
import os
import platform
import sys

import psutil
import torch


def get_hardware_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pytorch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "transformers_installed": importlib.util.find_spec("transformers") is not None,
        "triton_installed": importlib.util.find_spec("triton") is not None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_vram_gb": round(props.total_memory / (1024**3), 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "gpu_count": torch.cuda.device_count(),
        })
    else:
        info.update({"gpu": None, "gpu_vram_gb": None, "compute_capability": None, "gpu_count": 0})
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(get_hardware_info(), indent=2))
