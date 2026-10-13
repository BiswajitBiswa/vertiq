"""Run an MLX training command with a cap on MLX's buffer cache and a soft memory limit.

On a 16 GB Mac, preference training ran out of GPU memory right after its first validation pass: MLX
keeps freed buffers in a cache, and the next training step needed more on top. Capping the cache makes
MLX reuse or release them instead. Limits come from MLX_CACHE_GB (default 1) and MLX_MEMORY_GB
(default 10; allocations beyond it wait for the cache to be freed rather than failing).

Usage: lm/.venv/bin/python lm/mlx_limits.py <console script, e.g. mlx_lm_lora.train> [args...]
"""
import os
import sys

import mlx.core as mx

mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_GB", "1")) * 1024 ** 3))
mx.set_memory_limit(int(float(os.environ.get("MLX_MEMORY_GB", "10")) * 1024 ** 3))
command, sys.argv = sys.argv[1], sys.argv[1:]
if command == "mlx_lm_lora.train":
    from mlx_lm_lora.train import main
elif command == "mlx_lm.lora":
    from mlx_lm.lora import main
else:
    raise SystemExit(f"unknown command {command!r}")
sys.exit(main())
