"""mlx_lm.lora for a GPU mesh job: the same training, reported through mlx-lm's training callback.

  python lm/mesh_train.py <mlx_lm.lora arguments>        (run by lm/mesh_job.py)

Each report becomes a line `MESH {"step": n, "metrics": {...}}` (train loss, validation loss, steps per second,
peak memory), which the mesh worker turns into the dashboard's curves; reading the callback is steadier than
reading mlx-lm's console output, whose layout changes between versions. Steps count from MESH_STEP_OFFSET, so
a resumed job continues its curves. out/progress.json records the last step whose adapter was certainly saved
(the save after a report at step n lands before the next report), so a moved job resumes from it.

MLX's buffer cache and memory limit are capped as in lm/mlx_limits.py (MLX_CACHE_GB, MLX_MEMORY_GB).
"""
import json
import os
import sys
from pathlib import Path

import mlx.core as mx

mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_GB", "1")) * 1024 ** 3))
mx.set_memory_limit(int(float(os.environ.get("MLX_MEMORY_GB", "10")) * 1024 ** 3))

from mlx_lm import lora  # noqa: E402
from mlx_lm.tuner.callbacks import TrainingCallback  # noqa: E402

OFFSET = int(os.environ.get("MESH_STEP_OFFSET", "0"))
SAVE_EVERY = int(os.environ.get("MESH_SAVE_EVERY", "25"))
PROGRESS = Path(os.environ.get("MESH_PROGRESS_FILE", "out/progress.json"))


def report(step: int, **metrics) -> None:
    print("MESH " + json.dumps({"step": step, "metrics": {k: round(float(v), 5) for k, v in metrics.items()}}), flush=True)


class MeshCallback(TrainingCallback):
    def on_train_loss_report(self, info: dict) -> None:
        it = info["iteration"]
        report(OFFSET + it, train_loss=info["train_loss"], it_per_sec=info["iterations_per_second"],
               peak_mem_gb=info["peak_memory"])
        saved = (it - 1) // SAVE_EVERY * SAVE_EVERY            # saves before this report are on disk
        if saved:
            PROGRESS.parent.mkdir(parents=True, exist_ok=True)
            PROGRESS.write_text(json.dumps({"saved": OFFSET + saved}))

    def on_val_loss_report(self, info: dict) -> None:
        report(OFFSET + info["iteration"], val_loss=info["val_loss"])


lora.get_reporting_callbacks = lambda *a, **k: MeshCallback()
sys.argv = ["mlx_lm.lora"] + sys.argv[1:]
sys.exit(lora.main())
