# TabArena PluRel-16B Inference (Updated Single-Table Context)

This runbook is for running TabArena inference with the PluRel 16B checkpoint using
the latest RelBench TabArena task semantics:

- task names are `split-N` (not `fold-N`)
- task tables are edge-free (`fkey_col_to_pkey_table = {}`)
- no synthetic task timestamps (`time_col=None`)

This is intended to keep TabArena task context single-node for RT/rustler-style samplers.

## 1) Pin Revisions

Use these exact code revisions for reproducibility:

- `relbench`: branch `tabarena-single-table`, commit `f796cba`
- `relational-transformer`: commit `f1243c35b8410610102cfc478cdb93c1e0ab3d50`

## 2) Verify TabArena Is Edge-Free (Required)

Run in the `relbench` environment:

```bash
cd /home/pc0618/relbench
python - <<'PY'
from relbench.tasks import get_task
t = get_task("tabarena-apsfailure", "split-0", download=False)
train = t.get_table("train", mask_input_cols=False)
test = t.get_table("test")
print("train_fkeys:", train.fkey_col_to_pkey_table)
print("test_fkeys:", test.fkey_col_to_pkey_table)
print("time_col:", train.time_col)
assert train.fkey_col_to_pkey_table == {}
assert test.fkey_col_to_pkey_table == {}
assert train.time_col is None
print("OK")
PY
```

## 3) Patch RT Sweep Script For `split-*` Tasks

`scripts/rt_tabarena_sweep.py` in `relational-transformer` currently references `fold-*`.
Patch it to use `split-*` for task-table access:

```bash
cd /home/pc0618/relational-transformer
rg -n "fold-" scripts/rt_tabarena_sweep.py
```

Update these patterns:

- `f"fold-{fold}"` -> `f"split-{fold}"`
- `/tasks/f"fold-{fold}"` -> `/tasks/f"split-{fold}"`

Note: keeping the CSV column name as `fold` is fine; it can still hold split index values.

## 4) Random Sampling = True

Rustler sampling is stochastic by default (no extra flag required):

- random neighbor subsampling is used when fanout exceeds `max_bfs_width`
- random traversal order is used in sampling
- sampling is controlled by `seed` and `shuffle_py(...)`

So "use random sampling = true" corresponds to running the default sampler behavior
with a fixed `--seed` for reproducibility.

## 5) Run PluRel-16B Inference (No Training)

Recommended environment and CPU-safe settings:

```bash
cd /home/pc0618/relational-transformer
source "$HOME/.cargo/env"
source .venv-rt-cpu-smoke/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

### Context length 2048

```bash
PYTHONPATH=. python scripts/rt_tabarena_sweep.py \
  --manifest_csv /home/pc0618/relbench/results/tabarena_all51/tabarena_manifest_no_multiclass.csv \
  --output_csv /home/pc0618/relbench/results/tabarena_all51/tabarena_rt_plurel_16b_split_ctx2048_eval200.csv \
  --resume --preprocess --embed \
  --load_ckpt_path /home/pc0618/scratch/rt_ckpts_plurel/synthetic-pretrain_rdb_512_size_16b.pt \
  --train_steps 0 --skip_train_eval --skip_val_eval \
  --eval_batches 200 \
  --torch_num_threads 1 \
  --seq_len 2048 --max_bfs_width 16 --batch_size 8 \
  --num_blocks 12 --d_model 256 --num_heads 8 --d_ff 1024 \
  --disable_full_attention --use_qk_norm \
  --seed 42
```

### Context length 4096

```bash
PYTHONPATH=. python scripts/rt_tabarena_sweep.py \
  --manifest_csv /home/pc0618/relbench/results/tabarena_all51/tabarena_manifest_no_multiclass.csv \
  --output_csv /home/pc0618/relbench/results/tabarena_all51/tabarena_rt_plurel_16b_split_ctx4096_eval200.csv \
  --resume --preprocess --embed \
  --load_ckpt_path /home/pc0618/scratch/rt_ckpts_plurel/synthetic-pretrain_rdb_512_size_16b.pt \
  --train_steps 0 --skip_train_eval --skip_val_eval \
  --eval_batches 200 \
  --torch_num_threads 1 \
  --seq_len 4096 --max_bfs_width 16 --batch_size 4 \
  --num_blocks 12 --d_model 256 --num_heads 8 --d_ff 1024 \
  --disable_full_attention --use_qk_norm \
  --seed 42
```

If memory is tight at `4096`, reduce `--batch_size` to `2`.

## 6) Confirm Single-Node Context

After patching the RT script to `split-*`, a quick check for one sample:

```bash
cd /home/pc0618/relational-transformer
PYTHONPATH=. python - <<'PY'
import numpy as np
from rt.data import RelationalDataset

ds = RelationalDataset(
    tasks=[("tabarena-apsfailure", "split-0", "target", "test", [])],
    batch_size=1,
    seq_len=4096,
    rank=0,
    world_size=1,
    max_bfs_width=16,
    embedding_model="all-MiniLM-L12-v2",
    d_text=384,
    seed=42,
)
item = ds[0]
mask = (~item["is_padding"][0]).cpu().numpy()
uniq = np.unique(item["node_idxs"][0].cpu().numpy()[mask])
print("unique_nodes_in_context =", len(uniq))
assert len(uniq) == 1
print("OK")
PY
```
