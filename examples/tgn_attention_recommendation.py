"""
Temporal recommendation baseline: TGN + attention (TransformerConv).

This script ports the core TGB2 "TGN + GraphAttentionEmbedding" training recipe
to RelBench's RecommendationTask interface.

Key idea:
- Use a *history* event table from the RelBench database (not the task tables,
  since task tables contain *future* targets by construction).
- Train a Temporal Graph Network (TGN) on observed (src, dst, time) interactions.
- Evaluate at RelBench split cutoffs by producing top-k dst predictions per src.

This is intended as an *example* script (for small/medium tasks). For large
dst spaces, full ranking can be expensive.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.seed import seed_everything
from torch_geometric.utils import scatter
from tqdm import tqdm

from relbench.base import Dataset, RecommendationTask, TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.lin = Linear(1, self.out_channels)

    def reset_parameters(self) -> None:
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = int(raw_msg_dim + 2 * memory_dim + time_dim)

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor) -> Tensor:
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, msg.size(-1)))
        max_t = scatter(t, index, dim=0, dim_size=dim_size, reduce="max")
        is_max = t == max_t[index]
        pos = torch.arange(t.size(0), device=t.device, dtype=torch.long)
        pos = torch.where(is_max, pos, pos.new_full(pos.shape, -1))
        argmax = scatter(pos, index, dim=0, dim_size=dim_size, reduce="max")

        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax >= 0
        out[mask] = msg[argmax[mask]]
        return out


class LastNeighborLoader:
    """Keeps the last K neighbors per node (undirected)."""

    def __init__(self, num_nodes: int, size: int, *, device: Optional[torch.device] = None):
        self.size = int(size)
        self.neighbors = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.reset_state()

    def reset_state(self) -> None:
        self.cur_e_id = 0
        self.e_id.fill_(-1)

    def __call__(self, n_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]
        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor) -> None:
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0), device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat([self.neighbors[n_id, : self.size], dense_neighbors], dim=-1)

        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)


class TGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        *,
        message_module: torch.nn.Module,
        aggregator_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.raw_msg_dim = int(raw_msg_dim)
        self.memory_dim = int(memory_dim)
        self.time_dim = int(time_dim)

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.memory_updater = GRUCell(self.msg_module.out_channels, memory_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))
        self.register_buffer("_assoc", torch.empty(num_nodes, dtype=torch.long))

        self._reset_message_store()

    def reset_state(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()
        self._reset_message_store()

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            memory, last_update = self._get_updated_memory(n_id)
        else:
            memory, last_update = self.memory[n_id], self.last_update[n_id]
        return memory, last_update

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor) -> None:
        n_id = torch.cat([src, dst]).unique()
        if self.training:
            self._update_memory(n_id)
            self._update_msg_store(src, dst, t, raw_msg)
        else:
            self._update_msg_store(src, dst, t, raw_msg)
            self._update_memory(n_id)

    def _reset_message_store(self) -> None:
        i = self.last_update.new_empty((0,))
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        self.msg_store = {j: (i, i, i, msg) for j in range(self.num_nodes)}

    def _update_msg_store(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor) -> None:
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            self.msg_store[i] = (src[idx], dst[idx], t[idx], raw_msg[idx])

    def _update_memory(self, n_id: Tensor) -> None:
        memory, last_update = self._get_updated_memory(n_id)
        self.memory[n_id] = memory
        self.last_update[n_id] = last_update

    def _get_updated_memory(self, n_id: Tensor) -> tuple[Tensor, Tensor]:
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        data = [self.msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)

        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.to(raw_msg.dtype if raw_msg.numel() else torch.float32))
        msg = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)

        aggr = self.aggr_module(msg, self._assoc[src], t, n_id.size(0))
        memory = self.memory_updater(aggr, self.memory[n_id])

        dim_size = self.last_update.size(0)
        last_update = scatter(t, src, 0, dim_size, reduce="max")[n_id]
        return memory, last_update

    def train(self, mode: bool = True):
        if self.training and not mode:
            self._update_memory(torch.arange(self.num_nodes, device=self.memory.device))
            self._reset_message_store()
        return super().train(mode)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = int(msg_dim + time_enc.out_channels)
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be divisible by 2 (TransformerConv heads=2).")
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, x: Tensor, last_update: Tensor, edge_index: Tensor, t: Tensor, msg: Tensor) -> Tensor:
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


@dataclass(frozen=True)
class _EventSpec:
    table: str
    src_col: str
    dst_col: str
    time_col: str


def _infer_event_spec(
    db,
    task: RecommendationTask,
    *,
    task_name: str,
    event_table: Optional[str],
) -> _EventSpec:
    candidates: list[_EventSpec] = []
    for name, table in db.table_dict.items():
        if event_table is not None and name != event_table:
            continue
        if table.time_col is None:
            continue
        fks = table.fkey_col_to_pkey_table
        src_cols = [c for c, t in fks.items() if t == task.src_entity_table]
        dst_cols = [c for c, t in fks.items() if t == task.dst_entity_table]
        if not src_cols or not dst_cols:
            continue
        candidates.append(_EventSpec(table=name, src_col=src_cols[0], dst_col=dst_cols[0], time_col=table.time_col))

    if not candidates:
        raise RuntimeError(
            "Could not infer an event table with FK columns to both the task's "
            f"src table ({task.src_entity_table}) and dst table ({task.dst_entity_table}) "
            "and a time column. Specify one via --event_table."
        )

    # Heuristic: prefer tables whose name matches task name tokens (e.g.
    # `user-post-comment` should pick `comments` over `votes`).
    tokens = [t for t in str(task_name).lower().split("-") if t]
    action_token = tokens[-1] if tokens else ""

    def _score(spec: _EventSpec) -> tuple[int, int, int]:
        name_l = spec.table.lower()
        # Prioritize the single-word suffix (RelBench convention):
        # <src>-<dst>-<single_word>
        action_hit = 1 if action_token and action_token in name_l else 0
        token_hits = sum(1 for tok in tokens if tok in name_l)
        return (action_hit, token_hits, len(db.table_dict[spec.table]))

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _to_unix_seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts, utc=True)
    ns = ts.astype("int64").to_numpy()
    return (ns // 1_000_000_000).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-f1")
    parser.add_argument("--task", type=str, default="driver-race-compete")
    parser.add_argument("--event_table", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--mem_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_neg_train", type=int, default=50)
    parser.add_argument("--eval_dst_block_size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_num_threads(1)

    dataset: Dataset = get_dataset(args.dataset, download=True)
    task: RecommendationTask = get_task(args.dataset, args.task, download=True)
    if task.task_type != TaskType.LINK_PREDICTION:
        raise ValueError(f"Task {args.dataset}/{args.task} is not a link prediction task.")

    db = dataset.get_db()
    spec = _infer_event_spec(db, task, task_name=args.task, event_table=args.event_table)
    df = db.table_dict[spec.table].df[[spec.src_col, spec.dst_col, spec.time_col]].dropna()

    src_np = df[spec.src_col].astype("int64").to_numpy()
    dst_np = df[spec.dst_col].astype("int64").to_numpy()
    t_np = _to_unix_seconds(df[spec.time_col])

    order = np.argsort(t_np, kind="mergesort")
    src_np, dst_np, t_np = src_np[order], dst_np[order], t_np[order]

    num_src = task.num_src_nodes
    num_dst = task.num_dst_nodes
    if task.src_entity_table != task.dst_entity_table:
        dst_global_np = dst_np + num_src
        num_nodes = num_src + num_dst
    else:
        dst_global_np = dst_np
        num_nodes = max(num_src, num_dst)

    src = torch.from_numpy(src_np).to(device=device, dtype=torch.long)
    dst = torch.from_numpy(dst_global_np).to(device=device, dtype=torch.long)
    t = torch.from_numpy(t_np).to(device=device, dtype=torch.long)
    msg = torch.zeros((t.numel(), 0), device=device, dtype=torch.float32)

    val_ts = int(dataset.val_timestamp.timestamp())
    test_ts = int(dataset.test_timestamp.timestamp())
    val_cut = int(np.searchsorted(t_np, val_ts, side="left"))
    test_cut = int(np.searchsorted(t_np, test_ts, side="left"))

    def build_pred_at(timestamp_s: int) -> np.ndarray:
        target_table = task.get_table("val" if timestamp_s == val_ts else "test")
        target_ts = int(pd.to_datetime(target_table.df[task.time_col].iloc[0], utc=True).timestamp())
        if target_ts != timestamp_s:
            raise RuntimeError("This example assumes a single timestamp per split.")

        memory.eval()
        gnn.eval()
        memory.reset_state()
        neighbor_loader.reset_state()

        cutoff = val_cut if timestamp_s == val_ts else test_cut
        for start in range(0, cutoff, args.batch_size):
            end = min(start + args.batch_size, cutoff)
            src_b, dst_b, t_b = src[start:end], dst[start:end], t[start:end]
            msg_b = msg[start:end]
            memory.update_state(src_b, dst_b, t_b, msg_b)
            neighbor_loader.insert(src_b, dst_b)

        with torch.no_grad():
            n_id = torch.arange(num_nodes, device=device, dtype=torch.long)
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(num_nodes, device=device, dtype=torch.long)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t[e_id], msg[e_id])

            src_ids = torch.from_numpy(target_table.df[task.src_entity_col].astype("int64").to_numpy()).to(device)
            src_emb = z[assoc[src_ids]]

            if task.src_entity_table != task.dst_entity_table:
                dst_ids = torch.arange(num_dst, device=device, dtype=torch.long) + num_src
            else:
                dst_ids = torch.arange(num_dst, device=device, dtype=torch.long)
            dst_emb = z[assoc[dst_ids]]

            k = int(task.eval_k)
            topk_scores = src_emb.new_full((src_emb.size(0), k), float("-inf"))
            topk_idx = torch.full((src_emb.size(0), k), -1, device=device, dtype=torch.long)

            block = int(args.eval_dst_block_size)
            for start in range(0, dst_emb.size(0), block):
                end = min(start + block, dst_emb.size(0))
                scores = src_emb @ dst_emb[start:end].t()  # [B, block]
                cand_scores, cand_idx = torch.topk(scores, k=min(k, end - start), dim=1)
                cand_idx = cand_idx + start

                merged_scores = torch.cat([topk_scores, cand_scores], dim=1)
                merged_idx = torch.cat([topk_idx, cand_idx], dim=1)
                topk_scores, sel = torch.topk(merged_scores, k=k, dim=1)
                topk_idx = torch.gather(merged_idx, 1, sel)

            return topk_idx.detach().cpu().numpy()

    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=0,
        memory_dim=args.mem_dim,
        time_dim=args.time_dim,
        message_module=IdentityMessage(0, args.mem_dim, args.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=args.mem_dim,
        out_channels=args.emb_dim,
        msg_dim=0,
        time_enc=memory.time_enc,
    ).to(device)

    neighbor_loader = LastNeighborLoader(num_nodes, size=args.num_neighbors, device=device)
    optimizer = torch.optim.Adam(set(memory.parameters()) | set(gnn.parameters()), lr=args.lr)

    def train_epoch() -> float:
        memory.train()
        gnn.train()
        memory.reset_state()
        neighbor_loader.reset_state()

        total_loss = 0.0
        total_events = 0
        for start in tqdm(range(0, val_cut, args.batch_size), desc="train", leave=False):
            end = min(start + args.batch_size, val_cut)
            src_b, pos_dst_b, t_b = src[start:end], dst[start:end], t[start:end]
            msg_b = msg[start:end]

            # negatives are sampled from dst-type only
            if task.src_entity_table != task.dst_entity_table:
                neg_dst_b = torch.randint(num_src, num_src + num_dst, (pos_dst_b.size(0), args.num_neg_train), device=device)
            else:
                neg_dst_b = torch.randint(0, num_dst, (pos_dst_b.size(0), args.num_neg_train), device=device)

            n_id = torch.cat([src_b, pos_dst_b, neg_dst_b.reshape(-1)]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = torch.empty(num_nodes, device=device, dtype=torch.long)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t[e_id], msg[e_id])

            src_z = z[assoc[src_b]]
            pos_z = z[assoc[pos_dst_b]]
            neg_z = z[assoc[neg_dst_b]]

            pos_score = (src_z * pos_z).sum(dim=-1)  # [B]
            neg_score = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)  # [B, N]

            loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            memory.update_state(src_b, pos_dst_b, t_b, msg_b)
            neighbor_loader.insert(src_b, pos_dst_b)
            memory.detach()

            total_loss += float(loss.detach()) * (end - start)
            total_events += (end - start)

        return total_loss / max(total_events, 1)

    print(f"[event_table] {spec.table} (src_col={spec.src_col}, dst_col={spec.dst_col}, time_col={spec.time_col})")
    print(f"[events] total={t.numel():,} train(<val)={val_cut:,} hist_for_test(<test)={test_cut:,}")

    best_val = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch()
        val_pred = build_pred_at(val_ts)
        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        tune = "link_prediction_map"
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val={val_metrics}")
        if tune in val_metrics and val_metrics[tune] >= best_val:
            best_val = float(val_metrics[tune])
            best_state = {"memory": memory.state_dict(), "gnn": gnn.state_dict()}

    if best_state is not None:
        memory.load_state_dict(best_state["memory"])
        gnn.load_state_dict(best_state["gnn"])

    val_pred = build_pred_at(val_ts)
    val_metrics = task.evaluate(val_pred, task.get_table("val"))
    test_pred = build_pred_at(test_ts)
    test_metrics = task.evaluate(test_pred)

    print(f"Best val:  {val_metrics}")
    print(f"Best test: {test_metrics}")


if __name__ == "__main__":
    main()
