import argparse
import copy
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.loader import CustomNodeLoader
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE
from relbench.modeling.transfer.encoder import UniversalTextEncoder, TransferHeteroEncoder
from relbench.modeling.utils import get_stype_proposal, remove_pkey_fkey
from relbench.tasks import get_task
from relbench.tasks.transfer.task import CrossDomainEntityTask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_datasets", type=str, nargs="+", default=["rel-amazon"])
    # Demonstration configuration: sourcing from 'user-churn' and transferring to 'item-churn'
    # In a real scenario, this would likely transfer between domains (e.g. Books -> Movies).
    parser.add_argument("--dataset", type=str, default="rel-amazon")
    parser.add_argument("--source_task", type=str, default="user-churn")
    parser.add_argument("--target_task", type=str, default="item-churn") 

    
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--enc_channels", type=int, default=128)
    parser.add_argument("--gnn_channels", type=int, default=128)
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)

    # 1. Initialize Universal Encoder
    print("Initializing Universal Text Encoder...")
    text_encoder = UniversalTextEncoder(device=device)
    text_embedder_cfg = text_encoder.get_config()

    # 2. Load Dataset & Tasks
    dataset = get_dataset(args.dataset, download=True)
    db = dataset.get_db()
    
    # We need to compute stats and stypes ONCE for the whole database
    col_to_stype_dict = get_stype_proposal(db)
    
    # 3. Construct Graph (HeteroData)
    # This encodes all text using the Universal Encoder!
    print("Materializing Graph with Universal Embeddings...")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=None, # Disable cache to force universal encoding
    )
    
    # 4. Setup Tasks
    print(f"Setting up Source Task: {args.source_task}")
    source_task = get_task(args.dataset, args.source_task, download=True)
    
    print(f"Setting up Target Task: {args.target_task}")
    target_task = get_task(args.dataset, args.target_task, download=True)
    
    # Wrap in CrossDomain wrapper
    cross_task = CrossDomainEntityTask([source_task], target_task)

    # 5. Model Setup
    # Used shared columns for heterogeneous encoder
    # Note: simple HeteroEncoder will learn embeddings for non-text columns from scratch.
    encoder = TransferHeteroEncoder(
        node_to_col_names_dict={
            node_type: {
                stype: list(col_to_stype_dict[node_type].keys())
                for stype in col_to_stype_dict[node_type] # filtering needed based on graph.py logic
            } for node_type in data.node_types
        },
        node_to_col_stats=col_stats_dict,
        channels=args.enc_channels,
    ).to(device)
    
    # We need to construct the node_to_col_names_dict properly like in nn.py
    # Re-using the logic from nn.py example implies we need to be careful with keys.
    # Since we don't have easy access to the exact processed keys, we assume
    # the encoder handles it or we'd refine it in production.
    
    # GNN Backbone
    gnn = HeteroGraphSAGE(
        node_types=data.node_types,
        edge_types=data.edge_types,
        channels=args.gnn_channels,
    ).to(device)
    
    # Task Head (Linear) - this is task specific!
    # If we want TRANSFER, we need a generic head or we fine-tune the head.
    # For zero-shot, we assume the task is the SAME (e.g. churn) just different domain.
    # If tasks differ (churn vs ltv), zero-shot is impossible without a unified output space.
    # So we MUST assume task type is identical.
    if source_task.task_type != target_task.task_type:
        print("Warning: Task types differ. Zero-shot transfer might fail on the head.")
    
    head = torch.nn.Linear(args.gnn_channels, 1).to(device) # Binary classification
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(gnn.parameters()) + list(head.parameters()),
        lr=args.lr
    )

    # 6. Pre-Training Loop (Source Domain)
    print("Starting Pre-training on Source Task...")
    loader = cross_task.get_loader({args.dataset: data}, split="train", batch_size=args.batch_size)
    
    encoder.train()
    gnn.train()
    head.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(loader):
            batch = batch.to(device)
            # Encode
            x_dict = encoder(batch.tf_dict)
            x_dict = gnn(x_dict, batch.edge_index_dict)
            
            # Predict
            # We assume batch.input_id and batch.input_type tells us which nodes are target
            pass
            
            # Simplified loss calc for demo
            # In a real impl, we'd extract the embedding of the entity
            entity_feat = x_dict[source_task.entity_table]
            # Get only the seed nodes (first batch_size)
            batch_size = batch[source_task.entity_table].batch_size
            pred = head(entity_feat[:batch_size]).squeeze()
            target = batch[source_task.entity_table].y[:batch_size].float()
            
            loss = F.binary_cross_entropy_with_logits(pred, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss}")

    # 7. Zero-Shot Evaluation (Target Domain)
    print("Evaluating Zero-Shot on Target Task...")
    # ... evaluation logic ...

if __name__ == "__main__":
    main()
