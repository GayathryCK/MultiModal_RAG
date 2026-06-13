import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, degree
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.suppress_warnings import *

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NODES_CSV, EDGES_CSV, GNN_EMB_NPY,
    GNN_INPUT_DIM, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM,
    GNN_EPOCHS, GNN_LR, GRAPH_DB_DIR
)

class GCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.heads  = 4
        self.conv1  = GATConv(input_dim,  hidden_dim, heads=self.heads,
                              dropout=0.3, concat=True)
        self.conv2  = GATConv(hidden_dim * self.heads, output_dim,
                              heads=1, dropout=0.3, concat=False)
        self.bn1    = nn.BatchNorm1d(hidden_dim * self.heads)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim,  hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggr="mean")
        self.bn1   = nn.BatchNorm1d(hidden_dim)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def link_prediction_loss(embeddings: torch.Tensor,
                          edge_index: torch.Tensor,
                          num_nodes: int,
                          num_neg: int = 1) -> torch.Tensor:
    emb = F.normalize(embeddings, p=2, dim=1)
    src = edge_index[0]
    dst = edge_index[1]
    pos_sim = (emb[src] * emb[dst]).sum(dim=1)
    pos_loss = (1 - pos_sim).mean()
    neg_src = torch.randint(0, num_nodes,
                            (edge_index.size(1) * num_neg,),
                            device=embeddings.device)
    neg_dst = torch.randint(0, num_nodes,
                            (edge_index.size(1) * num_neg,),
                            device=embeddings.device)
    neg_sim  = (emb[neg_src] * emb[neg_dst]).sum(dim=1)
    neg_loss = F.relu(neg_sim - 0.3).mean()
    return pos_loss + neg_loss

def load_graph_data() -> tuple[Data, pd.DataFrame, pd.DataFrame]:
    nodes_df = pd.read_csv(NODES_CSV)
    edges_df = pd.read_csv(EDGES_CSV)
    logger.info(f"  Nodes loaded : {len(nodes_df)}")
    logger.info(f"  Edges loaded : {len(edges_df)}")
    node_features = np.load(GNN_EMB_NPY)
    logger.info(f"  Feature matrix: {node_features.shape}")
    src = torch.tensor(edges_df["src_idx"].values, dtype=torch.long)
    dst = torch.tensor(edges_df["dst_idx"].values, dtype=torch.long)
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ], dim=0)
    edge_index, _ = add_self_loops(edge_index,
                                   num_nodes=len(nodes_df))
    x    = torch.tensor(node_features, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    logger.info(f"  PyG Data: x={data.x.shape}, "
                f"edge_index={data.edge_index.shape}")
    return data, nodes_df, edges_df

def train_gnn(model_type: str = "gcn") -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GNN ({model_type.upper()}) on {device}")
    data, nodes_df, edges_df = load_graph_data()
    data = data.to(device)
    num_nodes  = data.x.shape[0]
    input_dim  = data.x.shape[1]
    logger.info(f"  Nodes: {num_nodes} | "
                f"Input dim: {input_dim} | "
                f"Output dim: {GNN_OUTPUT_DIM}")
    if model_type == "gat":
        model = GATEncoder(input_dim, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM)
    elif model_type == "sage":
        model = SAGEEncoder(input_dim, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM)
    else:
        model = GCNEncoder(input_dim, GNN_HIDDEN_DIM, GNN_OUTPUT_DIM)
    model     = model.to(device)
    optimizer = Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=20, gamma=0.5)
    logger.info(f"  Model parameters: "
                f"{sum(p.numel() for p in model.parameters()):,}")
    best_loss    = float("inf")
    best_embeddings = None
    history      = []
    model.train()
    pbar = tqdm(range(GNN_EPOCHS), desc=f"Training {model_type.upper()}")
    for epoch in pbar:
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        loss = link_prediction_loss(
            embeddings, data.edge_index, num_nodes
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_val = loss.item()
        history.append(loss_val)
        pbar.set_postfix({"loss": f"{loss_val:.4f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.5f}"})
        if loss_val < best_loss:
            best_loss       = loss_val
            best_embeddings = embeddings.detach().cpu().numpy()
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1:3d}/{GNN_EPOCHS} "
                        f"| Loss: {loss_val:.4f} "
                        f"| Best: {best_loss:.4f}")
    logger.success(f"Training complete! Best loss: {best_loss:.4f}")
    np.save(GNN_EMB_NPY, best_embeddings)
    logger.success(f"Saved refined embeddings "
                   f"{best_embeddings.shape} → {GNN_EMB_NPY}")
    history_path = GRAPH_DB_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"model": model_type,
                   "epochs": GNN_EPOCHS,
                   "best_loss": best_loss,
                   "history": history}, f, indent=2)
    logger.success(f"Saved training history → {history_path}")
    return best_embeddings

def get_similar_nodes(query_node_id: str,
                       nodes_df: pd.DataFrame,
                       embeddings: np.ndarray,
                       top_k: int = 5) -> list[dict]:
    matches = nodes_df[nodes_df["node_id"] == query_node_id]
    if matches.empty:
        logger.warning(f"Node not found: {query_node_id}")
        return []
    query_idx = matches.iloc[0]["node_idx"]
    query_emb = embeddings[query_idx]
    norms     = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb_norm  = embeddings / norms
    q_norm    = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims      = emb_norm @ q_norm
    sims[query_idx] = -1
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        row = nodes_df.iloc[idx]
        results.append({
            "node_id"   : row["node_id"],
            "node_type" : row["node_type"],
            "name"      : row["name"],
            "similarity": float(sims[idx]),
        })
    return results

if __name__ == "__main__":
    import pandas as pd
    print("GNN Training Pipeline")
    embeddings = train_gnn(model_type="gcn")
    nodes_df = pd.read_csv(NODES_CSV)
    print(f"\nGNN Training Summary")
    print(f"  Model       : GCN")
    print(f"  Input dim   : {GNN_INPUT_DIM}")
    print(f"  Hidden dim  : {GNN_HIDDEN_DIM}")
    print(f"  Output dim  : {GNN_OUTPUT_DIM}")
    print(f"  Epochs      : {GNN_EPOCHS}")
    print(f"  Embeddings  : {embeddings.shape}")
    print(f"\n── Similarity Test ──")
    test_nodes = [
        nodes_df[nodes_df["node_type"] == "PDF"].iloc[0]["node_id"]
        if len(nodes_df[nodes_df["node_type"] == "PDF"]) > 0 else None,
        nodes_df[nodes_df["node_type"] == "Video"].iloc[0]["node_id"]
        if len(nodes_df[nodes_df["node_type"] == "Video"]) > 0 else None,
    ]
    for test_node in test_nodes:
        if test_node is None:
            continue
        print(f"\n  Similar to: {test_node}")
        similar = get_similar_nodes(test_node, nodes_df,
                                    embeddings, top_k=3)
        for s in similar:
            print(f"    [{s['node_type']:8s}] {s['name'][:50]:50s} "
                  f"sim={s['similarity']:.3f}")
    print(f"\n  Saved to: {GNN_EMB_NPY}")
