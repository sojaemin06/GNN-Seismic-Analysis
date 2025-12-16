import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_add_pool

class PushoverGNN(nn.Module):
    """
    1차 모드 지배형 RC 모멘트 골조의 푸쉬오버 곡선을 예측하는 GNN 모델.
    
    Architecture:
    1. Encoder (GNN Backbone): GATv2Conv Layers processing Node & Edge features.
    2. Pooling: Concatenation of Global Mean & Add Pooling to capture both average and total structural properties.
    3. Fusion: Concatenate Graph Embedding with Global Context (Analysis Direction).
    4. Decoder (MLP): Predicts the 100-point Pushover Curve.
    """
    def __init__(
        self, 
        node_dim: int = 6, 
        edge_dim: int = 16, 
        global_dim: int = 8, 
        hidden_dim: int = 128, # [Optimized] Increased capacity
        output_dim: int = 100, 
        num_layers: int = 4,   # [Optimized] Deeper network
        heads: int = 4,        # [Optimized] More attention heads
        dropout: float = 0.2   # [Optimized] Higher dropout
    ):
        super(PushoverGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_ratio = dropout
        
        # --- 1. GNN Backbone (Encoder) ---
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input Layer
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            
        self.gnn_out_dim = hidden_dim * heads

        # --- 2. Decoder (MLP) ---
        # Input: Graph Embedding (Mean + Add) + Global Feature
        # Using both Mean and Add pooling enriches the representation
        self.pool_dim = self.gnn_out_dim * 2 
        input_dim = self.pool_dim + global_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feat):
        # 1. Message Passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            
        # 2. Readout (Global Pooling)
        # Combine Mean (average properties) and Add (total mass/stiffness)
        embed_mean = global_mean_pool(x, batch)
        embed_add = global_add_pool(x, batch)
        graph_embed = torch.cat([embed_mean, embed_add], dim=1)
        
        # 3. Fusion
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
            
        combined = torch.cat([graph_embed, global_feat], dim=1)
        
        # 4. Decoding
        out = self.decoder(combined)
        return out


class BaselineGCN(nn.Module):
    """
    Comparison Model: Standard GCN (Graph Convolutional Network).
    Differs from PushoverGNN (GAT) in that it uses isotropic aggregation (GCNConv)
    and typically ignores edge features (or handles them less effectively).
    Demonstrates the value of Attention and Edge Features.
    """
    def __init__(
        self, 
        node_dim: int = 6, 
        edge_dim: int = 16, # Not used in standard GCNConv
        global_dim: int = 8, 
        hidden_dim: int = 128, 
        output_dim: int = 100, 
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super(BaselineGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_ratio = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # GCNConv doesn't natively support edge_dim in the same way GAT does
        # It aggregates node features only.
        
        # Input Layer
        self.convs.append(GCNConv(node_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        self.gnn_out_dim = hidden_dim

        # Decoder
        self.pool_dim = self.gnn_out_dim * 2 
        input_dim = self.pool_dim + global_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feat):
        # Note: edge_attr is IGNORED in standard GCNConv forward
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # No edge_attr
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            
        embed_mean = global_mean_pool(x, batch)
        embed_add = global_add_pool(x, batch)
        graph_embed = torch.cat([embed_mean, embed_add], dim=1)
        
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)
            
        combined = torch.cat([graph_embed, global_feat], dim=1)
        return self.decoder(combined)


class SimpleMLP(nn.Module):
    """
    Comparison Model: Simple MLP that ignores graph topology.
    It aggregates node/edge features via simple averaging and predicts the curve.
    Used to demonstrate the value of GNN.
    """
    def __init__(
        self, 
        node_dim: int = 6, 
        edge_dim: int = 16, # Not used for topology, but for fairness maybe?
        global_dim: int = 8, 
        hidden_dim: int = 128, 
        output_dim: int = 100,
        dropout: float = 0.2
    ):
        super(SimpleMLP, self).__init__()
        
        # Naive aggregation: Mean of Node features + Global features
        # We ignore edge features here to simulate "loss of topological/connection info".
        # Or we could pool edges too, but let's keep it simple: "Node Stats Only".
        input_dim = node_dim + global_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feat):
        # Global Mean Pool of Node Features ONLY (Ignoring Edges/Topology)
        # x: [Num_Nodes, node_dim] -> [Batch_Size, node_dim]
        node_pool = global_mean_pool(x, batch)
        
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0)

        # Concatenate Node Stats + Global Context
        combined = torch.cat([node_pool, global_feat], dim=1)
        
        return self.net(combined)


if __name__ == "__main__":
    # --- Simple Test for Dimensionality Check ---
    print("Testing PushoverGNN Model Dimensions...")
    
    # Mock Data
    num_nodes = 20
    num_edges = 40
    batch_size = 4
    
    # Node Feats: [x, y, z, is_base, mass, node_degree]
    x = torch.randn(num_nodes, 6)
    # Edge Index: random connectivity
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    # Edge Feats: 12 dim
    edge_attr = torch.randn(num_edges, 12)
    # Batch: assign nodes to 4 graphs randomly
    batch = torch.randint(0, batch_size, (num_nodes,)).sort()[0]
    # Global Feat: 4 dim one-hot
    global_feat = torch.randn(batch_size, 4)
    
    model = PushoverGNN(node_dim=6, edge_dim=12, global_dim=4, hidden_dim=32, heads=2)
    
    # Forward Pass
    try:
        output = model(x, edge_index, edge_attr, batch, global_feat)
        print(f"Input Node Shape: {x.shape}")
        print(f"Input Edge Shape: {edge_attr.shape}")
        print(f"Global Feat Shape: {global_feat.shape}")
        print(f"Output Shape: {output.shape}") # Should be [4, 100]
        
        assert output.shape == (batch_size, 100)
        print("✅ Model Forward Pass Successful!")
        
    except Exception as e:
        print(f"❌ Model Forward Pass Failed: {e}")