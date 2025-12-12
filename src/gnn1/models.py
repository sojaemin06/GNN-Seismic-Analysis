import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool

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
        edge_dim: int = 16, # [Modified] Increased feature dim
        global_dim: int = 8, # [Modified] Direction (4) + Modal (4)
        hidden_dim: int = 128, # [Modified] Increased capacity
        output_dim: int = 100, 
        num_layers: int = 3,
        heads: int = 2,
        dropout: float = 0.1
    ):
        super(PushoverGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout_ratio = dropout
        
        # --- 1. GNN Backbone (Encoder) ---
        # GATv2Conv supports edge attributes and is generally more expressive.
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input Layer
        # Node features are projected to hidden_dim * heads
        self.convs.append(GATv2Conv(node_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            # Input dim is previous hidden_dim * heads
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
            
        # Post-GNN dimension: hidden_dim * heads
        self.gnn_out_dim = hidden_dim * heads

        # --- 2. Decoder (MLP) ---
        # Input: Graph Embedding (Mean) + Global Feature (Direction)
        input_dim = self.gnn_out_dim + global_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, global_feat):
        """
        Args:
            x (Tensor): Node features [Num_Nodes, node_dim]
            edge_index (LongTensor): Edge indices [2, Num_Edges]
            edge_attr (Tensor): Edge features [Num_Edges, edge_dim]
            batch (LongTensor): Batch vector mapping nodes to graphs [Num_Nodes]
            global_feat (Tensor): Global features (Analysis Direction) [Batch_Size, global_dim]
            
        Returns:
            Tensor: Predicted Pushover Curve [Batch_Size, output_dim]
        """
        
        # 1. Message Passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            
        # 2. Readout (Global Pooling)
        # Aggregates node features to graph-level representation
        graph_embed = global_mean_pool(x, batch)  # [Batch_Size, gnn_out_dim]
        
        # 3. Fusion with Global Context
        # global_feat should be [Batch_Size, global_dim]
        if global_feat.dim() == 1:
            global_feat = global_feat.unsqueeze(0) # Handle single sample case
            
        combined = torch.cat([graph_embed, global_feat], dim=1) # [Batch_Size, gnn_out_dim + global_dim]
        
        # 4. Decoding
        out = self.decoder(combined) # [Batch_Size, output_dim]
        
        return out

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