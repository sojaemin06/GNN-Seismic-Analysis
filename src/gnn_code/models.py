import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GNN_Pushover(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=4):
        super(GNN_Pushover, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=edge_dim)

        self.pool = global_mean_pool

        self.fc1 = torch.nn.Linear(hidden_channels, hidden_channels * 2)
        self.fc2 = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # GNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        # Global pooling
        x = self.pool(x, batch)

        # Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        
        return x
