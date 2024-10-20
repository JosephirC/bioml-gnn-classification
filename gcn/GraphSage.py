import torch
from GNN_Base import GNN_Base
from torch_geometric.nn import SAGEConv
from torch.nn import Linear
import torch.nn.functional as F


class GraphSage(GNN_Base):
    def __init__(self, in_features, out_features, n_neurons):
        torch.manual_seed(0)
        super().__init__()
        self.conv1 = SAGEConv(in_features, n_neurons)
        self.conv2 = SAGEConv(n_neurons, n_neurons)
        self.fc1 = Linear(n_neurons, n_neurons*2)
        self.fc2 = Linear(n_neurons*2, n_neurons*4)
        self.fc3 = Linear(n_neurons*4, out_features)


    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.conv2(x, A)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)
