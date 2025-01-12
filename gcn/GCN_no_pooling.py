import torch 
import torch.utils
import torch.utils.data
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MaxPool1d
from GNN_Base import GNN_Base


class GCN_no_pooling(GNN_Base):
    def __init__(self, in_features, out_features, n_neurons):
        torch.manual_seed(0)
        super().__init__()
        self.out = out_features
        self.conv1 = GCNConv(in_features, n_neurons, add_self_loops=False)
        self.conv2 = GCNConv(n_neurons, n_neurons, add_self_loops=False)
        self.fc1 = Linear(n_neurons, n_neurons)
        self.fc2 = Linear(n_neurons, out_features)
        self.pooling = MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.conv2(x, A)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)
