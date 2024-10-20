import torch 
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MaxPool1d
from GNN_Base import GNN_Base


class GCN(GNN_Base):
    def __init__(self, in_features, out_features, n_neurons):
        torch.manual_seed(0)
        super().__init__()
        self.conv1 = GCNConv(in_features, n_neurons, add_self_loops=False)
        self.conv2 = GCNConv(n_neurons//2, n_neurons//2, add_self_loops=False)
        self.fc1 = Linear(n_neurons//4, n_neurons//2)
        self.fc2 = Linear(n_neurons//2, out_features)
        self.pooling = MaxPool1d(kernel_size=2, stride=2)

    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.conv2(x, A)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)        #  essayer d'autres fonction d'activ/ pas finir par une fonction d'activ 
        return F.relu(x)
    
   