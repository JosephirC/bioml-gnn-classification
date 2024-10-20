import torch 
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MaxPool1d
from GNN_Base import GNN_Base


class GCN_var1(GNN_Base):
    def __init__(self, in_features, out_features, n_neurons, n_layer_mlp, n_layer_conv):
        torch.manual_seed(0)
        super().__init__()
        self.conv1 = GCNConv(in_features, n_neurons, add_self_loops=False)
        self.convs = []
        for i in range(n_layer_conv):
            self.convs.append(GCNConv(n_neurons, n_neurons, add_self_loops=False))
        self.fc1 = Linear(n_neurons, n_neurons)
        self.fcs = []
        for i in range(n_layer_mlp):
            self.fcs.append(Linear(n_neurons, n_neurons))                
        self.fc2 = Linear(n_neurons, out_features)
    
    def forward(self, x, A):
        x = self.conv1(x, A)
        for conv in self.convs:
            x = conv(x, A)
        x = self.fc1(x)
        x = F.relu(x)
        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)
    
   