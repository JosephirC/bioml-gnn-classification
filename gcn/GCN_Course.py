from GNN_Base import GNN_Base
from torch_geometric.nn import GCNConv

class GCN_Course(GNN_Base):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = GCNConv(in_features, out_features, add_self_loops=False)
    
    def forward(self, x, A):
        x = self.fc1(x, A)
        return x