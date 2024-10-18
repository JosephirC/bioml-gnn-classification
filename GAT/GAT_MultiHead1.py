from GAT_Base import GAT_Base
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear

class GAT_MultiHead1(GAT_Base):
    def __init__(self, in_channels, nb_neurons, out_channels, heads=2, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(in_channels, nb_neurons, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(nb_neurons * heads, nb_neurons, heads=heads, concat=True, dropout=dropout)
        self.l1 = Linear(nb_neurons * heads, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.l1(x)
        return F.log_softmax(x, dim=1)
