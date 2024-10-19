import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from GAT_Base import GAT_Base

class GAT_Pooling(GAT_Base):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)