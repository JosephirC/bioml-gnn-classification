from GAT_Base import GAT_Base
import torch.nn.functional as F

# co forward

class GAT_Forward2(GAT_Base):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super().__init__()

        print(f"param are in_channels: {in_channels}, hidden_channels: {hidden_channels}, out_channels: {out_channels}, heads: {heads}, dropout: {dropout}")


    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)