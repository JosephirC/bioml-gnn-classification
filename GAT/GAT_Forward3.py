from GAT_Base import GAT_Base
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch

# mis forward

class GAT_Forward3(GAT_Base):
    def __init__(self, in_channels, nb_neurons, out_channels, heads=1, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(in_channels, nb_neurons, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(nb_neurons* heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

        # print(f"param are in_channels: {in_channels}, nb_neurons: {nb_neurons}, out_channels: {out_channels}, heads: {heads}, dropout: {dropout}")

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x
    
    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True)
        best_hyperparameters = checkpoint['best_hyperparameters']
        
        model = GAT_Forward3(
            checkpoint['input_dim'],
            best_hyperparameters['nb_neurons'],
            checkpoint['output_dim'],
            dropout=best_hyperparameters['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, best_hyperparameters