import torch.nn.functional as F
from torch_geometric.nn import GATConv
from GAT_Base import GAT_Base
from torch.nn import MaxPool1d
import torch

class GAT_Pooling(GAT_Base):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout)
        self.pooling = MaxPool1d(kernel_size=2, stride=2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat3(x, edge_index)

        assert x.dim() == 2, f"Expected 2D tensor, but got {x.dim()}D tensor"
        assert x.size(1) % 2 == 0, f"Expected even number of features, but got {x.size(1)}"

        x = x.unsqueeze(0)
        x = self.pooling(x)
        x = x.squeeze(0)
        return F.log_softmax(x, dim=1)
    
    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True)
        best_hyperparameters = checkpoint['best_hyperparameters']
        
        model = GAT_Pooling(
            checkpoint['input_dim'],
            best_hyperparameters['nb_neurons'],
            checkpoint['output_dim'],
            dropout=best_hyperparameters['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, best_hyperparameters