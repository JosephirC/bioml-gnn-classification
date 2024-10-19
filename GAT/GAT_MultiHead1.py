from GAT_Base import GAT_Base
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear
import torch

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
    
    def save_model(self, path, best_accuracy, best_precision, best_rmse, best_hyperparameters, input_dim, output_dim):
        hyperparameters_to_save = {
            'nb_neurons': best_hyperparameters['nb_neurons'],
            'heads': best_hyperparameters['heads'],
            'dropout': best_hyperparameters['dropout'],
            'lr': best_hyperparameters['lr'],
            'weight_decay': best_hyperparameters['weight_decay']
        }

        torch.save({
            'model_state_dict': self.state_dict(),
            'best_hyperparameters': hyperparameters_to_save,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'best_accuracy': best_accuracy,
            'best_precision': best_precision,
            'best_rmse': best_rmse
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True)
        best_hyperparameters = checkpoint['best_hyperparameters']
        
        model = GAT_MultiHead1(
            checkpoint['input_dim'],
            best_hyperparameters['nb_neurons'],
            checkpoint['output_dim'],
            heads=best_hyperparameters['heads'],
            dropout=best_hyperparameters['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, best_hyperparameters
