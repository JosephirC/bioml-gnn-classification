import torch
import torch.nn as nn
# import GATConv
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super(GAT, self).__init__()

        # Première couche GATConv
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)

        # Deuxième couche GATConv pour la classification, concaténée pour obtenir une sortie unique par nœud
        self.gat2 = GATConv(hidden_channels* heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.dropout = dropout

    # co
    def forward3(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
  
    # original
    def forward2(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index)
        return x
    
    # mis
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x
    
    def fit(self, data, epochs):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = loss_func(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.forward(data.x, data.edge_index)
            
        


