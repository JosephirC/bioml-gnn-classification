import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score
from sklearn.metrics import root_mean_squared_error

class GAT_Base(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.6):
        super().__init__()

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels* heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        pass

    def fit(self, data, epochs, lr, wd):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = loss_func(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.forward(data.x, data.edge_index)
    
    def test_model(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.to(device)
        data = data.to(device)

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

        precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='weighted')
        rmse = root_mean_squared_error(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

        return acc, precision, rmse
