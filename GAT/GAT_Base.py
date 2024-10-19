import torch
import torch.nn as nn
from sklearn.metrics import precision_score, root_mean_squared_error

class GAT_Base(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        pass

    def fit(self, data, epochs, lr=0.01, wd=5e-4, optimizer_fn=torch.optim.Adam):
        loss_func = nn.CrossEntropyLoss()
        optimizer = optimizer_fn(self.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self.forward(data.x, data.edge_index)
            loss = loss_func(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

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

        precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='weighted', zero_division=0)
        rmse = root_mean_squared_error(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

        return acc, precision, rmse
