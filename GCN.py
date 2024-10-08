import torch 
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.fc1 = GCNConv(in_features, out_features, add_self_loops=False)
    
    def forward(self, x, A):
        x = self.fc1(x, A)
        return x
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        self.train()
        for epoch in range(epochs+1):
            optimizer.zero_grad()
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')

        return self.forward(data.x, data.edge_index)
