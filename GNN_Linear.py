import torch 
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class GNN_Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, n_neurons):
        super(GNN_Linear, self).__init__()
        self.fc1 = Linear(in_features, n_neurons)
        self.fc2 = Linear(n_neurons, out_features)

    
    def forward(self, x, A):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.forward(data.x, data.edge_index)
