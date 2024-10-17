import torch 
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MaxPool1d
import copy
from torch.utils.tensorboard import SummaryWriter 


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_features, n_neurons):
        super().__init__()
        self.conv1 = GCNConv(in_features, n_neurons, add_self_loops=False)
        self.conv2 = GCNConv(n_neurons//2, n_neurons//2, add_self_loops=False)
        self.fc1 = Linear(n_neurons//4, n_neurons//4)
        self.fc2 = Linear(n_neurons//4, out_features)
        self.pooling = MaxPool1d(kernel_size=2, stride=2)

    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.pooling(x)   
        x = self.conv2(x, A)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)
    
    def fit(self, data, epochs):
        torch.manual_seed(0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

        best_model = copy.deepcopy(self)
        min_loss = None

        writer = SummaryWriter()

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            if min_loss is None or loss < min_loss:
                min_loss = loss
                best_model = copy.deepcopy(self)
            writer.add_scalar('Loss/train', loss, epoch)
            optimizer.step()
            optimizer.zero_grad()
        writer.flush()
        writer.close()

        return best_model
