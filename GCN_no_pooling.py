import torch 
import torch.utils
import torch.utils.data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import MaxPool1d
from torch.utils.tensorboard import SummaryWriter 


class GCN_no_pooling(torch.nn.Module):
    def __init__(self, in_features, out_features, n_neurons, lr = 0.01, wd=5e-4):
        super(GCN_no_pooling, self).__init__()
        self.conv1 = GCNConv(in_features, n_neurons, add_self_loops=False)
        self.conv2 = GCNConv(n_neurons, n_neurons, add_self_loops=False)
        self.fc1 = Linear(n_neurons, n_neurons)
        self.fc2 = Linear(n_neurons, out_features)
        self.pooling = MaxPool1d(kernel_size=2, stride=2)
        self.lr = lr
        self.wd = wd
    
    def forward(self, x, A):
        x = self.conv1(x, A)
        x = F.relu(x)
        x = self.conv2(x, A)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.relu(x)
    
    def fit(self, data, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self = self.to(device)
        data = data.to(device)
    
        writer = SummaryWriter()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            writer.add_scalar('Loss/train', loss, epoch)
            optimizer.step()
            optimizer.zero_grad()
        
        writer.flush()
        writer.close()

        return self
