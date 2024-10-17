import torch
import copy
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import precision_score
from sklearn.metrics import root_mean_squared_error

class GNN_Base(torch.nn.Module):
    """
    Abstract class
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, A):
        pass

    def fit(self, data, epochs, lr=0.01, wd=5e-4):
        torch.manual_seed(0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

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
    
    def save(self,filepath):
        torch.save(self.state_dict(), filepath)

    def load(self,filepath):
        self.load_state_dict(torch.load(filepath))
        return self


    def validate_model(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.cpu()
        data= data.cpu()

        pred = self.forward(data.x, data.edge_index).argmax(dim=1)
        print(pred)

        correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        acc = int(correct) / int(data.val_mask.sum())

        precision = precision_score(data.y[data.val_mask], pred[data.val_mask], average='weighted', zero_division=0)

        rmse = root_mean_squared_error(data.y[data.val_mask], pred[data.val_mask])

        model = model.to(device)
        data = data.to(device)

        return acc, precision, rmse
    
    def test_model(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.cpu()
        data= data.cpu()

        pred = self.forward(data.x, data.edge_index).argmax(dim=1)
        print(pred)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

        precision = precision_score(data.y[data.test_mask], pred[data.test_mask], average='weighted', zero_division=0)

        rmse = root_mean_squared_error(data.y[data.test_mask], pred[data.test_mask])

        model = model.to(device)
        data = data.to(device)

        return acc, precision, rmse