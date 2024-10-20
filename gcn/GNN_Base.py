import torch
import copy
from torch.utils.tensorboard import SummaryWriter 
from sklearn.metrics import precision_score
from sklearn.metrics import root_mean_squared_error
from EarlyStopping import EarlyStopping

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
        es = EarlyStopping(patience=1000)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        writer = SummaryWriter()

        for epoch in range(epochs+1):
            out = self.forward(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            writer.add_scalar('Loss/train', loss, epoch)
            # acc, precision, rmse = self.validate_model(data)
            # writer.add_scalar('Validation Accuracy/train', acc, epoch)
            optimizer.step()
            optimizer.zero_grad()

            if es.early_stop(criterion(out[data.val_mask], data.y[data.val_mask]), self):
                return es.get_best_model()
            
        writer.flush()
        writer.close()

        return self
    
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


        print("pred : " , pred[data.val_mask].shape)
        print("data.y : " , data.y[data.val_mask].shape)


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

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

        precision = precision_score(data.y[data.test_mask], pred[data.test_mask], average='weighted', zero_division=0)

        rmse = root_mean_squared_error(data.y[data.test_mask], pred[data.test_mask])

        model = model.to(device)
        data = data.to(device)

        return acc, precision, rmse