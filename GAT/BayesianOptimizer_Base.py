import optuna
import torch
from sklearn.metrics import precision_score, root_mean_squared_error

class BayesianOptimizer_Base:
    def __init__(self, data, nbr_trials, train_nbr_epochs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.nbr_trials = nbr_trials
        self.train_nbr_epochs = train_nbr_epochs

    def create_study(self, model_class):
        study = optuna.create_study(directions=['maximize', 'maximize', 'minimize'])
        study.optimize(lambda trial: self.objective(trial, self.data, model_class), n_trials=self.nbr_trials)
        return study

    def objective(self, trial, data, model_class: type):
        pass

    def train_and_evaluate(self, data, lr, weight_decay, model):
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_accuracy = 0
        total_precision = 0
        total_rmse = 0

        for _ in range(self.train_nbr_epochs):
            model.train()   

            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = loss_func(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            accuracy, precision, rmse = self.test_model(model, data)
            total_accuracy += accuracy
            total_precision += precision
            total_rmse += rmse

        average_accuracy = total_accuracy / self.train_nbr_epochs
        average_precision = total_precision / self.train_nbr_epochs
        average_rmse = total_rmse / self.train_nbr_epochs

        return average_accuracy, average_precision, average_rmse

    def test_model(self, model, data):
        model.eval()

        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)

        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())

        precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='weighted', zero_division=0)
        rmse = root_mean_squared_error(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

        return acc, precision, rmse
        
    def get_best_study(self, study):
        best_trials = study.best_trials

        for trial in best_trials:
            best_accuracy = trial.values[0]
            best_precision = trial.values[1]
            best_rmse = trial.values[2]
            best_hyperparameters = trial.params

            # print(f'Best Accuracy: {best_accuracy}')
            # print(f'Best Precision: {best_precision}')
            # print(f'Best RMSE: {best_rmse}')
            # print(f'Best hyperparameters: {best_hyperparameters}')

        return best_accuracy, best_precision, best_rmse, best_hyperparameters
