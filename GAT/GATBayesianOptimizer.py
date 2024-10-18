import optuna
import torch

class GCNBayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')

    def run(self,nb_trial, data, model_class, hidden_channels, dropout):
        self.study.optimize(lambda trial: self.objective(trial, data, model_class, hidden_channels, dropout), n_trials=nb_trial)
        return self.study.best_trial


    def objective(self, trial, data, model_class, hidden_channels, dropout):
        lr = trial.suggest_float('lr', 0.001, 0.1)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1)
        hidden_channels = trial.suggest_int('hidden_channels', 16, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.8)

        accuracy = self.train_and_evaluate(data, lr, weight_decay, hidden_channels, dropout, model_class)

        return accuracy
    
    def train_and_evaluate(self, data, lr, weight_decay, model_class, hidden_channels, dropout):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        gat = model_class(data.x.shape[1], hidden_channels, data.num_classes, dropout=dropout).to(device)
        gat.fit(data, 10000, lr, weight_decay)

        return gat.test_model(data)[0]

        

