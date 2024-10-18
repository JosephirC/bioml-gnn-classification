import optuna
import torch

class GATBayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, nb_trial, data, model_class: type):
        self.study.optimize(lambda trial: self.objective(trial, data, model_class), n_trials=nb_trial)
        return self.study.best_trial

    def objective(self, trial, data, model_class: type):
        lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        nb_neurons = trial.suggest_int('nb_neurons', 16, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.8)

        accuracy = self.train_and_evaluate(data, lr, weight_decay, nb_neurons, dropout, model_class)    
        
        return accuracy
    
    def train_and_evaluate(self, data, lr, weight_decay, nb_neurons, dropout, model_class: type):
        data = data.to(self.device)
        gat = model_class(data.x.shape[1], nb_neurons, data.num_classes, dropout=dropout)
        gat = gat.to(self.device)
        gat.fit(data, 10000, lr, weight_decay)

        return gat.test_model(data)
    