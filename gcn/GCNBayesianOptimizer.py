import optuna
import torch

class GCNBayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')

    def exec(self,nb_trial,data, model_type:type):
        self.study.optimize(lambda trial : self.objective(trial,data, model_type), n_trials=nb_trial)        
        return self.study.best_trial


    def objective(self, trial, data, model_type):
        torch.manual_seed(0)
        lr = trial.suggest_float('lr', 0.001, 0.1)
        wd = trial.suggest_float('wd', 1e-5, 1e-1)
        nb_neuron = trial.suggest_int('nb_neuron', 50, 400)

        accuracy = self.train_and_evaluate(data, lr, wd, nb_neuron, model_type)

        return accuracy
    
    def train_and_evaluate(self, data, lr, wd, nb_neuron, model_type: type):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        data = data.to(device)
        gcn = model_type(data.x.shape[1], data.num_classes,nb_neuron)
        gcn = gcn.to(device)
        data = data
        gcn = gcn.fit(data, 10000, lr, wd)
        return gcn.test_model(data)[0]
