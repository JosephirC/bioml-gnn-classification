import optuna
from GCN_no_pooling import GCN_no_pooling

class GCNBayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')

    def exec(self,nb_trial, data):
        for _ in range (nb_trial):
            trial = self.study.ask()
            accuracy = self.objective(trial, data)
            self.study.tell(trial, accuracy)
        
        return self.study.best_trial


    def objective(self, trial, data):
        lr = trial.suggest_float('lr', 0.001, 0.1)
        wd = trial.suggest_float('wd', 1e-5, 1e-1)

        accuracy = self.train_and_evaluate(data, lr, wd)

        return accuracy

    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.nbr_trials, n_neurons_range=self.nbr_neurons)

        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best score: {study.best_value}")

        return study.best_params, study.best_value
    
    def train_and_evaluate(self, data, lr, wd):
        gcn = GCN_no_pooling(data.x.shape[1], data.num_classes,150,lr, wd)
        gcn = gcn.fit(data, 2000)
        return gcn.test_model(data)[0]
