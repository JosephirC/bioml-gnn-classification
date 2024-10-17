import optuna
from GCN_no_pooling import GCN_no_pooling

class GCNBayesianOptimizer:
    def __init__(self):
        self.study = optuna.create_study(direction='maximize')

    def exec(self,nb_trial,data):
        self.study.optimize(lambda trial : self.objective(trial,data), n_trials=nb_trial)        
        return self.study.best_trial


    def objective(self, trial, data):
        lr = trial.suggest_float('lr', 0.001, 0.1)
        wd = trial.suggest_float('wd', 1e-5, 1e-1)
        nb_neuron = trial.suggest_int('nb_neuron', 50, 400)

        accuracy = self.train_and_evaluate(data, lr, wd, nb_neuron)

        return accuracy
    
    def train_and_evaluate(self, data, lr, wd, nb_neuron):
        gcn = GCN_no_pooling(data.x.shape[1], data.num_classes,nb_neuron,lr, wd)
        gcn = gcn.fit(data, 2000)
        return gcn.test_model(data)[0]
