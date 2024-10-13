import optuna

class BayesianOptimizer:
    def __init__(self, model_class, data, nbr_neurons, nbr_trials, objective_fn, train_nbr_epochs):
        self.model_class = model_class
        self.data = data
        self.nbr_neurons = nbr_neurons
        self.nbr_trials = nbr_trials
        self.objective_fn = objective_fn
        self.train_nbr_epochs = train_nbr_epochs

    def objective(trial, n_neurons_range, model_class, data):
        n_neurons = trial.suggest_int('n_neurons', n_neurons_range[0], n_neurons_range[1])
        pass

    def optimize(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=self.nbr_trials, n_neurons_range=self.nbr_neurons)

        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best score: {study.best_value}")

        return study.best_params, study.best_value
    
    def train_and_evaluate(self, model, data, hyperparameters, device):
        pass

    def test(self, model, data, device):
        pass
