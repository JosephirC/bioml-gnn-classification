from BayesianOptimizer_Base import BayesianOptimizer_Base

class MultiHeadGATBayesianOptimizer(BayesianOptimizer_Base):
    def __init__(self, data, objective_fn, nbr_trials, train_nbr_epochs):
        super().__init__(data, objective_fn, nbr_trials, train_nbr_epochs)

    def objective(self, trial, data, model_class: type):
        lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        nb_neurons = trial.suggest_int('nb_neurons', 16, 256)
        nb_heads = trial.suggest_int('nb_heads', 2, 8)
        dropout = trial.suggest_float('dropout', 0.0, 0.8)

        model = model_class(data.x.shape[1], nb_neurons, data.num_classes, heads=nb_heads, dropout=dropout).to(self.device)

        accuracy, _, _ = self.train_and_evaluate(data, lr, weight_decay, model)

        return accuracy