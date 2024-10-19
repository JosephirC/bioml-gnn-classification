from BayesianOptimizer_Base import BayesianOptimizer_Base

class GATBayesianOptimizer(BayesianOptimizer_Base):
    def __init__(self, data, nbr_trials, train_nbr_epochs):
        super().__init__(data, nbr_trials, train_nbr_epochs)

    def objective(self, trial, data, model_class: type):
        lr = trial.suggest_float('lr', 0.001, 0.05, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        nb_neurons = trial.suggest_int('nb_neurons', 64, 512)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)

        model = model_class(data.x.shape[1], nb_neurons, data.num_classes, dropout=dropout).to(self.device)

        accuracy, precision, rmse = self.train_and_evaluate(data, lr, weight_decay, model)

        return accuracy, precision, rmse