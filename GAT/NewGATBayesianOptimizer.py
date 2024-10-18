import optuna
import torch
from sklearn.metrics import precision_score, root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_study(model_class, data, nb_trials):

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data, model_class), n_trials=nb_trials)

    trial = study.best_trial
    print('Best Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def objective(trial, data, model_class: type):
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    nb_neurons = trial.suggest_int('nb_neurons', 16, 256)
    dropout = trial.suggest_float('dropout', 0.0, 0.8)

    model = model_class(data.x.shape[1], nb_neurons, data.num_classes, dropout=dropout).to(device)

    accuracy = train_and_evaluate(data, lr, weight_decay, model)

    return accuracy

def train_and_evaluate(data, lr, weight_decay, model, num_epochs=1000):
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_accuracy = 0
    total_precision = 0
    total_rmse = 0

    for _ in range(num_epochs):
        model.train()   

        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_func(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        accuracy, precision, rmse = test_model(model, data)
        total_accuracy += accuracy
        total_precision += precision
        total_rmse += rmse

    average_accuracy = total_accuracy / num_epochs
    average_precision = total_precision / num_epochs
    average_rmse = total_rmse / num_epochs

    return average_accuracy, average_precision, average_rmse

def test_model(model, data):
    model.eval()

    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), average='weighted', zero_division=0)
    rmse = root_mean_squared_error(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

    return acc, precision, rmse