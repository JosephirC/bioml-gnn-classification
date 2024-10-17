class EarlyStopping:
    def __init__(self, patience = 10) -> None:
        self.patience = patience
        self.count = 0
        self.best_score = None
        self.delta = 0
        self.best_model = None
        pass

    def early_stop(self, val_loss, model):
        if self.best_score is None:
            self.best_score = -val_loss
            self.count = 0
            self.best_model = model
        elif -val_loss < self.best_score+self.delta:
            self.count += 1
            if self.count >= self.patience:
                return True
        else:
            self.best_score = -val_loss
            self.count = 0
            self.best_model = model
        return False
    
    def get_best_model(self):
        return self.best_model
