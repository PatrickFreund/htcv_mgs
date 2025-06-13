

class EarlyStopping:
    def __init__(self, mode: str,  patience: int = 10):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric: float
    
    def __call__(self, metric: float) -> bool:
        if self.best_metric is None:
            self.best_metric = metric
            return False
        
        if self._is_improvement(metric):
            self.best_metric = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < self.best_metric
        elif self.mode == "max":
            return metric > self.best_metric 
        else:
            raise ValueError(f"Invalid mode '{self.mode}' for EarlyStopping. Use 'min' or 'max'.")

    def reset(self):
        self.counter = 0
        self.best_metric = None