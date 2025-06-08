from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Union

from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int):
        pass

    @abstractmethod
    def log_params(self, params: dict):
        pass

    @abstractmethod
    def log_model(self, model, name: str):
        pass
    
    @abstractmethod
    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        pass
    
    @abstractmethod
    def close(self):
        pass

class TensorBoardLogger(Logger):
    def __init__(self, log_dir: Union[str, Path]):
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def log_params(self, params):
        for key, value in params.items():
            self.writer.add_text(f"param/{key}", str(value))

    def log_model(self, model, name):
        self.writer.add_graph(model)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]): 
        self.writer.add_hparams(
            hparams,
            metrics
        )
    
    def close(self):
        self.writer.close()