import logging
from typing import Callable

import torch
from ignite.engine import Engine


class PredSaver:
    def __init__(self, output_transform: Callable = lambda x: x):
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.addHandler(logging.StreamHandler())
        self._output_transform = output_transform

        self.y_pred = None
        self.y_true = None

    def __call__(self, engine: Engine) -> None:
        y_true, y_pred = self._output_transform(engine.state.output)
        if self.y_pred is None:
            self.y_pred = y_pred.cpu()
            self.y_true = y_true.cpu()
        else:
            self.y_pred = torch.cat([self.y_pred, y_pred.cpu()], dim=0)
            self.y_true = torch.cat([self.y_true, y_true.cpu()], dim=0)
