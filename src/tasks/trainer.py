from abc import ABC, abstractmethod
from torch import nn, Tensor, optim
from typing import Tuple
from torch.utils.data import DataLoader

class OmniTask(ABC):
    def __init__(
        self,
        name: str,
        args,
        global_step: int = 0,
    ):
        self.name = name
        self.args = args
        self.global_step = global_step

    @property
    def config(self):
        return vars(self.args)

    @classmethod
    def get_loss_fn(cls, **kwargs) -> nn.modules.loss._Loss:
        """
        return the loss used during training
        :return: loss function
        """
        ...

    @classmethod
    def compute_correct(
        cls, logits: Tensor, labels: Tensor, **kwargs
    ) -> Tuple[Tensor, int]:
        """
        compute the number of correct predictions
        :param logit:
        :param labels:
        :return:
        """
        ...

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LambdaLR,
        dataloader: DataLoader,
        device,
        **kwargs
    ):
        """
        training function
        :param model: model to train
        :param optimizer: optimizer used to train
        :param scheduler: scheduler used to train
        :param dataloader: dataloader
        :param device: device used to train
        :return:
        """
        ...

    @abstractmethod
    def eval(self, model: nn.Module, dataloader: DataLoader, device, **kwargs):
        """
        Evaluation function
        :param model:
        :param dataloader:
        :param device:
        :param kwargs:
        :return:
        """
        ...

