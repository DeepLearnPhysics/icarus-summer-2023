import torch
import torch.nn as nn
from .match_modules import XShift, GenFlash, SirenFlash

class GradientModel(torch.nn.Module):
    """
    Gradient-based optimization model for flash matching
    """
    def __init__(self, flash_algo, dx0, dx_min, dx_max):
        super(GradientModel, self).__init__()
        self.xshift = XShift(dx0, dx_min, dx_max)
        self.flash_algo = flash_algo
        if flash_algo.siren_path:
            self.genflash = SirenFlash(flash_algo)
            pass
        else:
            self.genflash = GenFlash.apply

    def forward(self, input):
        x = self.xshift(input)
        if self.flash_algo.siren_path:
            flash = self.genflash(x)
        else:
            flash = self.genflash(x, self.flash_algo)
        return flash
    
class PoissonMatchLoss(nn.Module):
    """
    Poisson NLL Loss for gradient-based optimization model
    """
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")

    def forward(self, input, target, weight=1.):
        H = torch.clamp(input, min=0.01)
        O = torch.clamp(target, min=0.01)
        loss = self.poisson_nll(H, O) - torch.log(H) / 2
        return torch.mean(weight * loss)
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain iterations.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True