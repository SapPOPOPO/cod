"""Exponential moving average of a model's parameters."""

import copy
import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ep, p in zip(self.module.parameters(), model.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for eb, b in zip(self.module.buffers(), model.buffers()):
            eb.copy_(b)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)