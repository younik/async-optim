from types import SimpleNamespace
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch.distributed as dist

class AsyncSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, asynchronous=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, asynchronous=asynchronous)
        params = list(params)
        super(AsyncSGD, self).__init__(params, defaults)

        numel = sum([p.numel() for p in params])
        self.average_buffer = torch.zeros(numel, device=params[0].device)
        self.handler = SimpleNamespace(wait = lambda: None)
        
        if asynchronous:
            self.next_average_buffer = torch.zeros_like(self.average_buffer)
            self.next_handler = None

    @torch.no_grad()
    def step(self, closure=None, timing=None):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            params_with_grad = []
            lr = group['lr']
            asynchronous = group["asynchronous"]
            momentum = group["momentum"]
            weight_decay = group['weight_decay']

            start_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grad = p.grad
                    numel = grad.numel()
                    grad.add_(p, alpha=weight_decay)

                    if not asynchronous:
                        buffer = self.average_buffer[start_idx : start_idx + numel]
                    else:
                        self.handler.wait()
                        buffer = self.next_average_buffer[start_idx : start_idx + numel]
                        buffer[:] = self.average_buffer[start_idx : start_idx + numel]
                    
                    buffer.mul_(momentum)
                    buffer.add_(grad.view(-1))
                    buffer.div_(dist.get_world_size())
                    start_idx += numel


            if not asynchronous:
                timing[0].record() #
                self.handler = dist.all_reduce(self.average_buffer, async_op=True)
                self.handler.wait() #
                timing[1].record() #
            else:
                timing[0].record() #
                self.next_handler = dist.all_reduce(self.next_average_buffer, async_op=True)
                timing[1].record() #

            self.handler.wait()
            start_idx = 0
            for param in params_with_grad:
                numel = param.numel()
                grad = self.average_buffer[start_idx : start_idx + numel].view_as(param)
                param.add_(grad, alpha=-lr)
                start_idx += numel

            if asynchronous:
                self.average_buffer, self.next_average_buffer = self.next_average_buffer, self.average_buffer
                self.handler, self.next_handler = self.next_handler, self.handler
