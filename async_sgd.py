from types import SimpleNamespace
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import torch.distributed as dist

class AsyncSGD(Optimizer):

    def __init__(self, params, lr=required, asynchronous=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, asynchronous=asynchronous)
       
        super(AsyncSGD, self).__init__(params, defaults)

        numel = sum([p.numel() for p in params])
        self.average_buffer = torch.zeros(numel, device=params[0].device())
        self.handler = SimpleNamespace(wait = lambda: None)
        
        if asynchronous:
            self.next_average_buffer = torch.empty_like(self.average_buffer)
            self.next_handler = None

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            params_with_grad = []
            lr = group['lr']
            asynchronous = group["asynchronous"]

            start_idx = 0
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    numel = p.grad.numel()

                    if not asynchronous:
                        self.average_buffer[start_idx : start_idx + numel] = p.grad / dist.get_world_size()
                    else:
                        self.next_average_buffer[start_idx : start_idx + numel] = p.grad / dist.get_world_size()

                    start_idx += numel

            if not asynchronous:
                self.handler = dist.all_reduce(self.average_buffer, async_op=True)
            else:
                self.next_handler = dist.all_reduce(self.next_average_buffer, async_op=True)

            self.handler.wait()
            start_idx = 0
            for param in zip(params_with_grad):
                numel = param.numel()
                grad = self.average_buffer[start_idx : start_idx + numel]
                param.add_(grad, alpha=-lr)
                start_idx += numel

            if asynchronous:
                self.average_buffer, self.next_average_buffer = self.next_average_buffer, self.average_buffer
                self.handler, self.next_handler = self.next_handler, self.handler
