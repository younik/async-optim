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


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            params_with_grad = []
            current_grad = []
            grad_handlers = []
            lr = group['lr']
            asynchronous = group["asynchronous"]

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    state = self.state[p]

                    p.grad.div_(dist.get_world_size())
                    if not asynchronous:
                        current_grad.append(p.grad)
                        grad_handlers.append(dist.all_reduce(p.grad, async_op=True))
                    else:
                        if 'next_grad' not in state:
                            current_grad.append(torch.zeros_like(p))
                            grad_handlers.append(SimpleNamespace(wait = lambda: None))
                        else:
                            assert 'next_grad_handler' in state
                            current_grad.append(state['next_grad'])
                            grad_handlers.append(state['next_grad_handler'])
                            
                        state['next_grad'] = torch.clone(p.grad)
                        state["next_grad_handler"] = dist.all_reduce(state['next_grad'], async_op=True)

            for param, grad, handler in zip(params_with_grad, current_grad, grad_handlers):
                handler.wait()
                param.add_(grad, alpha=-lr)
