import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required


from powersgd.powersgd import Aggregator, AllReduce, Config, PowerSGD
from powersgd.utils import params_in_optimizer

class PowerSGDOptimizer(Optimizer):
    def __init__(self, params, async_error=False, optimizer=required, powersgd_config=None, **kwargs):
        if powersgd_config is None:
            powersgd_config = Config(
            rank=1,  # lower rank => more aggressive compression
            min_compression_rate=10,  # don't compress gradients with less compression
            num_iters_per_step=2,  # lower number => more aggressive compression
            start_compressing_after_num_steps=0,
            async_error=async_error,
        )

        defaults = dict(optimizer=optimizer, powersgd_config=powersgd_config, **kwargs)
        self.params = list(params)
        super(PowerSGDOptimizer, self).__init__(self.params, defaults)
        self.optimizer = optimizer(self.params, **kwargs)
        self.powersgd = PowerSGD(self.params, config=powersgd_config)


    def step(self, timing=None):
        grads = [p.grad for p in self.params]  # type: ignore

        self.powersgd.aggregate(grads, timing=timing)  # subtracts the approximation from grads

        # Run an optimizer step
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
    