import torch
import torch.nn as nn
import torchvision
from async_sgd import AsyncSGD
from torch.optim import SGD
from powersgd import PowerSGDOptimizer
import numpy as np
import plotly.express as px
import pandas as pd
from train import train

import argparse
import os
import wandb

torch.manual_seed(42);

project_name = "AsyncSGD"
parser = argparse.ArgumentParser(description='PowerSGD experiments')
parser.add_argument('--epochs', default=20, type=int, help='number of epochs per train')
parser.add_argument('--total-batch-size', default=128, type=int, help='batch size')
parser.add_argument('--start-lr', default=0.1, type=float, help='first learning rate')
parser.add_argument('--tune-lr', default=True, type=bool, help='tune learning rate')

dist_url = 'tcp://127.0.0.1:58472'
world_size = torch.cuda.device_count()

models = {
    'ResNet18': lambda : nn.Sequential(
        torchvision.models.resnet18(pretrained = False),
        nn.ReLU(),
        nn.Linear(1000, 37),
    )
}

optimizers = {
    "sgd": lambda param, lr : AsyncSGD(param, lr=lr, asynchronous=False),
    "async-sgd": lambda param, lr : AsyncSGD(param, lr=lr, asynchronous=True),
    "powersgd-async": lambda param, lr: PowerSGDOptimizer(param, optimizer=SGD, lr=lr)
}


def main(device):
    args = parser.parse_args()
    args.world_size = world_size
    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl', init_method=dist_url, rank=device, world_size=args.world_size)

    args.batch_size = args.total_batch_size // args.world_size

    for name, model_f in models.items():
        losses = []
        val_losses = []
        time_list = []
        best_lrs = []


        for optimizer_f in optimizers.values():
            loss_path, val_loss, times, best_lr = train(model_f, optimizer_f, device, args)
            losses.append(loss_path)
            val_losses.append(val_loss)
            time_list.append(times)
            best_lrs.append(best_lr)


        if device == 0:
            os.system('wandb login eb458e621dd4d01128d5e91ef26c84ddcc82a24e')
            wandb.init(project=project_name, entity="younis")

            keys = [f"{optim_name} lr={lr}" for optim_name, lr in zip(optimizers.keys(), best_lrs)]
            xs = list(range(args.epochs))
            time_xs = [np.sum(t, axis=1).cumsum() for t in time_list]
            wandb.log({
                f"loss-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=xs, 
                                ys=losses,
                                keys=keys,
                                title=f"{name} loss (batch_size = {args.batch_size})",
                                xname="epochs"
                            )
                })

            wandb.log({
                f"loss-time-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=time_xs, 
                                ys=losses,
                                keys=keys,
                                title=f"{name} loss (batch_size = {args.batch_size})",
                                xname="time"
                            )
                })
            
            wandb.log({
                  f"val-loss-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                  xs=xs, 
                                  ys=val_losses,
                                  keys=keys,
                                  title=f"{name} val loss (batch_size = {args.batch_size})",
                                  xname="epochs"
                              )
                })

            
            phase_names = ["forward", "backward", "step"]
            values = np.array([np.mean(t, axis=0) for t in time_list])
            df = pd.DataFrame(values, columns=phase_names, index=list(optimizers.keys()))
            fig = px.bar(df, y=phase_names, title="Times")

            wandb.log({"times-chart": fig})



if __name__ == "__main__":
    torch.multiprocessing.spawn(main, nprocs=world_size)
