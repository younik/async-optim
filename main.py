from typing import List
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
from models import vgg19
import torchvision.datasets
import torchvision.transforms as T

import argparse
import os
import wandb

torch.manual_seed(42);

project_name = "Accuracy tuning - error slices"
parser = argparse.ArgumentParser(description='PowerSGD experiments')
parser.add_argument('--epochs', default=200, type=int, help='number of train epochs')
parser.add_argument('--total-batch-size', default=1024, type=int, help='batch size')
parser.add_argument('--momentum', default=0, type=float, help='momentum value')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='momentum value')
parser.add_argument('--start-lr', default=0.2, type=float, help='first learning rate')
parser.add_argument('--tune-lr', default=True, type=bool, help='tune learning rate')
parser.add_argument('--tune-epochs', default=200, type=int, help='number of epochs for tuning')
parser.add_argument('--lr-drop-epochs', default=[150, 175], type=List, help='Epochs at which drop the lr')
args = parser.parse_args()


dist_url = 'tcp://127.0.0.1:58472'
args.world_size = torch.cuda.device_count()

models = {
    'VGG19': vgg19
}

optimizers = {
    #"sgd": lambda param, lr : AsyncSGD(param, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, asynchronous=False),
    #"async-sgd": lambda param, lr : AsyncSGD(param, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay, asynchronous=True),
    #"powersgd": lambda param, lr: PowerSGDOptimizer(param, async_error=False, optimizer=SGD, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay),
    "powersgd-adapt-2": lambda param, lr: PowerSGDOptimizer(param, cut=2, async_error=True, optimizer=SGD, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay),
    "powersgd-async-4": lambda param, lr: PowerSGDOptimizer(param, cut=3, async_error=True, optimizer=SGD, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay),
    "powersgd-async-8": lambda param, lr: PowerSGDOptimizer(param, cut=4, async_error=True, optimizer=SGD, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay),

}

datasets = {
    "cifar10": torchvision.datasets.CIFAR10
}

transform = T.Compose([
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def main(device):
    if device == 0:
        os.system('wandb login eb458e621dd4d01128d5e91ef26c84ddcc82a24e')
        wandb.init(project=project_name, entity="younis", sync_tensorboard=True)

    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl', init_method=dist_url, rank=device, world_size=args.world_size)

    args.batch_size = args.total_batch_size // args.world_size
    for name, model_f in models.items():
        losses = []
        val_losses = []
        best_accuracies = []
        time_list = []
        best_lrs = []


        for optimizer_name, optimizer_f in optimizers.items():
            args.log_dir = f'./log/{optimizer_name}'
            train_loader, val_loader = _make_loaders(datasets['cifar10'], args.batch_size)

            loss_path, val_loss, accuracy, times, best_lr = train(model_f, optimizer_f, train_loader, val_loader, device, args)
            losses.append(loss_path)
            val_losses.append(val_loss)
            best_accuracies.append(accuracy)
            time_list.append(times)
            best_lrs.append(best_lr)


        if device == 0:
            keys = [f"{optim_name} lr={lr}" for optim_name, lr in zip(optimizers.keys(), best_lrs)]
            xs = list(range(args.epochs))
            time_xs = [np.cumsum( np.max([t[:, 3], t[:, 0] + t[:, 1]], axis=0) + t[:, 2] ) 
                    for t in time_list
                    ]
            wandb.log({
                f"loss-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=xs, 
                                ys=losses,
                                keys=keys,
                                title=f"{name} loss (batch_size = {args.total_batch_size})",
                                xname="epochs"
                            )
                })

            wandb.log({
                f"loss-time-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=time_xs, 
                                ys=losses,
                                keys=keys,
                                title=f"{name} loss (batch_size = {args.total_batch_size})",
                                xname="time"
                            )
                })
            
            wandb.log({
                f"val-loss-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=xs, 
                                ys=val_losses,
                                keys=keys,
                                title=f"{name} val loss (batch_size = {args.total_batch_size})",
                                xname="epochs"
                            )
                })

            wandb.log({
                f"accuracy-b{args.batch_size}-{name}" : wandb.plot.line_series(
                                xs=xs, 
                                ys=best_accuracies,
                                keys=keys,
                                title=f"{name} Accuracy (batch_size = {args.total_batch_size})",
                                xname="epochs"
                            )
                })

            
            phase_names = ["forward", "backward", "step", "communication"]
            values = np.array([np.mean(t, axis=0) for t in time_list])
            df = pd.DataFrame(values, columns=phase_names, index=list(optimizers.keys()))
            fig = px.bar(df, y=phase_names, title="Times")

            wandb.log({"times-chart": fig})

            wandb.log(args.__dict__)


def _make_loaders(dataset, batch_size):
    train_set = dataset('./datasets/', train=True, transform=transform, download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True, sampler=train_sampler)
    
    val_set = dataset('./datasets/', train=False, transform=transform, download=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

if __name__ == "__main__":
    torch.multiprocessing.spawn(main, nprocs=args.world_size)
