import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as T

import numpy as np
from tqdm import tqdm

transform = T.Compose([
     T.Resize((224, 224)),
     T.ToTensor(),
     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.OxfordIIITPet('./datasets/', split='trainval', transform=transform, download=True)
val_set = torchvision.datasets.OxfordIIITPet('./datasets/', split='test', transform=transform, download=True)

def train(model_f, optimizer_f, device, args):
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, sampler=train_sampler)
  val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, pin_memory=True, sampler=val_sampler)

  criterion = torch.nn.CrossEntropyLoss().cuda(device)

  def make_model():
    model = model_f()
    ddp_model = nn.parallel.DistributedDataParallel(model.cuda(device), device_ids=[device])
    ddp_model.train()
    return ddp_model

  def train_step(model, data, optimizer, times):
    start_time_forward = torch.cuda.Event(enable_timing=True)
    end_time_forward = torch.cuda.Event(enable_timing=True)
    start_time_back = torch.cuda.Event(enable_timing=True)
    end_time_back = torch.cuda.Event(enable_timing=True)
    start_time_step = torch.cuda.Event(enable_timing=True)
    end_time_step = torch.cuda.Event(enable_timing=True)
    
    start_time_forward.record()
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    end_time_forward.record()
    
    start_time_back.record()
    optimizer.zero_grad()
    loss.backward()
    end_time_back.record()

    start_time_step.record()
    optimizer.step()
    end_time_step.record()

    torch.cuda.synchronize()
    times[0] = start_time_forward.elapsed_time(end_time_forward)
    times[1] = start_time_back.elapsed_time(end_time_back)
    times[2] = start_time_step.elapsed_time(end_time_step)
    return loss

  def train_model(lr):
    if device == 0:
      print(f"Trying lr = {lr} ...")
    model = make_model()
    optimizer = optimizer_f(model.parameters(), lr)
    losses = []
    val_losses = []
    times = np.empty((args.epochs, 3))
    
    for i in range(args.epochs):
      epoch_losses = []
      val_loss = torch.zeros(1, device=device)
      epoch_times = np.empty((len(train_loader), 3))

      for batch_index, batch_data in enumerate(tqdm(train_loader)):
          loss = train_step(model, batch_data, optimizer, epoch_times[batch_index])
          epoch_losses.append(loss.item())

      with torch.no_grad():
        for val_data in val_loader:
            inputs, labels = val_data[0].to(device=device), val_data[1].to(device=device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels) / len(val_loader)
        
      loss = torch.mean(torch.Tensor(epoch_losses)).div_(args.world_size).to(device=device)
      torch.distributed.all_reduce(loss)
      losses.append(loss)

      val_loss.div_(args.world_size)
      torch.distributed.all_reduce(val_loss)
      val_losses.append(val_loss)

      times[i] = np.mean(epoch_times, axis=0)
    return losses, val_losses, times
    


  best_lr = args.start_lr
  best_losses, best_val_losses, best_times = train_model(args.start_lr) 

  continue_ = args.tune_lr
  while continue_:
    continue_ = False

    if best_lr <= args.start_lr:
      losses, val_losses, times = train_model(best_lr // 2)
      if losses[-1] < best_losses[-1]:
        continue_ = True
        best_losses, best_val_losses, best_times, best_lr = losses, val_losses, times, best_lr // 2

    if best_lr >= args.start_lr:
      losses, val_losses, times = train_model(best_lr * 2)
      if losses[-1] < best_losses[-1]:
        continue_ = True
        best_losses, best_val_losses, best_times, best_lr = losses, val_losses, times, best_lr * 2

  return best_losses, best_val_losses, best_times, best_lr