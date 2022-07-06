import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def train(model_f, optimizer_f, train_loader, val_loader, device, args):
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
    start_time_communication = torch.cuda.Event(enable_timing=True)
    end_time_communication = torch.cuda.Event(enable_timing=True)
    
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
    optimizer.step(timing=(start_time_communication, end_time_communication))
    end_time_step.record()

    torch.cuda.synchronize()
    times[0] = start_time_forward.elapsed_time(end_time_forward)
    times[1] = start_time_back.elapsed_time(end_time_back)
    times[3] = start_time_communication.elapsed_time(end_time_communication)
    times[2] = start_time_step.elapsed_time(end_time_step) - times[3]
    return loss

  def train_model(lr, epochs, prof=None):
    if device == 0:
      print(f"Trying lr = {lr} ...")
    model = make_model()
    optimizer = optimizer_f(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs)
    losses = []
    val_losses = []
    accuracies = []
    times = np.empty((epochs, 4))
    
    for i in range(epochs):
      epoch_losses = []
      val_loss = torch.zeros(1, device=device)
      epoch_times = np.empty((len(train_loader), 4))
      accuracy = 0

      data_bar = tqdm(train_loader) if device == 0 else train_loader
      for batch_index, batch_data in enumerate(data_bar):
          loss = train_step(model, batch_data, optimizer, epoch_times[batch_index])
          epoch_losses.append(loss.item())
          if prof is not None:
            prof.step()

      scheduler.step()

      with torch.no_grad():
        for val_data in val_loader:
            inputs, labels = val_data[0].to(device=device), val_data[1].to(device=device)
            outputs = model(inputs)
            accuracy += torch.count_nonzero(torch.argmax(outputs, dim=1) == labels)
            val_loss += criterion(outputs, labels) / len(val_loader)

      accuracies.append(accuracy / len(val_loader.dataset))

      loss = torch.mean(torch.Tensor(epoch_losses)).div_(args.world_size).to(device=device)
      torch.distributed.all_reduce(loss)
      losses.append(loss)

      val_loss.div_(args.world_size)
      torch.distributed.all_reduce(val_loss)
      val_losses.append(val_loss)

      times[i] = np.mean(epoch_times, axis=0)
    return losses, val_losses, accuracies, times
    


  best_lr = args.start_lr
  best_losses, best_val_losses, best_accuracies, best_times = train_model(args.start_lr, args.tune_epochs) 

  continue_ = args.tune_lr
  while continue_:
    continue_ = False

    if best_lr <= args.start_lr:
      losses, val_losses, accuracies, times = train_model(best_lr // 2, args.tune_epochs)
      if accuracies[-1] > best_accuracies[-1]:
        continue_ = True
        best_losses, best_val_losses, best_accuracies, best_times, best_lr = losses, val_losses, accuracies, times, best_lr // 2

    if best_lr >= args.start_lr:
      losses, val_losses, accuracies, times = train_model(best_lr * 2, args.tune_epochs)
      if accuracies[-1] > best_accuracies[-1]:
        continue_ = True
        best_losses, best_val_losses, best_accuracies, best_times, best_lr = losses, val_losses, accuracies, times, best_lr * 2


  # if device == 0:
  #   with torch.profiler.profile(
  #         schedule=torch.profiler.schedule(wait=50, warmup=10, active=100, repeat=5),
  #         on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
  #         record_shapes=True,
  #         profile_memory=True,
  #         with_stack=True,
  #         with_modules=True,
  #   ) as prof:
  #     best_losses, best_val_losses, best_accuracies, best_times = train_model(best_lr, args.epochs, prof) 
  # else:
  best_losses, best_val_losses, best_accuracies, best_times = train_model(best_lr, args.epochs)

  return best_losses, best_val_losses, best_accuracies, best_times, best_lr