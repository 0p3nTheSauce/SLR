import argparse
import torch # type: ignore
import os
import tqdm   # type: ignore
from torch.utils.tensorboard import SummaryWriter # type: ignore
import json

def train_model_3(model, train_loader, optimizer, loss_func, epochs=10,val_loader=None, 
                  output='runs/exp_0', logs='logs', save='checkpoints', save_every=1, load=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  begin_epoch = 0
  
  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      begin_epoch = checkpoint['epoch'] + 1
      print(f"Resuming from epoch {begin_epoch}")
      print(f"Loaded model from {load}")
    else:
      print(f"Checkpoint {load} does not exist, starting from scratch")
  
  if os.path.exists(output) and output[-1].isdigit() and begin_epoch == 0:
    output = output[:-1] + str(int(output[-1])+ 1) #enumerate file name
    
  if save:
    save_path = os.path.join(output, save)
    os.makedirs(save_path,exist_ok=True)
  
  logs_path = os.path.join(output, logs)
  writer = SummaryWriter(logs_path) #watching loss
  train_losses = []
  val_losses = []
  best_val_loss = float('inf')
  
  model.train()
  for epoch in tqdm.tqdm(range(begin_epoch, epochs), desc="Training R3D"):
    #Training phase
    running_loss = 0.0
    train_samples = 0
    
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      
      optimizer.zero_grad()
      model_output = model(data)
      loss = loss_func(model_output, target)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item() * data.size(0) #weight by batch size
      train_samples += data.size(0)
      
    avg_train_loss = running_loss / train_samples
    train_losses.append(avg_train_loss)
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    #Validation phase
    if val_loader:
      model.eval()
      val_loss = 0.0
      val_samples = 0
      
      with torch.no_grad():
        for data, target in val_loader:
          data, target = data.to(device), target.to(device)
          
          model_output = model(data)
          loss = loss_func(model_output, target)
          
          val_loss += loss.item() * data.size(0) #weight by batch size
          val_samples += data.size(0)
          
      avg_val_loss = val_loss / val_samples
      val_losses.append(avg_val_loss)
      writer.add_scalar('Loss/Val', avg_val_loss, epoch)
      
      if save and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),
                   os.path.join(save_path, 'best.pth')) # type: ignore
      
      print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
      model.train() # return back to train
    else:
      print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_train_loss:.4f}')
    
    if save and epoch % save_every == 0:
      avg_train_loss = avg_train_loss if avg_train_loss else 'N/A'
      avg_val_loss = avg_val_loss if avg_val_loss else 'N/A' # type: ignore
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train loss': avg_train_loss,
        'val loss': avg_val_loss,
        'train losses': train_losses,
        'val losses': val_losses
        }, os.path.join(save_path, f'checkpoint_{epoch}.pth')) # type: ignore
    
    with open(os.path.join(logs_path, 'train_losses.json'), "w") as f:
      json.dump(train_losses, f)
    if val_loader:
      with open(os.path.join(logs_path, 'val_losses.json'), "w") as f:
        json.dump(val_losses, f)
    
  return train_losses, val_losses

def enum_dir(path, make=False):
  if os.path.exists(path):
    if not path[-1].isdigit():
      path += '0'
    while os.path.exists(path):
      path = path[:-1] + str(int(path[-1]) + 1)
  if make:
    os.makedirs(path, exist_ok=True)
  return path

def train_model_4(model, train_loader, optimizer, loss_func, epochs=10,val_loader=None, schedular=None,
                  output='runs/exp_0', logs='logs', save='checkpoints', save_every=1, load=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  begin_epoch = 0
  train_metrics = []
  val_metrics = []
  best_val_loss = float('inf')

  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      begin_epoch = checkpoint['epoch'] + 1
      print(f"Resuming from epoch {begin_epoch}")
      print(f"Loaded model from {load}")
      if output and logs:
        logs_path = os.path.join(output, logs)
        try:
          with open(os.path.join(logs_path, 'train_metrics.json'), "r") as f:
            train_losses = json.load(f)
          if val_loader:
            with open(os.path.join(logs_path, 'val_metrics.json'), "r") as f:
              val_losses = json.load(f)
        except FileNotFoundError:
          print("Metrics history files not found, starting fresh metrics tracking")
    else:
      cont = input(f"Checkpoint {load} does not exist, starting from scratch? [y]")
      if cont.lower() != 'y':
        return
  
  if output:
    if begin_epoch == 0:
      output = enum_dir(output, make=True) 
    print(f"Output directory set to: {output}")
    
  if save:
    save_path = os.path.join(output, save)
    if begin_epoch == 0:
      save_path = enum_dir(save_path, make=True)
    print(f"Save directory set to: {save_path}")
  
  if logs:
    logs_path = os.path.join(output, logs)
    if begin_epoch == 0:
      logs_path = enum_dir(logs_path, make=True)
    print(f"Logs directory set to: {logs_path}")
    writer = SummaryWriter(logs_path) #watching loss
    
  model.train()
  for epoch in tqdm.tqdm(range(begin_epoch, epochs), desc="Training R3D"):
    #Training phase
    running_loss = 0.0
    train_samples = 0
    train_correct = 0 
    
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      
      optimizer.zero_grad()
      model_output = model(data)
      loss = loss_func(model_output, target)
      loss.backward()
      optimizer.step()
      
      #Accumulate metrics
      running_loss += loss.item() * data.size(0) #weight by batch size
      train_samples += data.size(0)
      _, predicted = model_output.max(1)
      train_correct += predicted.eq(target).sum().item()
    
    #Calculate average loss and accuracy 
    avg_train_loss = running_loss / train_samples
    train_acc = 100. * train_correct / train_samples
    train_metrics.append({'epoch': epoch, 'loss': avg_train_loss, 'accuracy': train_acc})
    
    if logs: 
      writer.add_scalar('Loss/Train', avg_train_loss, epoch) # type: ignore
      writer.add_scalar('Accuracy/Train', train_acc, epoch) # type: ignore
      
    #Validation phase
    if val_loader:
      model.eval()
      val_loss = 0.0
      val_samples = 0
      val_correct = 0
      
      with torch.no_grad():
        for data, target in val_loader:
          data, target = data.to(device), target.to(device)
          
          model_output = model(data)
          loss = loss_func(model_output, target)
          
          #Accumulate validation metrics
          val_loss += loss.item() * data.size(0) #weight by batch size
          val_samples += data.size(0)
          _, predicted = model_output.max(1)
          val_correct += predicted.eq(target).sum().item()
     
      avg_val_loss = val_loss / val_samples
      val_acc = 100. * val_correct / val_samples
      val_metrics.append({'epoch': epoch, 'loss': avg_val_loss, 'accuracy': val_acc})

      if logs:
        writer.add_scalar('Loss/Val', avg_val_loss, epoch) # type: ignore
        writer.add_scalar('Accuracy/Val', val_acc, epoch) # type: ignore
      
      if schedular:
        schedular.step(avg_val_loss)
      
      if save and avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),
                   os.path.join(save_path, 'best.pth')) # type: ignore
      
      model.train() # return back to train
    
    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f'  Epoch {epoch+1}/{epochs}:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    
    if val_loader:
      print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%') # type: ignore
    
    print(f'  Learning Rate: {current_lr:.6f}')
      
    # Early stopping check 
    if current_lr < 1e-6:
      print(f"Learning rate too small ({current_lr}), stopping training")
      break
    
    if save and (epoch % save_every == 0 or epoch == epochs - 1):
      checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train loss': avg_train_loss
      }
      
      if val_loader:
        checkpoint_data.update({
          'val loss': avg_val_loss, # type: ignore
          'best val loss': best_val_loss,
        })
      
      torch.save(checkpoint_data, os.path.join(save_path, f'checkpoint_{epoch}.pth')) # type: ignore
        
    if logs:
      with open(os.path.join(logs_path, 'train_metrics.json'), "w") as f: # type: ignore
        json.dump(train_metrics, f) 
      if val_loader:
        with open(os.path.join(logs_path, 'val_metrics.json'), "w") as f: # type: ignore
          json.dump(val_metrics, f)  
      
  return train_metrics, val_metrics


def main():
    # parser = argparse.ArgumentParser(description='Train a model')
    
    # parser.add_argument('model', help='model to use')
    # parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    # parser.add_argument('-n', '--lines', type=int, default=10, help='Number of lines to process')
    
    # args = parser.parse_args()
    
    # print(f"Processing {args.input_file}")
    # if args.output:
    #     print(f"Output will go to {args.output}")
    # if args.verbose:
    #     print("Verbose mode enabled")
    # print(f"Processing {args.lines} lines")
    direc = '/home/luke/ExtraStorage/WLASL/lukes-code/runs/exp_enum'
    enum = enum_dir(direc)
    print(f"Enumerated directory: \n{enum}")
    
if __name__ == '__main__':
    main()