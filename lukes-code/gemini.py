import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm

# Assuming these are defined elsewhere or imported from your project
from video_dataset import VideoDataset # Replace with your actual dataset import
from utils import enum_dir # Replace with your actual utility import

def run_2(configs, root='../data/WLASL2000', labels='./preprocessed/labels/asl300',
        label_suffix='_fixed_frange_bboxes_len.json', output='runs/exp_0', logs='logs',
        save='checkpoints', load=None, save_every=5, recover=False):
  
  train_transforms, test_transforms = configs.get_transforms()
  
  train_instances = os.path.join(labels, f'train_instances{label_suffix}')
  val_instances = os.path.join(labels, f'val_instances{label_suffix}' )
  train_classes = os.path.join(labels, f'train_classes{label_suffix}')
  val_classes = os.path.join(labels,f'val_classes{label_suffix}' )
  
  dataset = VideoDataset(root,train_instances, train_classes,
    transforms=train_transforms, num_frames=configs.num_frames)
  dataloader = DataLoader(dataset, batch_size=configs.batch_size,
    shuffle=True, num_workers=2,pin_memory=True)
  num_classes = len(set(dataset.classes))
  
  val_dataset = VideoDataset(root, val_instances, val_classes,
    transforms=test_transforms, num_frames=configs.num_frames)
  val_dataloader = torch.utils.data.DataLoader(val_dataset,
    batch_size=configs.batch_size, shuffle=True, num_workers=2,pin_memory=False)
  val_classes = len(set(val_dataset.classes))
  assert num_classes == val_classes
  
  dataloaders = {'train': dataloader, 'val': val_dataloader}
  datasets = {'train': dataset, 'val': val_dataset}
  
  model = configs.create_model() #this handles new fc, freezing, different learning rates

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  
  # Initialize steps and epoch
  steps = 0
  begin_epoch = 0
  best_val_score = 0

  param_groups = [ 
    {
      'params': model.backbone.parameters(),
      'lr': configs.backbone_init_lr, 
      'weight_decay': configs.backbone_weight_decay
    },
    {
      'params': model.classifier.parameters(), 
      'lr': configs.classifier_init_lr, 
      'weight_decay': configs.classifier_weight_decay
    }
  ]
  
  optimizer = optim.AdamW(param_groups)
  
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=configs.t_max,
                                                         eta_min=configs.eta_min)
  loss_func = nn.CrossEntropyLoss()
  
  # Determine if we are continuing and set `load` path
  if recover:
    save_dir = os.path.join(output, save)
    if os.path.exists(save_dir):
      files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pth')]) # Only consider .pth files
      if len(files) > 0:
        # Find the latest checkpoint (e.g., by epoch number in filename)
        # Assuming filenames are like 'checkpoint_001.pth', 'checkpoint_002.pth'
        # Or 'best.pth' if you want to load the best. For recovery, usually latest checkpoint.
        latest_checkpoint_file = None
        latest_epoch_num = -1
        for f in files:
            if 'checkpoint_' in f:
                try:
                    epoch_str = f.split('_')[-1].replace('.pth', '')
                    epoch_num = int(epoch_str)
                    if epoch_num > latest_epoch_num:
                        latest_epoch_num = epoch_num
                        latest_checkpoint_file = f
                except ValueError:
                    continue # Skip files that don't match the expected checkpoint format
        
        if latest_checkpoint_file:
            load = os.path.join(save_dir, latest_checkpoint_file)
            print(f"Recovering from latest checkpoint: {load}")
        else:
            print(f"No valid numbered checkpoints found in {save_dir}. Starting from scratch.")
            load = None
      else:
        print(f'Directory: {save_dir} is empty. Starting from scratch.')
        load = None
    else:
      print(f'Could not find directory: {save_dir}. Starting from scratch.')
      load = None
    
  # Load checkpoint if `load` is set
  if load:
    if os.path.exists(load):
      checkpoint = torch.load(load, map_location=device)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      begin_epoch = checkpoint['epoch'] + 1
      steps = checkpoint['steps'] # Load steps as well
      if 'best_val_score' in checkpoint:
          best_val_score = checkpoint['best_val_score']
      print(f"Resuming from epoch {begin_epoch}, steps {steps}")
      print(f"Loaded model from {load}")
    else:
      cont = input(f"Checkpoint {load} does not exist, starting from scratch? [y]")
      if cont.lower() != 'y':
        return
      # If starting from scratch, ensure begin_epoch and steps are 0
      begin_epoch = 0
      steps = 0
  
  # Admin: Setup output, save, and logs directories
  # These need to be determined AFTER potentially loading a checkpoint,
  # so that the logs_path correctly points to the existing run's log directory.
  
  # Ensure `output` path is correctly set for recovery to point to existing run
  # If `load` was used, the `output` directory should be the parent of `save_dir`
  # where the checkpoint was found.
  if load and 'output_dir' in checkpoint: # Assuming you save output_dir in checkpoint
      output = checkpoint['output_dir']
      print(f"Re-using output directory from checkpoint: {output}")
  elif load: # Fallback if output_dir not in checkpoint
      # Try to infer output_dir from loaded path
      # This is a bit brittle, assumes standard structure like runs/exp_0/checkpoints/checkpoint_XYZ.pth
      _temp_output_candidate = os.path.dirname(os.path.dirname(load))
      if os.path.basename(os.path.dirname(load)) == save: # Check if the parent is 'checkpoints'
          output = _temp_output_candidate
          print(f"Inferred output directory: {output}")
      else:
          # If it's not a standard structure, let enum_dir handle it as a new run
          if load is None: # This case means no load, it's a fresh run
              output = enum_dir(output, make=True)
          print(f"Output directory set to: {output}")
  else: # Fresh run, no load
      output = enum_dir(output, make=True)
      print(f"Output directory set to: {output}")

  if save:
    save_path = os.path.join(output, save)
    if not os.path.exists(save_path): # Create if it doesn't exist for fresh run or inferred path
      save_path = enum_dir(save_path, make=True) # use enum_dir to create unique folder if needed
    print(f"Save directory set to: {save_path}")
  
  if logs:
    logs_path = os.path.join(output, logs)
    if not os.path.exists(logs_path): # Create if it doesn't exist
      logs_path = enum_dir(logs_path, make=True) # use enum_dir to create unique folder if needed
    print(f"Logs directory set to: {logs_path}")
    
    # Initialize SummaryWriter with the global_step from the checkpoint
    # This is the key change!
    writer = SummaryWriter(logs_path, purge_step=steps) 
    print(f"TensorBoard SummaryWriter initialized with purge_step={steps}") # type: ignore
  
      
  # Train loop
  epoch = begin_epoch # Start the epoch counter from the loaded value
  while steps < configs.max_steps and epoch < 400:
    print(f'Epoch {epoch}: Step {steps}/{configs.max_steps}') # More informative print
    print('-'*10)
    
    #each epoch has training and validation stage
    for phase in ['train', 'val']:
      
      if phase == 'train':
        model.train()
      else:
        model.eval()
        
      #Reset matrics for this phase
      running_loss = 0.0
      running_corrects = 0
      total_samples = 0
      # num_batches = 0 # Not strictly needed if using total_samples for averaging
      
      #for gradient accumulation  
      accumulated_loss = 0.0
      accumulated_steps_in_batch = 0 # Renamed to avoid confusion with overall 'steps'
      optimizer.zero_grad()
    
      #Iterate over data for this phase
      for batch_idx, item in enumerate(dataloaders[phase]):
        data, target = item['frames'], item['label_num']
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        total_samples += batch_size
        # num_batches += 1 # Not strictly needed

        #Forward pass
        if phase == 'train':
          model_output = model(data)
        else:
          with torch.no_grad():
            model_output = model(data)
            
        # Calculate loss
        loss = loss_func(model_output, target)

        #Accumulate metrics
        running_loss += loss.item() * batch_size  
        _, predicted = model_output.max(1)
        running_corrects += predicted.eq(target).sum().item()
        
        if phase == 'train':
          scaled_loss = loss / configs.update_per_step # Use configs.update_per_step here
          scaled_loss.backward()
          
          accumulated_loss += loss.item()
          accumulated_steps_in_batch += 1
          
          if accumulated_steps_in_batch == configs.update_per_step:
            optimizer.step()
            optimizer.zero_grad()
            steps += 1 # Increment overall step counter only after an optimizer step
            
            # Print progress every few steps and log to TensorBoard
            if steps % 10 == 0: # Log every 10 optimizer steps
              avg_acc_loss = accumulated_loss / accumulated_steps_in_batch
              current_acc_batch = 100.0 * running_corrects / total_samples # Accuracy up to current point in epoch
              print(f'Step {steps}: Accumulated Batch Loss: {avg_acc_loss:.4f}, '
                    f'Current Epoch Accuracy: {current_acc_batch:.2f}%')
              
              if logs:
                writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps)
                writer.add_scalar('Accuracy/Train_Step', current_acc_batch, steps)
            
            # Reset accumulation for the next optimizer step
            accumulated_loss = 0.0
            accumulated_steps_in_batch = 0
    
      # If there are remaining accumulated gradients at the end of an epoch
      # (e.g., if total_batches % configs.update_per_step != 0)
      if phase == 'train' and accumulated_steps_in_batch > 0:
          optimizer.step()
          optimizer.zero_grad()
          steps += 1 # Ensure step is incremented even for partial batches at epoch end
          # Optionally log this final partial step as well if desired
          if logs:
              avg_acc_loss = accumulated_loss / accumulated_steps_in_batch
              current_acc_batch = 100.0 * running_corrects / total_samples
              writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps)
              writer.add_scalar('Accuracy/Train_Step', current_acc_batch, steps)


      # Calculate epoch metrics
      epoch_loss = running_loss / total_samples # Average loss per sample
      epoch_acc = 100.0 * running_corrects / total_samples

      print(f'{phase.upper()} - Epoch {epoch}:')
      print(f'  Loss: {epoch_loss:.4f}')
      print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
      
      # Log learning rates (optional)
      try:
        for i, param_group in enumerate(optimizer.param_groups):
          # Log LR for each parameter group
          if logs:
            writer.add_scalar(f'LearningRate/Group_{i}', param_group['lr'], epoch)
          print(f"Group {i} learning rate: {param_group['lr']:.8f}")
      except Exception as e:
        print(f'Failed to print all learning rates due to {e}')
        
      # Log epoch metrics using the current epoch number as the global_step
      if logs:
        writer.add_scalar(f'Loss/{phase.capitalize()}_Epoch', epoch_loss, epoch)
        writer.add_scalar(f'Accuracy/{phase.capitalize()}_Epoch', epoch_acc, epoch)
      
      # Validation specific logic
      if phase == 'val':
          # Save best model
          if epoch_acc > best_val_score:
              best_val_score = epoch_acc
              model_name = os.path.join(save_path, f'best.pth')
              torch.save(model.state_dict(), model_name)
              print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
          
          # Step scheduler with validation loss (or epoch_acc)
          # Make sure scheduler.step() is called with the correct argument for your scheduler type
          # If CosineAnnealingLR, it typically doesn't take an argument, or takes `epoch`.
          # Your current code `scheduler.step(epoch_loss)` suggests a ReduceLROnPlateau,
          # but you defined CosineAnnealingLR. For CosineAnnealingLR, it's usually just `scheduler.step()`.
          # If you want to use a metric, you'd need a different scheduler.
          # For CosineAnnealingLR, `T_max` is total number of epochs.
          scheduler.step() 
          
          print(f'Best validation accuracy so far: {best_val_score:.2f}%')
      
    # Save checkpoint at the end of each epoch or based on save_every
    if save and (epoch % save_every == 0 or epoch == (400 -1) or steps >= (configs.max_steps -1)): # Added conditions for last epoch/step
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_score': best_val_score,
            'output_dir': output # Save the output directory to easily recover it
        }
        checkpoint_path = os.path.join(save_path, f'checkpoint_{str(epoch).zfill(3)}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')
    
    epoch+=1 # Increment epoch for the next iteration of the while loop

  print('Finished training successfully')
  if logs:
      writer.close() # Close the SummaryWriter at the end