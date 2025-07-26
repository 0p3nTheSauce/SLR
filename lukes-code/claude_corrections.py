# Corrected training loop with proper metrics calculation

while steps < configs.max_steps and epoch < 400:
    print(f'Step {steps}/{configs.max_steps}')
    print('-'*10)
    
    epoch += 1
    
    # Each epoch has training and validation stage
    for phase in ['train', 'test']:
        print(f'Phase: {phase}')
        
        if phase == 'train':
            r3d18.train()
        else:
            r3d18.eval()
            
        # Reset metrics for this phase
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        num_batches = 0
        
        # For gradient accumulation
        accumulated_loss = 0.0
        accumulation_steps = 0
        optimizer.zero_grad()
        
        # Iterate over data for this phase
        for batch_idx, (data, target) in enumerate(dataloaders[phase]):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            total_samples += batch_size
            num_batches += 1
            
            # Forward pass
            if phase == 'train':
                model_output = r3d18(data)
            else:
                with torch.no_grad():
                    model_output = r3d18(data)
            
            # Calculate loss
            loss = loss_func(model_output, target)
            
            # Accumulate metrics
            running_loss += loss.item() * batch_size  # Multiply by batch_size for proper averaging
            _, predicted = model_output.max(1)
            running_corrects += predicted.eq(target).sum().item()
            
            if phase == 'train':
                # Scale loss for gradient accumulation
                scaled_loss = loss / num_steps_per_update
                scaled_loss.backward()
                
                accumulated_loss += loss.item()
                accumulation_steps += 1
                
                # Update weights after accumulating gradients
                if accumulation_steps == num_steps_per_update:
                    optimizer.step()
                    optimizer.zero_grad()
                    steps += 1
                    
                    # Print progress every few steps
                    if steps % 10 == 0:
                        avg_acc_loss = accumulated_loss / accumulation_steps
                        current_acc = 100.0 * running_corrects / total_samples
                        print(f'Step {steps}: Accumulated Loss: {avg_acc_loss:.4f}, '
                              f'Current Accuracy: {current_acc:.2f}%')
                        
                        if logs:
                            writer.add_scalar('Loss/Train_Step', avg_acc_loss, steps)
                            writer.add_scalar('Accuracy/Train_Step', current_acc, steps)
                    
                    # Reset accumulation
                    accumulated_loss = 0.0
                    accumulation_steps = 0
        
        # Calculate epoch metrics
        epoch_loss = running_loss / total_samples  # Average loss per sample
        epoch_acc = 100.0 * running_corrects / total_samples
        
        print(f'{phase.upper()} - Epoch {epoch}:')
        print(f'  Loss: {epoch_loss:.4f}')
        print(f'  Accuracy: {epoch_acc:.2f}% ({running_corrects}/{total_samples})')
        
        # Log epoch metrics
        if logs:
            writer.add_scalar(f'Loss/{phase.capitalize()}_Epoch', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase.capitalize()}_Epoch', epoch_acc, epoch)
        
        # Validation specific logic
        if phase == 'test':
            # Save best model
            if epoch_acc > best_val_score:
                best_val_score = epoch_acc
                model_name = os.path.join(save_path, f'best_asl{num_classes}.pth')
                torch.save(r3d18.state_dict(), model_name)
                print(f'New best model saved: {model_name} (Acc: {epoch_acc:.2f}%)')
            
            # Step scheduler with validation loss
            scheduler.step(epoch_loss)
            
            print(f'Best validation accuracy so far: {best_val_score:.2f}%')
    
    # Save checkpoint
    if save and (epoch % save_every == 0 or not (steps < configs.max_steps and epoch < 400)):
        checkpoint_data = {
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': r3d18.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_score': best_val_score
        }
        checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')