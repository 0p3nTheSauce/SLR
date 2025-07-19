def train_model_4(model, train_loader, optimizer, loss_func, epochs=10, val_loader=None, scheduler=None,
                  output='runs/exp_0', logs='logs', save='checkpoints', save_every=1, load=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    begin_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Auto-increment output directory if it exists
    if os.path.exists(output) and output[-1].isdigit() and begin_epoch == 0:
        output = output[:-1] + str(int(output[-1]) + 1)
    
    logs_path = os.path.join(output, logs)
    
    # Load checkpoint if specified
    if load:
        if os.path.exists(load):
            checkpoint = torch.load(load, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            begin_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {begin_epoch}")
            print(f"Loaded model from {load}")
            
            # Load previous loss history
            try:
                with open(os.path.join(logs_path, 'train_losses.json'), "r") as f:
                    train_losses = json.load(f)
                if val_loader:
                    with open(os.path.join(logs_path, 'val_losses.json'), "r") as f:
                        val_losses = json.load(f)
            except FileNotFoundError:
                print("Loss history files not found, starting fresh loss tracking")
        else:
            print(f"Checkpoint {load} does not exist, starting from scratch")
    
    # Create save directory
    if save:
        save_path = os.path.join(output, save)
        os.makedirs(save_path, exist_ok=True)
    
    # Create logs directory and tensorboard writer
    os.makedirs(logs_path, exist_ok=True)
    writer = SummaryWriter(logs_path)
    
    model.train()
    for epoch in tqdm.tqdm(range(begin_epoch, epochs), desc="Training"):
        # Initialize metrics for this epoch
        running_loss = 0.0
        train_samples = 0
        train_correct = 0
        
        # Training phase
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            model_output = model(data)
            loss = loss_func(model_output, target)
            loss.backward()
            optimizer.step()
            
            # Accumulate training metrics
            running_loss += loss.item() * data.size(0)
            train_samples += data.size(0)
            _, predicted = model_output.max(1)
            train_correct += predicted.eq(target).sum().item()
        
        # Calculate training metrics
        avg_train_loss = running_loss / train_samples
        train_acc = 100. * train_correct / train_samples
        train_losses.append(avg_train_loss)
        
        # Log training metrics
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        
        # Validation phase
        avg_val_loss = None
        val_acc = None
        
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
                    
                    # Accumulate validation metrics
                    val_loss += loss.item() * data.size(0)
                    val_samples += data.size(0)
                    _, predicted = model_output.max(1)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate validation metrics
            avg_val_loss = val_loss / val_samples
            val_acc = 100. * val_correct / val_samples
            val_losses.append(avg_val_loss)
            
            # Log validation metrics
            writer.add_scalar('Loss/Val', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            # Update scheduler if provided
            if scheduler:
                scheduler.step(avg_val_loss)
            
            # Save best model
            if save and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            
            model.train()  # Return to training mode
        
        # Print epoch results
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        if val_loader:
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Early stopping check
        if current_lr < 1e-6:
            print(f"Learning rate too small ({current_lr:.2e}), stopping training")
            break
        
        # Save checkpoint
        if save and (epoch % save_every == 0 or epoch == epochs - 1):
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_losses': train_losses
            }
            
            # Add validation data if available
            if val_loader:
                checkpoint_data.update({
                    'val_loss': avg_val_loss,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                })
            
            torch.save(checkpoint_data, os.path.join(save_path, f'checkpoint_{epoch}.pth'))
        
        # Save loss histories
        with open(os.path.join(logs_path, 'train_losses.json'), "w") as f:
            json.dump(train_losses, f)
        
        if val_loader:
            with open(os.path.join(logs_path, 'val_losses.json'), "w") as f:
                json.dump(val_losses, f)
    
    writer.close()
    return train_losses, val_losses