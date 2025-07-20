import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
import numpy as np

class RandomNormalizationAugmentation:
    """
    Randomly adjusts normalization parameters as augmentation
    """
    def __init__(self, base_mean, base_std, mean_var=0.1, std_var=0.1, prob=0.5):
        """
        Args:
            base_mean: Original normalization mean [R, G, B]
            base_std: Original normalization std [R, G, B]
            mean_var: Variance for mean adjustment
            std_var: Variance for std adjustment
            prob: Probability of applying augmentation
        """
        self.base_mean = torch.tensor(base_mean)
        self.base_std = torch.tensor(base_std)
        self.mean_var = mean_var
        self.std_var = std_var
        self.prob = prob
    
    def __call__(self, video_tensor):
        """
        Apply random normalization augmentation
        Args:
            video_tensor: (C, T, H, W) or (T, C, H, W)
        """
        if random.random() > self.prob:
            # Apply base normalization
            return self.normalize_video(video_tensor, self.base_mean, self.base_std)
        
        # Generate random normalization parameters
        device = video_tensor.device
        
        # Randomly adjust mean and std
        mean_noise = torch.randn(3) * self.mean_var
        std_noise = torch.randn(3) * self.std_var
        
        new_mean = self.base_mean + mean_noise
        new_std = torch.clamp(self.base_std + std_noise, min=0.01)  # Prevent zero std
        
        return self.normalize_video(video_tensor, new_mean.to(device), new_std.to(device))
    
    def normalize_video(self, video_tensor, mean, std):
        """Normalize video with given mean and std"""
        if video_tensor.dim() == 4:
            if video_tensor.shape[0] == 3:  # (C, T, H, W)
                for c in range(3):
                    video_tensor[c] = (video_tensor[c] - mean[c]) / std[c]
            else:  # (T, C, H, W)
                for c in range(3):
                    video_tensor[:, c] = (video_tensor[:, c] - mean[c]) / std[c]
        return video_tensor

class AdaptiveNormalization:
    """
    Adapts normalization based on video statistics
    Useful for handling different lighting conditions
    """
    def __init__(self, base_mean, base_std, adaptation_strength=0.3):
        self.base_mean = torch.tensor(base_mean)
        self.base_std = torch.tensor(base_std)
        self.adaptation_strength = adaptation_strength
    
    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: (C, T, H, W) - normalized to [0,1]
        """
        device = video_tensor.device
        
        # Calculate per-channel statistics for this video
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            video_mean = video_tensor.mean(dim=[1, 2, 3])  # Per channel
            video_std = video_tensor.std(dim=[1, 2, 3])
        else:  # (T, C, H, W)
            video_mean = video_tensor.mean(dim=[0, 2, 3])  # Per channel
            video_std = video_tensor.std(dim=[0, 2, 3])
        
        # Adaptive normalization: blend video stats with base stats
        adaptive_mean = (1 - self.adaptation_strength) * self.base_mean.to(device) + \
                       self.adaptation_strength * video_mean
        adaptive_std = (1 - self.adaptation_strength) * self.base_std.to(device) + \
                      self.adaptation_strength * torch.clamp(video_std, min=0.01)
        
        # Apply adaptive normalization
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            for c in range(3):
                video_tensor[c] = (video_tensor[c] - adaptive_mean[c]) / adaptive_std[c]
        else:  # (T, C, H, W)
            for c in range(3):
                video_tensor[:, c] = (video_tensor[:, c] - adaptive_mean[c]) / adaptive_std[c]
        
        return video_tensor

class ColorJitterNormalization:
    """
    Combines color jittering with normalization for more aggressive augmentation
    """
    def __init__(self, base_mean, base_std, brightness=0.2, contrast=0.2, 
                 saturation=0.2, hue=0.1, prob=0.5):
        self.base_mean = torch.tensor(base_mean)
        self.base_std = torch.tensor(base_std)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob
    
    def __call__(self, video_tensor):
        """
        Apply color jitter before normalization
        Args:
            video_tensor: (C, T, H, W) - values in [0, 1]
        """
        if random.random() > self.prob:
            return self.normalize_standard(video_tensor)
        
        # Apply color transformations
        if self.brightness > 0:
            brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
            video_tensor = video_tensor * brightness_factor
        
        if self.contrast > 0:
            contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
            mean = video_tensor.mean(dim=[2, 3], keepdim=True)
            video_tensor = (video_tensor - mean) * contrast_factor + mean
        
        if self.saturation > 0:
            saturation_factor = random.uniform(1-self.saturation, 1+self.saturation)
            gray = video_tensor.mean(dim=0, keepdim=True)  # Average across RGB
            video_tensor = gray + (video_tensor - gray) * saturation_factor
        
        # Clamp to valid range
        video_tensor = torch.clamp(video_tensor, 0, 1)
        
        return self.normalize_standard(video_tensor)
    
    def normalize_standard(self, video_tensor):
        """Apply standard normalization"""
        device = video_tensor.device
        mean = self.base_mean.to(device)
        std = self.base_std.to(device)
        
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            for c in range(3):
                video_tensor[c] = (video_tensor[c] - mean[c]) / std[c]
        
        return video_tensor

# Modified version of your existing transform
def create_augmented_transforms(base_mean, base_std, augment_prob=0.5):
    """
    Create transforms with normalization augmentation
    """
    # Your existing preprocessing
    base_transform = transforms.Compose([
        transforms.Lambda(lambda x: correct_num_frames(x, 16)),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Lambda(lambda x: F.interpolate(x, size=(112, 112), 
                         mode='bilinear', align_corners=False)),
    ])
    
    # Add normalization augmentation
    norm_augment = RandomNormalizationAugmentation(
        base_mean=base_mean,
        base_std=base_std,
        mean_var=0.05,  # Conservative values for fine-tuning
        std_var=0.03,
        prob=augment_prob
    )
    
    final_transform = transforms.Compose([
        base_transform,
        transforms.Lambda(norm_augment),
        transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (T,C,H,W) -> (C,T,H,W)
    ])
    
    return final_transform

# Example usage with your existing code
def get_training_transforms_with_norm_augmentation():
    """
    Modified version of your transform with normalization augmentation
    """
    base_mean = [0.43216, 0.394666, 0.37645]
    base_std = [0.22803, 0.22145, 0.216989]
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: correct_num_frames(x, 16)),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Lambda(lambda x: F.interpolate(x, size=(112, 112), 
                         mode='bilinear', align_corners=False)),
        
        # Apply normalization augmentation
        transforms.Lambda(RandomNormalizationAugmentation(
            base_mean=base_mean,
            base_std=base_std,
            mean_var=0.08,  # Slightly more aggressive for training
            std_var=0.05,
            prob=0.6
        )),
        
        transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
    ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: correct_num_frames(x, 16)),
        transforms.Lambda(lambda x: x.float() / 255.0),
        transforms.Lambda(lambda x: F.interpolate(x, size=(112, 112), 
                         mode='bilinear', align_corners=False)),
        transforms.Lambda(lambda x: normalise(x, mean=base_mean, std=base_std)),
        transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
    ])
    
    return train_transform, val_transform

# Helper functions (assuming these exist in your Dataset module)
def correct_num_frames(video_tensor, target_frames):
    """Your existing function"""
    # Implementation depends on your Dataset module
    pass

def normalise(video_tensor, mean, std):
    """Your existing normalization function"""
    # Implementation depends on your Dataset module
    pass

# Progressive normalization augmentation strategy
class ProgressiveNormAugmentation:
    """
    Gradually increase augmentation strength during training
    """
    def __init__(self, base_mean, base_std, max_var=0.15, total_epochs=100):
        self.base_mean = torch.tensor(base_mean)
        self.base_std = torch.tensor(base_std)
        self.max_var = max_var
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
    
    def __call__(self, video_tensor):
        # Progressive augmentation strength
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        current_var = self.max_var * progress
        
        if current_var < 0.01:  # No augmentation in early epochs
            return self.normalize_standard(video_tensor)
        
        # Apply progressive augmentation
        device = video_tensor.device
        mean_noise = torch.randn(3) * current_var
        std_noise = torch.randn(3) * current_var * 0.5
        
        new_mean = self.base_mean + mean_noise
        new_std = torch.clamp(self.base_std + std_noise, min=0.01)
        
        return self.normalize_video(video_tensor, new_mean.to(device), new_std.to(device))
    
    def normalize_standard(self, video_tensor):
        """Apply standard normalization"""
        device = video_tensor.device
        mean = self.base_mean.to(device)
        std = self.base_std.to(device)
        
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            for c in range(3):
                video_tensor[c] = (video_tensor[c] - mean[c]) / std[c]
        
        return video_tensor
    
    def normalize_video(self, video_tensor, mean, std):
        """Normalize with custom parameters"""
        if video_tensor.shape[0] == 3:  # (C, T, H, W)
            for c in range(3):
                video_tensor[c] = (video_tensor[c] - mean[c]) / std[c]
        return video_tensor