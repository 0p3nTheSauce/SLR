import torch.nn as nn
import torch
import torchvision.models.video as video_models
from torchvision.models.video.resnet import BasicBlock, Bottleneck, Conv3DSimple,\
  Conv3DNoTemporal, Conv2Plus1D, VideoResNet, WeightsEnum, _ovewrite_named_param
from typing import Union, Callable, Sequence, Optional, Any
# from  torchvision.models.video.resnet import r3d_18, R3D_18_Weights
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import v2
# from .classifiers import AttentionClassifier

class Swin3DTiny_basic(nn.Module):
    def __init__(self, num_classes=100, drop_p=0.3, 
                weights_path=None):
        super().__init__()
        self.num_classes = num_classes,
        self.drop_p=drop_p,

        swin3dt = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)

        self.patch_embed = swin3dt.patch_embed
        self.pos_drop = swin3dt.pos_drop
        self.features = swin3dt.features
        self.norm = swin3dt.norm
        self.avgpool = swin3dt.avgpool

        in_features = swin3dt.head.in_features
        self.head = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(in_features, num_classes),
        )

        if weights_path:
            checkpoint = torch.load(weights_path, map_location='cpu')
            self.load_state_dict(checkpoint)
            print(f"Loaded pretrained weights from {weights_path}")
        
    def __str__(self):
        """Return string representation of model"""
        return f"Swin3D Tiny basic implimentation \n\
                (num_classes={self.num_classes},\n\
        Model architecture:\n\
            Backbone: {self.backbone}\n\
            Classifier: {self.classifier}"

    def forward(self, x):
        """Forward pass through the model"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)  
        return x

    @classmethod
    def from_config(cls, config):
        """Create model instance from config object"""
        instance = cls(
            num_classes=config.num_classes,
            drop_p=config.drop_p,
            weights=config.weights_path
        )
        if config.frozen:  # Only freeze if layers specified
            instance.freeze_layers(config.frozen)
        return instance

    def freeze_layers(self, frozen_layers):
        """Freeze specified layers of the Swin3D-T model"""
        if not frozen_layers:
            print('Warning: no frozen layers')
            return
        
        # Mapping from intuitive names to actual layer attributes
        layer_mapping = {
            'patch_embed': 'patch_embed',
            'pos_drop': 'pos_drop', 
            'features': 'features',  # All transformer blocks
            'norm': 'norm',
            'avgpool': 'avgpool',
            'head': 'head'
        }
        
        # For more granular control over transformer blocks
        # You can also specify individual stages like 'features.0', 'features.1', etc.
        
        for layer_name in frozen_layers:
            # Handle nested layer names (e.g., 'features.0', 'features.1')
            if '.' in layer_name:
                parts = layer_name.split('.')
                actual_layer_name = layer_mapping.get(parts[0], parts[0])
                nested_path = '.'.join(parts[1:])
            else:
                actual_layer_name = layer_mapping.get(layer_name, layer_name)
                nested_path = None
            
            # Check if the main layer exists
            if hasattr(self, actual_layer_name):
                if nested_path:
                    # Handle nested layers (e.g., features.0)
                    try:
                        current_layer = self
                        # print(f"Printing layer: {str(current_layer)}")
                        for part in layer_name.split('.'):
                            current_layer = getattr(current_layer, part)
                        layer_to_freeze = current_layer
                        print(f"Frozen nested layer: {layer_name}")
                    except AttributeError:
                        print(f"Warning: Nested layer '{layer_name}' not found")
                        continue
                else:
                    # Handle top-level layers
                    layer_to_freeze = getattr(self, actual_layer_name)
                    print(f"Frozen layer: {layer_name}")
                
                # Freeze all parameters in the layer
                for param in layer_to_freeze.parameters():
                    param.requires_grad = False
                
                # Handle BatchNorm and LayerNorm layers specifically
                def freeze_norm_layers(module):
                    # Handle BatchNorm layers (if any)
                    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
                        if hasattr(module, 'track_running_stats'):
                            module.track_running_stats = False
                    
                    # Handle LayerNorm layers (common in transformers)
                    elif isinstance(module, nn.LayerNorm):
                        module.eval()  # Set to eval mode
                        for param in module.parameters():
                            param.requires_grad = False
                    
                    # Handle GroupNorm if present
                    elif isinstance(module, nn.GroupNorm):
                        module.eval()
                        for param in module.parameters():
                            param.requires_grad = False
                
                # Apply normalization layer freezing recursively
                layer_to_freeze.apply(freeze_norm_layers)
                
            else:
                available_layers = [name for name, _ in self.named_children()]
                print(f"Warning: Layer '{layer_name}' not found. Available layers: {available_layers}")



    def check_frozen_layers(self, detailed=False):
        """
        Check which layers are frozen and provide statistics
        
        Args:
            detailed (bool): If True, shows parameter-level details for each layer
        """
        print("=" * 60)
        print("FROZEN LAYERS ANALYSIS")
        print("=" * 60)
        
        total_params = 0
        frozen_params = 0
        trainable_params = 0
        
        # Track layer-wise statistics
        layer_stats = {}
        
        # Check each main component
        main_components = ['patch_embed', 'pos_drop', 'features', 'norm', 'avgpool', 'head']
        
        for component_name in main_components:
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                
                # Count parameters in this component
                comp_total = 0
                comp_frozen = 0
                comp_trainable = 0
                
                for name, param in component.named_parameters():
                    comp_total += param.numel()
                    total_params += param.numel()
                    
                    if param.requires_grad:
                        comp_trainable += param.numel()
                        trainable_params += param.numel()
                    else:
                        comp_frozen += param.numel()
                        frozen_params += param.numel()
                
                # Determine if component is fully/partially/not frozen
                if comp_frozen == comp_total and comp_total > 0:
                    status = "🔒 FULLY FROZEN"
                elif comp_frozen > 0:
                    status = "🔶 PARTIALLY FROZEN"
                else:
                    status = "✅ TRAINABLE"
                
                layer_stats[component_name] = {
                    'total': comp_total,
                    'frozen': comp_frozen,
                    'trainable': comp_trainable,
                    'status': status
                }
                
                print(f"{component_name:15} | {status:20} | "
                    f"Params: {comp_total:>8,} | "
                    f"Frozen: {comp_frozen:>8,} | "
                    f"Trainable: {comp_trainable:>8,}")
                
                # Detailed breakdown for features (transformer blocks)
                if detailed and component_name == 'features' and comp_frozen > 0:
                    print(f"  └── Transformer Blocks Breakdown:")
                    for block_name, block in component.named_children():
                        block_total = sum(p.numel() for p in block.parameters())
                        block_frozen = sum(p.numel() for p in block.parameters() if not p.requires_grad)
                        block_trainable = block_total - block_frozen
                        
                        if block_frozen == block_total and block_total > 0:
                            block_status = "🔒 FROZEN"
                        elif block_frozen > 0:
                            block_status = "🔶 PARTIAL"
                        else:
                            block_status = "✅ TRAIN"
                        
                        print(f"      {block_name:10} | {block_status:10} | "
                            f"Params: {block_total:>7,} | Frozen: {block_frozen:>7,}")
        
        print("-" * 60)
        print("OVERALL STATISTICS")
        print("-" * 60)
        frozen_percentage = (frozen_params / total_params * 100) if total_params > 0 else 0
        trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        print(f"Total Parameters:     {total_params:>12,}")
        print(f"Frozen Parameters:    {frozen_params:>12,} ({frozen_percentage:.1f}%)")
        print(f"Trainable Parameters: {trainable_params:>12,} ({trainable_percentage:.1f}%)")
        
        # Check normalization layers status
        print("-" * 60)
        print("NORMALIZATION LAYERS STATUS")
        print("-" * 60)
        
        norm_layers_frozen = []
        norm_layers_trainable = []
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
                # Check if parameters are frozen
                params_frozen = all(not p.requires_grad for p in module.parameters())
                
                # Check if module is in eval mode
                is_eval = not module.training
                
                # Check track_running_stats for BatchNorm
                track_stats = getattr(module, 'track_running_stats', 'N/A')
                
                status_parts = []
                if params_frozen:
                    status_parts.append("Params Frozen")
                if is_eval:
                    status_parts.append("Eval Mode")
                if track_stats is False:
                    status_parts.append("No Running Stats")
                
                status = " | ".join(status_parts) if status_parts else "Active"
                
                print(f"{name:30} | {type(module).__name__:12} | {status}")
                
                if params_frozen:
                    norm_layers_frozen.append(name)
                else:
                    norm_layers_trainable.append(name)
        
        print("-" * 60)
        print(f"Normalization Summary: {len(norm_layers_frozen)} frozen, {len(norm_layers_trainable)} trainable")
        print("=" * 60)
        
        return {
            'total_params': total_params,
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'frozen_percentage': frozen_percentage,
            'layer_stats': layer_stats
        }

    # Additional helper function for quick status check
    def is_layer_frozen(self, layer_name):
        """
        Quick check if a specific layer is frozen
        
        Args:
            layer_name (str): Name of the layer to check (e.g., 'features', 'head', 'features.0')
        
        Returns:
            str: 'fully_frozen', 'partially_frozen', 'trainable', or 'not_found'
        """
        try:
            # Handle nested layer names
            layer = self
            for part in layer_name.split('.'):
                layer = getattr(layer, part)
            
            total_params = sum(p.numel() for p in layer.parameters())
            frozen_params = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
            
            if total_params == 0:
                return 'no_parameters'
            elif frozen_params == total_params:
                return 'fully_frozen'
            elif frozen_params > 0:
                return 'partially_frozen'
            else:
                return 'trainable'
                
        except AttributeError:
            return 'not_found'

if __name__ == '__main__':
    # swin3d = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    model = Swin3DTiny_basic()
    input = torch.rand(1, 3, 16, 112, 112)
    # for name, module in swin3d.named_modules():
    # print(f"{name}: {module}")
    output = model(input)
    print(output.shape)
    # features = output.view(output.size(0), -1)
    # print(features)


    # Example usage:
    model.freeze_layers(['patch_embed', 'features.0', 'features.1'])  # Freeze embedding and first 2 transformer blocks
    model.freeze_layers(['patch_embed', 'pos_drop', 'features'])      # Freeze everything except norm, avgpool, head
    model.freeze_layers(['features'])                                 # Freeze all transformer blocks only

    # Example usage:
    stats = model.check_frozen_layers(detailed=True)
    print(f"Overall frozen percentage: {stats['frozen_percentage']:.1f}%")

    # # Quick check for specific layers
    print(f"Features status: {model.is_layer_frozen('features')}")
    print(f"Head status: {model.is_layer_frozen('head')}")
    print(f"First transformer block: {model.is_layer_frozen('features.0')}")
