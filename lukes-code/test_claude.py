import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.transforms import v2

def debug_tensor_shapes(data, stage=""):
    """Debug function to print tensor shapes and help identify the issue"""
    print(f"{stage} tensor shape: {data.shape}")
    print(f"{stage} tensor size (total elements): {data.numel()}")
    
    if len(data.shape) == 4:
        print(f"4D tensor - Individual video format")
        T, C, H, W = data.shape
        print(f"Dimensions: T={T}, C={C}, H={H}, W={W}")
        print(f"Expected after permute(1,0,2,3): [C={C}, T={T}, H={H}, W={W}]")
        
    elif len(data.shape) == 5:
        print(f"5D tensor - Batched format")
        B, dim1, dim2, H, W = data.shape
        print(f"Dimensions: B={B}, dim1={dim1}, dim2={dim2}, H={H}, W={W}")
        
        # Try to determine if it's [B,C,T,H,W] or [B,T,C,H,W]
        if dim1 == 3:  # Likely channels
            print(f"Likely format: [B={B}, C={dim1}, T={dim2}, H={H}, W={W}] ✓")
        elif dim2 == 3:  # Likely channels
            print(f"Likely format: [B={B}, T={dim1}, C={dim2}, H={H}, W={W}] ❌")
        else:
            print(f"Unclear format - neither dim1 nor dim2 is 3 (channels)")
        
        # Check total size consistency
        expected_size = B * dim1 * dim2 * H * W
        actual_size = data.numel()
        print(f"Size consistency check: {expected_size} == {actual_size} -> {expected_size == actual_size}")
    
    print("-" * 50)
    return data

def debug_transforms_and_dataloader():
    """
    Add this to your train function to debug the data flow
    """
    
    print("=== DEBUGGING DATA FLOW ===")
    
    # Your original transforms (keeping the permute since it's needed)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    
    # Debug version with shape checking
    mvitv2s_final = v2.Compose([
        v2.Lambda(lambda x: debug_tensor_shapes(x, "Before normalize")),
        v2.Lambda(lambda x: x.float() / 255.0),
        v2.Normalize(mean=mean, std=std),
        v2.Lambda(lambda x: debug_tensor_shapes(x, "Before permute")),
        v2.Lambda(lambda x: x.permute(1,0,2,3)),  # [T,C,H,W] -> [C,T,H,W]
        v2.Lambda(lambda x: debug_tensor_shapes(x, "After permute"))
    ])
    
    return mvitv2s_final

def check_model_input_requirements():
    """
    Test what input shapes MViT actually expects
    """
    print("\n=== TESTING MVIT INPUT REQUIREMENTS ===")
    
    model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
    model.eval()
    
    # Test various input shapes to see what works
    test_shapes = [
        (2, 3, 32, 224, 224),   # [B, C, T, H, W] - Expected
        (2, 32, 3, 224, 224),   # [B, T, C, H, W] - Wrong
        (2, 3, 16, 224, 224),   # Different temporal size
        (2, 3, 64, 224, 224),   # Different temporal size
    ]
    
    for i, shape in enumerate(test_shapes):
        print(f"\nTest {i+1}: Shape {shape}")
        test_input = torch.randn(*shape)
        
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"✅ SUCCESS: Output shape {output.shape}")
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:100]}...")

def debug_gradient_accumulation_issue():
    """
    Debug the gradient accumulation batch size mismatch
    """
    
    print("=== GRADIENT ACCUMULATION DEBUGGING ===")
    print("Your config shows:")
    print("- batch_size: 6 (effective batch size)")  
    print("- update_per_step: 2")
    print("- Expected per-forward batch size: 6 ÷ 2 = 3")
    print("- But error shows batch size = 4")
    print()
    
    print("This suggests one of these issues:")
    print("1. DataLoader is not respecting batch_size=6")
    print("2. Last batch in dataset has different size (drop_last=False)")
    print("3. Gradient accumulation logic has an error")
    print()
    
    debugging_code = '''
# Add this debugging code to your DataLoader creation:

dataset = VideoDataset(config.admin['root'], train_instances, train_classes,
    transforms=train_transforms, num_frames=config.data['num_frames'])

print(f"Dataset length: {len(dataset)}")
print(f"Configured batch_size: {config.training['batch_size']}")
print(f"Expected batches per epoch: {len(dataset) // config.training['batch_size']}")
print(f"Remainder samples: {len(dataset) % config.training['batch_size']}")

dataloader = DataLoader(dataset, 
    batch_size=config.training['batch_size'],
    shuffle=True, 
    num_workers=2,
    pin_memory=True,
    drop_last=True  # ADD THIS to avoid irregular batch sizes
)

# Add this to your training loop before the model forward pass:
if batch_idx < 3:  # Debug first few batches
    print(f"\\nBatch {batch_idx}:")
    print(f"  Actual batch size: {data.size(0)}")
    print(f"  Expected: {config.training['batch_size']}")
    print(f"  Data shape: {data.shape}")
    print(f"  Total samples processed so far: {total_samples}")
'''
    
    print("DEBUGGING CODE TO ADD:")
    print(debugging_code)

def analyze_error_dimensions():
    """
    Analyze the specific error dimensions to understand the issue
    """
    
    print("\n=== ERROR ANALYSIS ===")
    print("Error shape: [4, 96, 8, 56, 56]")
    print("This means:")
    print("- Batch size: 4 (unexpected)")
    print("- Some dimension: 96") 
    print("- Some dimension: 8")
    print("- Height: 56 (should be 224)")
    print("- Width: 56 (should be 224)")
    print()
    
    print("The fact that H=W=56 instead of 224 suggests:")
    print("1. Input images are not being resized to 224x224")
    print("2. Or there's downsampling happening in the model")
    print("3. The error occurs inside the model, not at input")
    print()
    
    print("The dimension 96 could be:")
    print("- Number of attention heads × head dimension")
    print("- Feature channels at some layer")
    print()
    
    print("The dimension 8 could be:")
    print("- Downsampled temporal dimension (32 → 16 → 8)")
    print("- Spatial dimension after pooling")

def recommended_fixes():
    """
    Provide step-by-step fixes to try
    """
    
    print("\n=== RECOMMENDED FIXES ===")
    
    print("1. IMMEDIATE FIX - Add drop_last=True to DataLoader:")
    print("   This prevents irregular batch sizes from the last batch")
    print()
    
    print("2. VERIFY INPUT DIMENSIONS - Add this debug code:")
    debug_code = '''
# In your training loop, before model forward pass:
print(f"Batch {batch_idx}: shape={data.shape}, min={data.min():.3f}, max={data.max():.3f}")

# Verify the expected format
B, C, T, H, W = data.shape
expected = [config.training['batch_size'], 3, config.data['num_frames'], 224, 224]
actual = [B, C, T, H, W]
print(f"Expected: {expected}")
print(f"Actual:   {actual}")
print(f"Match: {expected == actual}")

if batch_idx == 0:  # Test model with single sample first
    try:
        test_sample = data[:1]  # Take just one sample
        print(f"Testing with single sample: {test_sample.shape}")
        with torch.no_grad():
            test_output = mvitv2s(test_sample)
        print(f"✅ Single sample works: {test_output.shape}")
    except Exception as e:
        print(f"❌ Single sample fails: {e}")
        return  # Exit to debug
'''
    print(debug_code)
    
    print("\n3. CHECK YOUR TRANSFORMS:")
    print("   Make sure CenterCrop(224) and RandomCrop(224) are working")
    print("   Your transforms should output [C=3, T=32, H=224, W=224]")
    
    print("\n4. VERIFY DATASET:")
    print("   Check if all videos have consistent frame counts")
    print("   Some videos might be shorter than expected")

if __name__ == "__main__":
    debug_gradient_accumulation_issue()
    analyze_error_dimensions() 
    recommended_fixes()

def setup_mvitv2s_model(num_classes, dropout=0.5):
    """Setup MViT_V2_S model with proper head replacement"""
    
    model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1)
    
    # Replace head
    original_linear = model.head[1]
    in_features = original_linear.in_features
    
    model.head = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

def test_mvit_input_format():
    """Test function to verify correct input format"""
    
    print("Testing MViT input format...")
    
    # Create model
    model = setup_mvitv2s_model(num_classes=100)
    model.eval()
    
    # Test different input formats
    batch_size = 2
    channels = 3
    temporal_frames = 32
    height = 224
    width = 224
    
    print(f"\nTesting with dimensions:")
    print(f"Batch: {batch_size}, Channels: {channels}, Frames: {temporal_frames}, H: {height}, W: {width}")
    
    # Correct format: [B, C, T, H, W]
    correct_input = torch.randn(batch_size, channels, temporal_frames, height, width)
    print(f"\nCorrect format [B,C,T,H,W]: {correct_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(correct_input)
        print(f"✅ SUCCESS: Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
    
    # Wrong format: [B, T, C, H, W] (what your permute might be creating)
    wrong_input = torch.randn(batch_size, temporal_frames, channels, height, width)
    print(f"\nWrong format [B,T,C,H,W]: {wrong_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(wrong_input)
        print(f"✅ SUCCESS: Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

# Here's what you need to change in your train function:

def get_corrected_transforms():
    """Your corrected transform code"""
    
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    
    # CORRECTED VERSION - no permute
    mvitv2s_final = v2.Compose([
        v2.Lambda(lambda x: x.float() / 255.0),
        v2.Normalize(mean=mean, std=std),
        # REMOVED: v2.Lambda(lambda x: x.permute(1,0,2,3)) 
    ])
    
    # Setup dataset transforms
    train_transforms = v2.Compose([
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(),
        mvitv2s_final
    ])
    
    test_transforms = v2.Compose([
        v2.CenterCrop(224),
        mvitv2s_final
    ])
    
    return train_transforms, test_transforms

# Replace lines 24-28 in your original code with:
# train_transforms, test_transforms = get_corrected_transforms()

if __name__ == "__main__":
    
    # Test the input format
    test_mvit_input_format()
    
    # Show fixed transforms
    print("\n" + "="*60)
    print("SOLUTION: Remove the permute operation from your transforms")
    print("="*60)
    
    train_transforms, test_transforms = fixed_train_loop_snippet()
    
    print("\nYour VideoDataset should return tensors in format [C, T, H, W]")
    print("After batching, this becomes [B, C, T, H, W] - exactly what MViT expects!")
    print("\nThe permute(1,0,2,3) was changing [C,T,H,W] to [T,C,H,W]")
    print("which after batching became [B,T,C,H,W] - causing the error!")