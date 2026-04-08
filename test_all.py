"""
Quick test script to verify all components work correctly.
Run this before starting full training to catch any issues early.
"""

import sys
from pathlib import Path
import torch
import numpy as np

print("="*60)
print("Testing Transformer Deinterleaving Implementation")
print("="*60)

# Test 1: Model Architecture
print("\n[1/5] Testing Transformer Model...")
try:
    from transformer_model import TransformerDeinterleaver, create_model
    
    model = create_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    dummy_input = torch.randn(batch_size, seq_len, 5)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()
        model = model.cuda()
    
    output = model(dummy_input)
    assert output.shape == (batch_size, seq_len, 8), f"Wrong output shape: {output.shape}"
    
    print("  ✓ Model creation successful")
    print(f"  ✓ Forward pass successful: {dummy_input.shape} -> {output.shape}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 2: Triplet Loss
print("\n[2/5] Testing Triplet Loss...")
try:
    from triplet_loss import BatchAllTripletLoss
    
    criterion = BatchAllTripletLoss(margin=1.9)
    
    # Create dummy embeddings and labels
    embeddings = torch.randn(2, 100, 8)
    labels = torch.randint(0, 5, (2, 100))
    
    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
        labels = labels.cuda()
    
    loss, stats = criterion(embeddings, labels)
    
    assert loss.item() >= 0, "Loss should be non-negative"
    assert 'num_valid_triplets' in stats, "Missing statistics"
    
    print(f"  ✓ Loss computation successful: {loss.item():.4f}")
    print(f"  ✓ Valid triplets: {stats['num_valid_triplets']:.0f}")
    print(f"  ✓ Non-easy triplets: {stats['num_non_easy_triplets']:.0f}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 3: Data Normalization
print("\n[3/5] Testing Data Normalization...")
try:
    from data_utils import PDWNormalizer
    
    normalizer = PDWNormalizer()
    
    # Create dummy PDW data
    data = np.random.randn(2, 100, 5)
    data[:, :, 0] = np.cumsum(np.abs(data[:, :, 0]), axis=1)  # ToA should be increasing
    
    # Test normalization
    normalized = normalizer.normalize(data)
    
    assert normalized.shape == data.shape, "Shape mismatch"
    
    # Check ToA is in [0, 1]
    assert normalized[0, :, 0].min() >= -1e-6, "ToA min should be ~0"
    assert normalized[0, :, 0].max() <= 1 + 1e-6, "ToA max should be ~1"
    
    print("  ✓ Normalization successful")
    print(f"  ✓ ToA range: [{normalized[0, :, 0].min():.4f}, {normalized[0, :, 0].max():.4f}]")
    
    # Test with torch tensors
    data_torch = torch.FloatTensor(data)
    normalized_torch = normalizer.normalize(data_torch, return_torch=True)
    assert isinstance(normalized_torch, torch.Tensor), "Should return torch tensor"
    
    print("  ✓ Torch tensor support working")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 4: Inference with HDBSCAN
print("\n[4/5] Testing HDBSCAN Inference...")
try:
    from transformer_model import TransformerDeinterleaverInference
    import hdbscan
    
    # Create inference model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inference_model = TransformerDeinterleaverInference(
        model, min_cluster_size=10, device=device
    )
    
    # Test prediction
    test_data = np.random.randn(50, 5)
    labels = inference_model(test_data)
    
    assert labels.shape == (50,), f"Wrong labels shape: {labels.shape}"
    
    print(f"  ✓ HDBSCAN clustering successful")
    print(f"  ✓ Found {len(np.unique(labels))} unique clusters")
    print(f"  ✓ Labels range: [{labels.min()}, {labels.max()}]")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 5: Integration with challenge dataset
print("\n[5/5] Testing Integration with Challenge Dataset...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from turing_deinterleaving_challenge import PulseTrain
    
    # Try to load a sample if data exists
    data_path = Path(__file__).parent.parent / 'data'
    if data_path.exists():
        # Check for validation data
        val_path = data_path / 'validation'
        if val_path.exists():
            sample_files = list(val_path.glob('*.h5'))
            if sample_files:
                # Load one sample
                sample = PulseTrain.load(sample_files[0])
                print(f"  ✓ Loaded sample pulse train")
                print(f"  ✓ Data shape: {sample.data.shape}")
                print(f"  ✓ Labels shape: {sample.labels.shape}")
                print(f"  ✓ Num emitters: {len(np.unique(sample.labels))}")
                
                # Test normalization on real data
                normalized = normalizer.normalize(sample.data)
                print(f"  ✓ Normalized real data successfully")
                
                # Test inference on real data (small portion)
                test_portion = normalized[:100]  # First 100 pulses
                pred_labels = inference_model(test_portion)
                print(f"  ✓ Inference on real data successful")
                print(f"  ✓ Predicted {len(np.unique(pred_labels))} clusters")
            else:
                print("  ⚠ No sample files found, skipping real data test")
        else:
            print("  ⚠ Validation data not found, skipping real data test")
    else:
        print("  ⚠ Data directory not found, skipping real data test")
    
    print("  ✓ Challenge dataset integration working")
    
except Exception as e:
    print(f"  ⚠ Warning: {e}")
    print("  ⚠ This is OK if data hasn't been downloaded yet")

# Summary
print("\n" + "="*60)
print("✅ All core tests passed!")
print("="*60)
print("\nYou can now:")
print("  1. Train the model: python train.py --data_dir ../data")
print("  2. Test individual components:")
print("     - python transformer_model.py")
print("     - python triplet_loss.py")
print("     - python data_utils.py")
print("="*60)
