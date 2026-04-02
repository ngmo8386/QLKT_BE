"""
Training Script for Transformer-based Pulse Deinterleaving
Based on: "Radar Pulse Deinterleaving with Transformer Based Deep Metric Learning" (arXiv:2503.13476)

Training configuration from paper:
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 8
- Epochs: 8
- Triplet loss margin: 1.9
- Window length: 1000 pulses
- Min emitters: 2 (to ensure valid triplets)
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import from turing_deinterleaving_challenge
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from turing_deinterleaving_challenge import (
    DeinterleavingChallengeDataset,
    evaluate_model_on_dataset,
)

from transformer_model import TransformerDeinterleaver, TransformerDeinterleaverInference
from triplet_loss import BatchAllTripletLoss
from data_utils import PDWNormalizer


class NormalizedDataset:
    """Wrapper to apply normalization to dataset."""
    
    def __init__(self, dataset, normalizer):
        self.dataset = dataset
        self.normalizer = normalizer
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, labels = self.dataset[idx]
        # Normalize data
        data_normalized = self.normalizer.normalize(data)
        return data_normalized, labels


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    data_list, labels_list = zip(*batch)
    
    # Stack into batch
    data = np.stack(data_list, axis=0)
    labels = np.stack(labels_list, axis=0)
    
    # Convert to tensors
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)
    
    return data_tensor, labels_tensor


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_valid_triplets = 0
    total_non_easy_triplets = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        # Move to device
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        embeddings = model(data)  # (batch, seq, embed_dim)
        
        # Compute loss
        loss, stats = criterion(embeddings, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (optional, not mentioned in paper)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        total_valid_triplets += stats['num_valid_triplets']
        total_non_easy_triplets += stats['num_non_easy_triplets']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'non_easy': f"{stats['num_non_easy_triplets']:.0f}",
            'frac': f"{stats['fraction_non_easy']:.3f}"
        })
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/ValidTriplets', stats['num_valid_triplets'], global_step)
            writer.add_scalar('Train/NonEasyTriplets', stats['num_non_easy_triplets'], global_step)
            writer.add_scalar('Train/FractionNonEasy', stats['fraction_non_easy'], global_step)
        
        global_step += 1
    
    avg_loss = total_loss / num_batches
    avg_valid = total_valid_triplets / num_batches
    avg_non_easy = total_non_easy_triplets / num_batches
    
    return avg_loss, avg_valid, avg_non_easy, global_step


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    min_cluster_size: int = 20,
) -> dict:
    """Validate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Validation"):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            embeddings = model(data)
            
            # Compute loss
            loss, stats = criterion(embeddings, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Evaluate clustering performance
    print("Evaluating clustering metrics...")
    inference_model = TransformerDeinterleaverInference(
        model, min_cluster_size=min_cluster_size, device=device
    )
    
    metrics = evaluate_model_on_dataset(inference_model, dataloader)
    
    return {
        'loss': avg_loss,
        **metrics
    }


def train(
    data_dir: Path,
    output_dir: Path,
    # Model hyperparameters
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 2048,
    embedding_dim: int = 8,
    dropout: float = 0.05,
    # Training hyperparameters
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    num_epochs: int = 8,
    triplet_margin: float = 1.9,
    # Data hyperparameters
    window_length: int = 1000,
    min_emitters: int = 2,
    min_cluster_size: int = 20,
    # Other
    device: str = None,
    num_workers: int = 4,
    save_every: int = 1,
    validate_every: int = 1,
):
    """Main training function."""
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'model': {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'embedding_dim': embedding_dim,
            'dropout': dropout,
        },
        'training': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'triplet_margin': triplet_margin,
        },
        'data': {
            'window_length': window_length,
            'min_emitters': min_emitters,
            'min_cluster_size': min_cluster_size,
        }
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = DeinterleavingChallengeDataset(
        subset='train',
        window_length=window_length,
        local_path=data_dir,
        min_emitters=min_emitters,
    )
    
    val_dataset = DeinterleavingChallengeDataset(
        subset='validation',
        window_length=window_length,
        local_path=data_dir,
        min_emitters=min_emitters,
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Wrap with normalization
    normalizer = PDWNormalizer()
    train_dataset = NormalizedDataset(train_dataset, normalizer)
    val_dataset = NormalizedDataset(val_dataset, normalizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == 'cuda' else False,
    )
    
    # Create model
    print("Creating model...")
    model = TransformerDeinterleaver(
        input_dim=5,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create loss and optimizer
    criterion = BatchAllTripletLoss(margin=triplet_margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    best_v_measure = 0.0
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_valid, train_non_easy, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step
        )
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Avg Valid Triplets: {train_valid:.0f}")
        print(f"Avg Non-Easy Triplets: {train_non_easy:.0f}")
        
        # Validate
        if epoch % validate_every == 0:
            print("\nValidating...")
            val_results = validate(model, val_loader, criterion, device, min_cluster_size)
            
            print(f"\nValidation Results:")
            print(f"  Loss: {val_results['loss']:.4f}")
            print(f"  V-measure: {val_results['V-measure']:.4f}")
            print(f"  Homogeneity: {val_results['Homogeneity']:.4f}")
            print(f"  Completeness: {val_results['Completeness']:.4f}")
            print(f"  AMI: {val_results['Adjusted Mutual Information']:.4f}")
            print(f"  ARI: {val_results['Adjusted Rand Index']:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Val/Loss', val_results['loss'], epoch)
            writer.add_scalar('Val/V-measure', val_results['V-measure'], epoch)
            writer.add_scalar('Val/AMI', val_results['Adjusted Mutual Information'], epoch)
            writer.add_scalar('Val/ARI', val_results['Adjusted Rand Index'], epoch)
            
            # Save best model
            if val_results['V-measure'] > best_v_measure:
                best_v_measure = val_results['V-measure']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'v_measure': best_v_measure,
                    'config': config,
                }, output_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (V-measure: {best_v_measure:.4f})")
        
        # Save checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, output_dir / f'checkpoint_epoch_{epoch}.pt')
    
    # Final save
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, output_dir / 'final_model.pt')
    
    writer.close()
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best V-measure: {best_v_measure:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Deinterleaver')
    
    # Paths
    parser.add_argument('--data_dir', type=str, 
                       default='../data',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       default='./outputs',
                       help='Output directory for models and logs')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--embedding_dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.05)
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--triplet_margin', type=float, default=1.9)
    
    # Data hyperparameters
    parser.add_argument('--window_length', type=int, default=1000)
    parser.add_argument('--min_emitters', type=int, default=2)
    parser.add_argument('--min_cluster_size', type=int, default=20)
    
    # Other
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--validate_every', type=int, default=1)
    
    args = parser.parse_args()
    
    # Convert paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f'run_{timestamp}'
    
    # Train
    train(
        data_dir=data_dir,
        output_dir=output_dir,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        triplet_margin=args.triplet_margin,
        window_length=args.window_length,
        min_emitters=args.min_emitters,
        min_cluster_size=args.min_cluster_size,
        device=args.device,
        num_workers=args.num_workers,
        save_every=args.save_every,
        validate_every=args.validate_every,
    )


if __name__ == '__main__':
    main()
