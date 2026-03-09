#!/usr/bin/env python3
"""
Run ORAM-integrated CIFAR-10 training experiment.

This measures the overhead of oblivious data access using Path ORAM
integrated with PyTorch data loading for ResNet-18 training.

Usage:
    python experiments/run_oram.py --epochs 100 --batch-size 128
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.oram_trainer import run_oram_training


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ORAM-integrated CIFAR-10 training'
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=str, default='results/oram')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--backend', type=str, default='file',
                        choices=['file', 'ram'])
    parser.add_argument('--block-size', type=int, default=4096)
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("ORAM-INTEGRATED CIFAR-10 TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples or 50000}")
    print(f"Backend: {args.backend}")
    print(f"Block size: {args.block_size}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.num_workers}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    history = run_oram_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        backend=args.backend,
        block_size=args.block_size,
        model_name=args.model,
        num_workers=args.num_workers,
    )
    
    print("\nORAM training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final test accuracy: {history['best_acc']:.2f}%")
    print(f"Total training time: {history['total_time']:.2f}s")


if __name__ == '__main__':
    main()
