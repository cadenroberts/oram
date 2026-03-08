#!/usr/bin/env python3
"""
Run baseline (standard PyTorch) CIFAR-10 training experiment.

This establishes the performance baseline without ORAM overhead
for comparison with ORAM-integrated training.

Usage:
    python experiments/run_baseline.py --epochs 100 --batch-size 128
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baseline_trainer import run_baseline_training


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run baseline CIFAR-10 training'
    )
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Training batch size (default: 128)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/baseline',
        help='Output directory for results (default: results/baseline)'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu, default: auto)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("BASELINE CIFAR-10 TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output dir: {args.output_dir}")
    print("="*60)
    
    history = run_baseline_training(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device
    )
    
    print("\nBaseline training complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Final test accuracy: {history['best_acc']:.2f}%")
    print(f"Total training time: {history['total_time']:.2f}s")


if __name__ == '__main__':
    main()
