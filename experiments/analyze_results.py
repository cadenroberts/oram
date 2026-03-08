#!/usr/bin/env python3
"""
Analyze and compare baseline vs ORAM training results.

Generates overhead breakdown reports and visualizations for the thesis.

Usage:
    python experiments/analyze_results.py \
        --baseline results/baseline \
        --oram results/oram \
        --output results/analysis
"""

import argparse
import os
import sys
import json
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir: str) -> Tuple[Dict, Dict]:
    """Load history and profile data from results directory."""
    history_path = os.path.join(results_dir, 'history.json')
    profile_path = os.path.join(results_dir, f'{os.path.basename(results_dir)}_profile.json')
    
    # Try alternate profile path naming
    if not os.path.exists(profile_path):
        for fname in os.listdir(results_dir):
            if fname.endswith('_profile.json'):
                profile_path = os.path.join(results_dir, fname)
                break
    
    history = {}
    profile = {}
    
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = json.load(f)
    
    return history, profile


def compute_overhead_comparison(
    baseline_profile: Dict,
    oram_profile: Dict
) -> pd.DataFrame:
    """Compute overhead comparison between baseline and ORAM."""
    
    baseline_timings = baseline_profile.get('summary', {}).get('timings', {})
    oram_timings = oram_profile.get('summary', {}).get('timings', {})
    
    # Get all categories from both
    all_categories = set(baseline_timings.keys()) | set(oram_timings.keys())
    
    data = []
    for category in all_categories:
        baseline_time = baseline_timings.get(category, {}).get('total_time', 0)
        oram_time = oram_timings.get(category, {}).get('total_time', 0)
        
        overhead = oram_time - baseline_time if baseline_time > 0 else oram_time
        overhead_ratio = oram_time / baseline_time if baseline_time > 0 else float('inf')
        
        data.append({
            'category': category,
            'baseline_time': baseline_time,
            'oram_time': oram_time,
            'overhead': overhead,
            'overhead_ratio': overhead_ratio
        })
    
    return pd.DataFrame(data)


def compute_oram_overhead_breakdown(oram_profile: Dict) -> pd.DataFrame:
    """
    Compute detailed breakdown of ORAM overhead sources.
    
    Categories:
    - io: Block I/O operations
    - oram_read/oram_write: ORAM access operations  
    - serialize/deserialize: Data format conversion
    - shuffle: Index shuffling
    - dataload: Total data loading overhead
    - compute: Model forward/backward
    - transfer: Data transfer to GPU
    """
    breakdown = oram_profile.get('overhead_breakdown', {})
    timings = oram_profile.get('summary', {}).get('timings', {})
    
    data = []
    for category, pct in breakdown.items():
        stats = timings.get(category, {})
        data.append({
            'category': category,
            'percentage': pct,
            'total_time': stats.get('total_time', 0),
            'call_count': stats.get('call_count', 0),
            'avg_time_ms': stats.get('avg_time', 0) * 1000
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('percentage', ascending=False)
    return df


def plot_overhead_breakdown(
    breakdown_df: pd.DataFrame,
    output_path: str,
    title: str = "ORAM Overhead Breakdown"
):
    """Create pie chart of overhead breakdown."""
    if breakdown_df.empty:
        print("Warning: No data for overhead breakdown plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = sns.color_palette("husl", len(breakdown_df))
    ax1.pie(
        breakdown_df['percentage'],
        labels=breakdown_df['category'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax1.set_title(title)
    
    # Bar chart of absolute times
    ax2.barh(breakdown_df['category'], breakdown_df['total_time'], color=colors)
    ax2.set_xlabel('Total Time (seconds)')
    ax2.set_title('Absolute Time by Category')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_comparison(
    baseline_history: Dict,
    oram_history: Dict,
    output_path: str
):
    """Plot training curves comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    ax = axes[0, 0]
    if baseline_history.get('train_loss'):
        ax.plot(baseline_history['epochs'], baseline_history['train_loss'], 
                label='Baseline', linewidth=2)
    if oram_history.get('train_loss'):
        ax.plot(oram_history['epochs'], oram_history['train_loss'],
                label='ORAM', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training accuracy
    ax = axes[0, 1]
    if baseline_history.get('train_acc'):
        ax.plot(baseline_history['epochs'], baseline_history['train_acc'],
                label='Baseline', linewidth=2)
    if oram_history.get('train_acc'):
        ax.plot(oram_history['epochs'], oram_history['train_acc'],
                label='ORAM', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test accuracy
    ax = axes[1, 0]
    if baseline_history.get('test_acc'):
        # Filter out zeros (epochs without evaluation)
        epochs = baseline_history['epochs']
        test_acc = baseline_history['test_acc']
        valid_idx = [i for i, acc in enumerate(test_acc) if acc > 0]
        ax.plot([epochs[i] for i in valid_idx], 
                [test_acc[i] for i in valid_idx],
                label='Baseline', marker='o', linewidth=2)
    if oram_history.get('test_acc'):
        epochs = oram_history['epochs']
        test_acc = oram_history['test_acc']
        valid_idx = [i for i, acc in enumerate(test_acc) if acc > 0]
        ax.plot([epochs[i] for i in valid_idx],
                [test_acc[i] for i in valid_idx],
                label='ORAM', marker='s', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Time comparison bar chart
    ax = axes[1, 1]
    times = []
    labels = []
    if baseline_history.get('total_time'):
        times.append(baseline_history['total_time'])
        labels.append('Baseline')
    if oram_history.get('total_time'):
        times.append(oram_history['total_time'])
        labels.append('ORAM')
    
    if times:
        bars = ax.bar(labels, times, color=['steelblue', 'coral'])
        ax.set_ylabel('Total Time (seconds)')
        ax.set_title('Total Training Time')
        
        # Add value labels
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{t:.1f}s', ha='center', va='bottom')
        
        # Add overhead annotation
        if len(times) == 2:
            overhead = times[1] / times[0]
            ax.annotate(f'{overhead:.1f}x overhead',
                       xy=(1, times[1]), xytext=(1.2, times[1]*0.8),
                       fontsize=12, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_operation_overhead(
    oram_profile: Dict,
    output_path: str
):
    """Plot distribution of per-operation times for ORAM."""
    timings = oram_profile.get('summary', {}).get('timings', {})
    
    # Focus on key ORAM operations
    key_ops = ['io', 'oram_read', 'oram_write', 'serialize', 'deserialize', 
               'shuffle', 'dataload', 'compute']
    
    data = []
    for op in key_ops:
        if op in timings:
            stats = timings[op]
            data.append({
                'operation': op,
                'avg_ms': stats.get('avg_time', 0) * 1000,
                'min_ms': stats.get('min_time', 0) * 1000,
                'max_ms': stats.get('max_time', 0) * 1000,
                'calls': stats.get('call_count', 0)
            })
    
    if not data:
        print("Warning: No operation timing data for plot")
        return
    
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average time per operation
    colors = sns.color_palette("viridis", len(df))
    bars = ax1.barh(df['operation'], df['avg_ms'], color=colors)
    ax1.set_xlabel('Average Time (ms)')
    ax1.set_title('Average Time per Operation')
    ax1.invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, df['avg_ms']):
        ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {val:.3f}ms', va='center')
    
    # Call counts
    ax2.barh(df['operation'], df['calls'], color=colors)
    ax2.set_xlabel('Number of Calls')
    ax2.set_title('Operation Call Counts')
    ax2.invert_yaxis()
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_report(
    baseline_history: Dict,
    baseline_profile: Dict,
    oram_history: Dict,
    oram_profile: Dict,
    output_path: str
):
    """Generate markdown report with analysis."""
    
    report = []
    report.append("# ORAM Overhead Analysis Report\n")
    report.append("## CSE239A Thesis Project - Baseline Characterization\n")
    
    # Summary statistics
    report.append("## Summary\n")
    report.append("| Metric | Baseline | ORAM | Overhead |\n")
    report.append("|--------|----------|------|----------|\n")
    
    baseline_time = baseline_history.get('total_time', 0)
    oram_time = oram_history.get('total_time', 0)
    overhead_ratio = oram_time / baseline_time if baseline_time > 0 else 0
    
    report.append(f"| Total Training Time | {baseline_time:.2f}s | {oram_time:.2f}s | "
                  f"{overhead_ratio:.1f}x |\n")
    
    baseline_acc = baseline_history.get('best_acc', 0)
    oram_acc = oram_history.get('best_acc', 0)
    report.append(f"| Best Test Accuracy | {baseline_acc:.2f}% | {oram_acc:.2f}% | - |\n")
    
    # Memory comparison
    baseline_mem = baseline_profile.get('summary', {}).get('memory', {}).get('peak_rss_mb', 0)
    oram_mem = oram_profile.get('summary', {}).get('memory', {}).get('peak_rss_mb', 0)
    mem_overhead = oram_mem / baseline_mem if baseline_mem > 0 else 0
    report.append(f"| Peak Memory (RSS) | {baseline_mem:.2f}MB | {oram_mem:.2f}MB | "
                  f"{mem_overhead:.1f}x |\n")
    
    # Overhead breakdown
    report.append("\n## ORAM Overhead Breakdown\n")
    breakdown_df = compute_oram_overhead_breakdown(oram_profile)
    if not breakdown_df.empty:
        report.append("| Category | Percentage | Total Time | Calls | Avg Time |\n")
        report.append("|----------|------------|------------|-------|----------|\n")
        for _, row in breakdown_df.iterrows():
            report.append(f"| {row['category']} | {row['percentage']:.1f}% | "
                         f"{row['total_time']:.2f}s | {row['call_count']:,} | "
                         f"{row['avg_time_ms']:.3f}ms |\n")
    
    # Theoretical analysis
    report.append("\n## Theoretical Analysis\n")
    report.append("### Path ORAM Overhead\n")
    report.append("- N = 50,000 samples (CIFAR-10 training set)\n")
    report.append("- Theoretical bandwidth overhead: O(log N) = O(16) block accesses per read\n")
    report.append("- Block size: 4KB\n")
    report.append(f"- Measured overhead ratio: {overhead_ratio:.1f}x\n")
    
    # Optimization opportunities
    report.append("\n## Optimization Opportunities\n")
    report.append("Based on the overhead breakdown, the following areas offer the "
                  "greatest optimization potential:\n\n")
    
    if not breakdown_df.empty:
        top_categories = breakdown_df.head(3)
        for i, (_, row) in enumerate(top_categories.iterrows(), 1):
            report.append(f"{i}. **{row['category']}** ({row['percentage']:.1f}%): ")
            if 'io' in row['category'] or 'oram' in row['category']:
                report.append("Batch ORAM operations to amortize per-access overhead\n")
            elif 'shuffle' in row['category']:
                report.append("Implement oblivious shuffling for batch access patterns\n")
            elif 'serialize' in row['category'] or 'deserialize' in row['category']:
                report.append("Optimize serialization format or cache deserialized data\n")
            else:
                report.append("Further investigation needed\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))
    
    print(f"Saved: {output_path}")


def plot_batch_size_sweep(sweep_path: str, output_path: str):
    """Plot batch-size sweep results: overhead ratio vs batch size."""
    if not os.path.exists(sweep_path):
        print(f"Warning: batch-size sweep data not found at {sweep_path}")
        return

    with open(sweep_path) as f:
        data = json.load(f)

    baseline_data = [r for r in data if r.get('mode') == 'baseline' and 'error' not in r]
    oram_data = [r for r in data if r.get('mode') == 'oram' and 'error' not in r]

    if not baseline_data or not oram_data:
        print("Warning: insufficient sweep data for batch-size plot")
        return

    bl_df = pd.DataFrame(baseline_data).sort_values('batch_size')
    or_df = pd.DataFrame(oram_data).sort_values('batch_size')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Total training time
    ax = axes[0]
    ax.plot(bl_df['batch_size'], bl_df['total_time'], 'o-', label='Baseline', linewidth=2)
    ax.plot(or_df['batch_size'], or_df['total_time'], 's-', label='ORAM', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Training Time vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-epoch time (total_time / epochs)
    ax = axes[1]
    bl_epoch = bl_df['total_time'] / bl_df['epochs']
    or_epoch = or_df['total_time'] / or_df['epochs']
    ax.plot(bl_df['batch_size'], bl_epoch, 'o-', label='Baseline', linewidth=2)
    ax.plot(or_df['batch_size'], or_epoch, 's-', label='ORAM', linewidth=2)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Per-Epoch Time vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overhead ratio
    ax = axes[2]
    merged = bl_df[['batch_size', 'total_time']].merge(
        or_df[['batch_size', 'total_time']],
        on='batch_size', suffixes=('_bl', '_oram')
    )
    if not merged.empty:
        ratio = merged['total_time_oram'] / merged['total_time_bl']
        ax.bar(merged['batch_size'].astype(str), ratio, color='coral')
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Overhead Ratio (ORAM / Baseline)')
        ax.set_title('ORAM Overhead by Batch Size')
        for i, (bs, r) in enumerate(zip(merged['batch_size'], ratio)):
            ax.text(i, r, f'{r:.1f}x', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_dataset_size_sweep(sweep_path: str, output_path: str):
    """Plot dataset-size sweep: time vs N with theoretical O(log N) reference."""
    if not os.path.exists(sweep_path):
        print(f"Warning: dataset-size sweep data not found at {sweep_path}")
        return

    with open(sweep_path) as f:
        data = json.load(f)

    oram_data = [r for r in data if 'error' not in r]
    if not oram_data:
        print("Warning: no valid dataset-size sweep data")
        return

    df = pd.DataFrame(oram_data).sort_values('num_samples')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Total time vs N
    ax = axes[0]
    ax.plot(df['num_samples'], df['total_time'], 'o-', linewidth=2, color='coral')
    ax.set_xlabel('Dataset Size (N)')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('ORAM Training Time vs Dataset Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Per-sample time vs N (should scale as O(log N))
    ax = axes[1]
    per_sample = df['total_time'] / (df['num_samples'] * df['epochs'])
    ax.plot(df['num_samples'], per_sample * 1000, 'o-', linewidth=2, color='coral',
            label='Measured')
    # Theoretical O(log N) reference
    ns = df['num_samples'].values.astype(float)
    log_n = np.log2(ns)
    # Scale theoretical to match measured at first point
    if len(per_sample) > 0 and per_sample.iloc[0] > 0:
        scale = (per_sample.iloc[0] * 1000) / log_n[0]
        ax.plot(ns, log_n * scale, '--', color='gray', label='O(log N) reference')
    ax.set_xlabel('Dataset Size (N)')
    ax.set_ylabel('Time per Sample (ms)')
    ax.set_title('Per-Sample ORAM Access Time')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Epochs normalized comparison
    ax = axes[2]
    epoch_time = df['total_time'] / df['epochs']
    ax.plot(df['num_samples'], epoch_time, 'o-', linewidth=2, color='coral')
    ax.set_xlabel('Dataset Size (N)')
    ax.set_ylabel('Time per Epoch (s)')
    ax.set_title('Per-Epoch Time vs Dataset Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze baseline vs ORAM training results'
    )
    parser.add_argument(
        '--baseline', type=str, default='results/baseline',
        help='Directory containing baseline results'
    )
    parser.add_argument(
        '--oram', type=str, default='results/oram',
        help='Directory containing ORAM results'
    )
    parser.add_argument(
        '--output', type=str, default='results/analysis',
        help='Output directory for analysis'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("ORAM OVERHEAD ANALYSIS")
    print("="*60)
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load results
    print(f"Loading baseline results from: {args.baseline}")
    baseline_history, baseline_profile = load_results(args.baseline)
    
    print(f"Loading ORAM results from: {args.oram}")
    oram_history, oram_profile = load_results(args.oram)
    
    # Check if we have data
    if not baseline_history and not oram_history:
        print("\nWarning: No primary results found. Checking sweep data...")
    
    # Generate primary plots (if data exists)
    print("\nGenerating visualizations...")
    
    if baseline_history or oram_history:
        plot_training_comparison(
            baseline_history, oram_history,
            os.path.join(args.output, 'training_comparison.png')
        )
    
    if oram_profile:
        breakdown_df = compute_oram_overhead_breakdown(oram_profile)
        plot_overhead_breakdown(
            breakdown_df,
            os.path.join(args.output, 'overhead_breakdown.png')
        )
        
        plot_per_operation_overhead(
            oram_profile,
            os.path.join(args.output, 'operation_times.png')
        )
    
    # Generate sweep plots (if data exists)
    bs_sweep = os.path.join(os.path.dirname(args.baseline), 'sweep_batch_size', 'sweep_summary.json')
    plot_batch_size_sweep(bs_sweep, os.path.join(args.output, 'batch_size_sweep.png'))

    ds_sweep = os.path.join(os.path.dirname(args.baseline), 'sweep_dataset_size', 'sweep_summary.json')
    plot_dataset_size_sweep(ds_sweep, os.path.join(args.output, 'dataset_size_sweep.png'))

    # Generate report
    print("\nGenerating report...")
    generate_report(
        baseline_history, baseline_profile,
        oram_history, oram_profile,
        os.path.join(args.output, 'overhead_report.md')
    )
    
    # Print summary
    if baseline_history and oram_history:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        baseline_time = baseline_history.get('total_time', 0)
        oram_time = oram_history.get('total_time', 0)
        if baseline_time > 0:
            print(f"Total overhead: {oram_time/baseline_time:.1f}x")
        print(f"Baseline accuracy: {baseline_history.get('best_acc', 0):.2f}%")
        print(f"ORAM accuracy: {oram_history.get('best_acc', 0):.2f}%")
        print("="*60)
    
    print(f"\nAnalysis complete. Results saved to: {args.output}")


if __name__ == '__main__':
    main()
