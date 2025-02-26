#!/usr/bin/env python3
"""
Script to reproduce the results from the Deep Grokking paper.
This script will train models with dataset sizes of 2000, 5000, and 7000 examples
for up to 100,000 steps, matching the parameters used in the original paper.

Note: This script requires significant computational resources and may take
several hours to complete. It is provided for reference and can be modified
to use fewer steps or dataset sizes if needed.
"""

from grokking import train_and_visualize

def reproduce_paper_results(max_steps=100000, step_size=5000, init_scale=8.0, weight_decay=0.01):
    """
    Reproduce the results from the Deep Grokking paper.
    
    Parameters:
    -----------
    max_steps : int
        Maximum number of training steps (default: 100000)
    step_size : int
        Interval at which to save metrics (default: 5000)
    init_scale : float
        Initialization scaling factor (default: 8.0 as per paper)
    weight_decay : float
        Weight decay parameter (default: 0.01 as per paper)
    """
    print("Reproducing results from the Deep Grokking paper...")
    print(f"Training with dataset sizes [2000, 5000, 7000] for {max_steps} steps")
    print(f"Using initialization scale: {init_scale}, weight decay: {weight_decay}")
    print("This may take several hours to complete.")
    print("Press Ctrl+C to cancel at any time.")
    
    # Train models with the same parameters as in the paper
    train_and_visualize(
        train_dataset_sizes=[2000, 5000, 7000],
        max_steps=max_steps,
        step_size=step_size,
        save_metrics=True,
        init_scale=init_scale,
        weight_decay=weight_decay
    )
    
    print("Training complete. Results saved to the metrics directory.")
    print("To analyze the results, run:")
    print("  python3 analyze_esd.py --dataset_size 2000 --metrics_dir metrics --output_dir esd_analysis_2000")
    print("  python3 analyze_esd.py --dataset_size 5000 --metrics_dir metrics --output_dir esd_analysis_5000")
    print("  python3 analyze_esd.py --dataset_size 7000 --metrics_dir metrics --output_dir esd_analysis_7000")
    print("  python3 compare_esd_metrics.py --dataset_sizes 2000 5000 7000 --metrics_dir metrics --output_dir esd_comparison")
    print("  python3 analyze_htsr.py --dataset_sizes 2000 5000 7000 --metrics_dir metrics --output_dir htsr_analysis")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reproduce results from the Deep Grokking paper')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum number of training steps (default: 100000)')
    parser.add_argument('--step_size', type=int, default=5000,
                        help='Interval at which to save metrics (default: 5000)')
    parser.add_argument('--init_scale', type=float, default=8.0,
                        help='Initialization scaling factor (default: 8.0 as per paper)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay parameter (default: 0.01 as per paper)')
    
    args = parser.parse_args()
    
    reproduce_paper_results(
        args.max_steps, 
        args.step_size,
        args.init_scale,
        args.weight_decay
    )
