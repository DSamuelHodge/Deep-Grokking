import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import argparse

def load_metrics(metrics_dir='metrics', dataset_size=None):
    """
    Load metrics from pickle files
    
    Args:
        metrics_dir (str): Directory containing metrics files
        dataset_size (int, optional): If provided, only load metrics for this dataset size
        
    Returns:
        dict: Dictionary of metrics by dataset size
    """
    metrics_by_dataset = {}
    
    # Check if metrics directory exists
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory '{metrics_dir}' not found. Please run training first.")
        return metrics_by_dataset
    
    # First try to load from step-specific files
    step_files = {}
    for filename in os.listdir(metrics_dir):
        if filename.endswith('.pkl') and 'metrics_dataset' in filename:
            try:
                # Extract dataset size from filename
                parts = filename.replace('.pkl', '').split('_')
                
                # Handle both formats: metrics_dataset_SIZE_step_STEP.pkl and metrics_dataset_SIZE.pkl
                if 'step' in filename:
                    # Format: metrics_dataset_SIZE_step_STEP.pkl
                    file_dataset_size = int(parts[2])
                else:
                    # Format: metrics_dataset_SIZE.pkl
                    file_dataset_size = int(parts[2])
                
                # Skip if we're only interested in a specific dataset size
                if dataset_size is not None and file_dataset_size != dataset_size:
                    continue
                
                if file_dataset_size not in step_files:
                    step_files[file_dataset_size] = []
                
                step_files[file_dataset_size].append(os.path.join(metrics_dir, filename))
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
    
    # Process step files for each dataset size
    for file_dataset_size, filepaths in step_files.items():
        # Initialize metrics for this dataset size
        combined_metrics = {
            'train_accuracy': {},
            'test_accuracy': {},
            'esd_metrics': {}
        }
        
        # Load and combine metrics from each step file
        for filepath in filepaths:
            try:
                with open(filepath, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Combine metrics
                if 'train_accuracy' in metrics:
                    for s, acc in metrics['train_accuracy'].items():
                        combined_metrics['train_accuracy'][s] = acc
                
                if 'test_accuracy' in metrics:
                    for s, acc in metrics['test_accuracy'].items():
                        combined_metrics['test_accuracy'][s] = acc
                
                if 'esd_metrics' in metrics:
                    for s, esd in metrics['esd_metrics'].items():
                        if esd is not None:  # Only add non-None ESD metrics
                            combined_metrics['esd_metrics'][s] = esd
                
                print(f"Loaded metrics from {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Add combined metrics to the result
        metrics_by_dataset[file_dataset_size] = combined_metrics
    
    # If no step files were found or processed, try loading from final metrics files
    if not metrics_by_dataset:
        for filename in os.listdir(metrics_dir):
            if filename.endswith('.pkl') and 'final_metrics' in filename:
                try:
                    # Extract dataset size from filename
                    file_dataset_size = int(filename.split('_')[-1].split('.')[0])
                    
                    # Skip if we're only interested in a specific dataset size
                    if dataset_size is not None and file_dataset_size != dataset_size:
                        continue
                    
                    filepath = os.path.join(metrics_dir, filename)
                    with open(filepath, 'rb') as f:
                        metrics = pickle.load(f)
                    
                    metrics_by_dataset[file_dataset_size] = metrics
                    print(f"Loaded final metrics for dataset size {file_dataset_size}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    return metrics_by_dataset

def plot_esd_metrics_over_time(metrics, metric_name='alpha', layer_idx=None, output_dir='.'):
    """
    Plot a specific ESD metric over time for all dataset sizes
    
    Args:
        metrics (dict): Dictionary of metrics by dataset size
        metric_name (str): Name of the metric to plot ('alpha', 'D', 'spectral_norm', etc.)
        layer_idx (int, optional): Index of the layer to plot. If None, plot all layers.
        output_dir (str): Directory to save output files
        
    Returns:
        None
    """
    if not metrics:
        print("No metrics to plot.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample dataset to determine layer names
    sample_dataset_size = list(metrics.keys())[0]
    sample_metrics = metrics[sample_dataset_size]
    
    # Find the first step with valid ESD metrics
    steps = sorted(list(sample_metrics['train_accuracy'].keys()))
    valid_steps = [step for step in steps if step in sample_metrics['esd_metrics'] and sample_metrics['esd_metrics'][step] is not None]
    
    if not valid_steps:
        print(f"No valid ESD metrics found for metric '{metric_name}'")
        return
    
    first_valid_step = valid_steps[0]
    
    # Get layer names
    if 'longname' not in sample_metrics['esd_metrics'][first_valid_step]:
        print(f"No layer names found in ESD metrics")
        return
    
    layer_names = sample_metrics['esd_metrics'][first_valid_step]['longname']
    
    # If layer_idx is None, plot all layers
    layer_indices = [layer_idx] if layer_idx is not None else range(len(layer_names))
    
    for idx in layer_indices:
        if idx >= len(layer_names):
            print(f"Layer index {idx} is out of range (max: {len(layer_names)-1})")
            continue
        
        layer_name = layer_names[idx]
        
        fig = go.Figure()
        
        for dataset_size, dataset_metrics in metrics.items():
            steps = sorted(list(dataset_metrics['train_accuracy'].keys()))
            
            # Extract the metric values for the specified layer
            metric_values = []
            valid_steps = []
            
            for step in steps:
                if step in dataset_metrics['esd_metrics'] and dataset_metrics['esd_metrics'][step] is not None:
                    if metric_name in dataset_metrics['esd_metrics'][step] and idx < len(dataset_metrics['esd_metrics'][step][metric_name]):
                        metric_values.append(dataset_metrics['esd_metrics'][step][metric_name][idx])
                        valid_steps.append(step)
            
            if valid_steps and metric_values:
                fig.add_trace(
                    go.Scatter(
                        x=valid_steps,
                        y=metric_values,
                        mode='lines+markers',
                        name=f"|D|={dataset_size}",
                        line=dict(width=2),
                    )
                )
        
        fig.update_layout(
            title=f"{metric_name.capitalize()} Values for Layer {layer_name} Over Training Steps",
            xaxis_title="Training Step",
            yaxis_title=f"{metric_name.capitalize()} Value",
            legend_title="Dataset Size",
            font=dict(size=14),
        )
        
        # Save the figure
        output_file = os.path.join(output_dir, f"esd_{metric_name}_layer_{idx}.html")
        pio.write_html(fig, output_file)
        print(f"Figure saved as {output_file}")

def plot_esd_metric_vs_accuracy(metrics, metric_name='alpha', layer_idx=None, use_test_accuracy=True, output_dir='.'):
    """
    Plot a specific ESD metric vs accuracy
    
    Args:
        metrics (dict): Dictionary of metrics by dataset size
        metric_name (str): Name of the metric to plot ('alpha', 'D', 'spectral_norm', etc.)
        layer_idx (int, optional): Index of the layer to plot. If None, plot all layers.
        use_test_accuracy (bool): If True, use test accuracy, otherwise use train accuracy
        output_dir (str): Directory to save output files
        
    Returns:
        None
    """
    if not metrics:
        print("No metrics to plot.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample dataset to determine layer names
    sample_dataset_size = list(metrics.keys())[0]
    sample_metrics = metrics[sample_dataset_size]
    
    # Find the first step with valid ESD metrics
    steps = sorted(list(sample_metrics['train_accuracy'].keys()))
    valid_steps = [step for step in steps if step in sample_metrics['esd_metrics'] and sample_metrics['esd_metrics'][step] is not None]
    
    if not valid_steps:
        print(f"No valid ESD metrics found for metric '{metric_name}'")
        return
    
    first_valid_step = valid_steps[0]
    
    # Get layer names
    if 'longname' not in sample_metrics['esd_metrics'][first_valid_step]:
        print(f"No layer names found in ESD metrics")
        return
    
    layer_names = sample_metrics['esd_metrics'][first_valid_step]['longname']
    
    # If layer_idx is None, plot all layers
    layer_indices = [layer_idx] if layer_idx is not None else range(len(layer_names))
    
    for idx in layer_indices:
        if idx >= len(layer_names):
            print(f"Layer index {idx} is out of range (max: {len(layer_names)-1})")
            continue
        
        layer_name = layer_names[idx]
        
        fig = go.Figure()
        
        for dataset_size, dataset_metrics in metrics.items():
            steps = sorted(list(dataset_metrics['train_accuracy'].keys()))
            
            # Extract the metric values and accuracies
            metric_values = []
            accuracies = []
            
            for step in steps:
                accuracy_key = 'test_accuracy' if use_test_accuracy else 'train_accuracy'
                
                if step in dataset_metrics['esd_metrics'] and dataset_metrics['esd_metrics'][step] is not None:
                    if metric_name in dataset_metrics['esd_metrics'][step] and idx < len(dataset_metrics['esd_metrics'][step][metric_name]):
                        if step in dataset_metrics[accuracy_key]:
                            metric_values.append(dataset_metrics['esd_metrics'][step][metric_name][idx])
                            accuracies.append(dataset_metrics[accuracy_key][step])
            
            if metric_values and accuracies:
                fig.add_trace(
                    go.Scatter(
                        x=accuracies,
                        y=metric_values,
                        mode='markers',
                        name=f"|D|={dataset_size}",
                        marker=dict(size=10),
                    )
                )
        
        accuracy_type = "Test" if use_test_accuracy else "Train"
        fig.update_layout(
            title=f"{metric_name.capitalize()} vs {accuracy_type} Accuracy for Layer {layer_name}",
            xaxis_title=f"{accuracy_type} Accuracy",
            yaxis_title=f"{metric_name.capitalize()} Value",
            legend_title="Dataset Size",
            font=dict(size=14),
        )
        
        # Save the figure
        output_file = os.path.join(output_dir, f"esd_{metric_name}_vs_{accuracy_type.lower()}_accuracy_layer_{idx}.html")
        pio.write_html(fig, output_file)
        print(f"Figure saved as {output_file}")

def plot_esd_eigenvalue_distribution(metrics, step=None, layer_idx=None, output_dir='.'):
    """
    Plot the eigenvalue distribution for a specific step and layer
    
    Args:
        metrics (dict): Dictionary of metrics by dataset size
        step (int, optional): Step to plot. If None, use the last step.
        layer_idx (int, optional): Index of the layer to plot. If None, plot all layers.
        output_dir (str): Directory to save output files
        
    Returns:
        None
    """
    if not metrics:
        print("No metrics to plot.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a sample dataset to determine layer names
    sample_dataset_size = list(metrics.keys())[0]
    sample_metrics = metrics[sample_dataset_size]
    
    # Find steps with valid ESD metrics
    steps = sorted(list(sample_metrics['train_accuracy'].keys()))
    valid_steps = [step for step in steps if step in sample_metrics['esd_metrics'] and sample_metrics['esd_metrics'][step] is not None]
    
    if not valid_steps:
        print("No valid ESD metrics found")
        return
    
    # If step is None, use the last valid step
    if step is None:
        step = valid_steps[-1]
    elif step not in valid_steps:
        print(f"Step {step} not found in valid steps. Using last valid step {valid_steps[-1]} instead.")
        step = valid_steps[-1]
    
    first_valid_step = valid_steps[0]
    
    # Get layer names
    if 'longname' not in sample_metrics['esd_metrics'][first_valid_step]:
        print("No layer names found in ESD metrics")
        return
    
    layer_names = sample_metrics['esd_metrics'][first_valid_step]['longname']
    
    # If layer_idx is None, plot all layers
    layer_indices = [layer_idx] if layer_idx is not None else range(len(layer_names))
    
    for idx in layer_indices:
        if idx >= len(layer_names):
            print(f"Layer index {idx} is out of range (max: {len(layer_names)-1})")
            continue
        
        layer_name = layer_names[idx]
        
        fig = go.Figure()
        
        for dataset_size, dataset_metrics in metrics.items():
            if step in dataset_metrics['esd_metrics'] and dataset_metrics['esd_metrics'][step] is not None:
                if 'eigs' in dataset_metrics['esd_metrics'][step] and idx < len(dataset_metrics['esd_metrics'][step]['eigs']):
                    eigenvalues = dataset_metrics['esd_metrics'][step]['eigs'][idx]
                    
                    # Create histogram of log eigenvalues
                    log_eigenvalues = np.log10(eigenvalues + 1e-10)  # Add small value to avoid log(0)
                    
                    fig.add_trace(
                        go.Histogram(
                            x=log_eigenvalues,
                            nbinsx=50,
                            name=f"|D|={dataset_size}",
                            opacity=0.7,
                        )
                    )
        
        fig.update_layout(
            title=f"Eigenvalue Distribution for Layer {layer_name} at Step {step}",
            xaxis_title="Log10 Eigenvalue",
            yaxis_title="Count",
            legend_title="Dataset Size",
            font=dict(size=14),
            barmode='overlay',
        )
        
        # Save the figure
        output_file = os.path.join(output_dir, f"esd_eigenvalue_distribution_step_{step}_layer_{idx}.html")
        pio.write_html(fig, output_file)
        print(f"Figure saved as {output_file}")

def analyze_esd_metrics(metrics_dir='metrics', dataset_size=None, output_dir='esd_analysis'):
    """
    Analyze ESD metrics for all dataset sizes and layers
    
    Args:
        metrics_dir (str): Directory containing metrics files
        dataset_size (int, optional): If provided, only analyze metrics for this dataset size
        output_dir (str): Directory to save output files
        
    Returns:
        None
    """
    # Load metrics
    metrics = load_metrics(metrics_dir, dataset_size)
    
    if not metrics:
        print("No metrics found. Make sure to run training first.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of layers from the first dataset's metrics
    first_dataset = list(metrics.values())[0]
    steps = sorted(list(first_dataset['train_accuracy'].keys()))
    valid_steps = [step for step in steps if step in first_dataset['esd_metrics'] and first_dataset['esd_metrics'][step] is not None]
    
    if not valid_steps:
        print("No valid ESD metrics found")
        return
    
    first_valid_step = valid_steps[0]
    
    if 'longname' not in first_dataset['esd_metrics'][first_valid_step]:
        print("No layer names found in ESD metrics")
        return
    
    num_layers = len(first_dataset['esd_metrics'][first_valid_step]['longname'])
    
    print(f"Analyzing ESD metrics for {len(metrics)} dataset sizes and {num_layers} layers")
    
    # Plot alpha values over time for each layer
    for layer_idx in range(num_layers):
        plot_esd_metrics_over_time(metrics, 'alpha', layer_idx, output_dir)
        plot_esd_metrics_over_time(metrics, 'D', layer_idx, output_dir)
        plot_esd_metrics_over_time(metrics, 'spectral_norm', layer_idx, output_dir)
        
        # Plot alpha vs accuracy
        plot_esd_metric_vs_accuracy(metrics, 'alpha', layer_idx, True, output_dir)
        plot_esd_metric_vs_accuracy(metrics, 'D', layer_idx, True, output_dir)
        plot_esd_metric_vs_accuracy(metrics, 'spectral_norm', layer_idx, True, output_dir)
    
    # Plot eigenvalue distributions for the last step
    last_step = valid_steps[-1]
    for layer_idx in range(num_layers):
        plot_esd_eigenvalue_distribution(metrics, last_step, layer_idx, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze ESD metrics from saved pickle files')
    parser.add_argument('--metrics_dir', type=str, default='metrics', help='Directory containing metrics files')
    parser.add_argument('--dataset_size', type=int, default=None, help='If provided, only analyze metrics for this dataset size')
    parser.add_argument('--output_dir', type=str, default='esd_analysis', help='Directory to save output files')
    
    args = parser.parse_args()
    
    analyze_esd_metrics(args.metrics_dir, args.dataset_size, args.output_dir)
