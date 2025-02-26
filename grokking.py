import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from scipy.linalg import svd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from collections import defaultdict
import pickle
import os
from esd import net_esd_estimator

# Deep MLP model with hook functionality to capture layer outputs
class DeepMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, init_scale=8.0):
        super(DeepMLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_size, output_size))
        
        # Activation function
        self.activation = nn.ReLU()
        
        # For storing activations
        self.activations = {}
        self.hooks = []
        
        # Register hooks for each layer
        for i, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._get_activation_hook(i)))
        
        # Apply initialization scaling
        self._apply_init_scaling(init_scale)
    
    def _apply_init_scaling(self, scale_factor):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                # Store original norm
                original_norm = layer.weight.data.norm()
                # Scale the weights
                layer.weight.data *= scale_factor
                # Print scaling info for verification
                print(f"Layer scaled: original norm={original_norm:.4f}, new norm={layer.weight.data.norm():.4f}")
    
    def _get_activation_hook(self, layer_idx):
        def hook(module, input, output):
            self.activations[f"layer_{layer_idx}"] = output.detach()
        return hook
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

# Function to compute rank (dimensionality) of feature representations
def compute_layer_ranks(model, dataloader, device):
    model.eval()
    layer_activations = defaultdict(list)
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(data.size(0), -1).to(device)
            _ = model(data)
            
            # Collect activations from all layers
            for layer_name, activations in model.activations.items():
                layer_activations[layer_name].append(activations.cpu())
    
    # Compute ranks for each layer
    layer_ranks = {}
    for layer_name, activations in layer_activations.items():
        combined_activations = torch.cat(activations, dim=0).numpy()
        _, singular_values, _ = svd(combined_activations, full_matrices=False)
        # Count values above threshold
        rank = np.sum(singular_values > 1e-5)
        layer_ranks[layer_name] = rank
    
    return layer_ranks

# Function to evaluate model accuracy
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(data.size(0), -1).to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target.to(device)).sum().item()
    
    return correct / total

# Create probe classifiers to measure representation quality at each layer
class ProbeClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(ProbeClassifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

# Train and evaluate probe classifiers
def train_evaluate_probes(model, train_loader, test_loader, hidden_size, output_size, device, num_epochs=5):
    probe_accuracies = {}
    
    for layer_name in model.activations.keys():
        # Get activations for this layer
        train_activations = []
        train_labels = []
        
        # Collect training data
        model.eval()
        with torch.no_grad():
            for data, target in train_loader:
                data = data.view(data.size(0), -1).to(device)
                _ = model(data)
                train_activations.append(model.activations[layer_name].cpu())
                train_labels.append(target)
        
        train_activations = torch.cat(train_activations, dim=0)
        train_labels = torch.cat(train_labels, dim=0)
        
        # Create and train the probe
        input_dim = train_activations.size(1)
        probe = ProbeClassifier(input_dim, output_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(probe.parameters(), lr=1e-3)
        
        for _ in range(num_epochs):
            probe.train()
            for i in range(0, len(train_activations), 64):  # Batch size of 64
                batch_x = train_activations[i:i+64].to(device)
                batch_y = train_labels[i:i+64].to(device)
                
                optimizer.zero_grad()
                outputs = probe(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate the probe on test data
        test_activations = []
        test_labels = []
        
        probe.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                _ = model(data)
                test_activations.append(model.activations[layer_name].cpu())
                test_labels.append(target)
        
        test_activations = torch.cat(test_activations, dim=0)
        test_labels = torch.cat(test_labels, dim=0)
        
        correct = 0
        total = 0
        for i in range(0, len(test_activations), 64):
            batch_x = test_activations[i:i+64].to(device)
            batch_y = test_labels[i:i+64].to(device)
            
            outputs = probe(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        probe_accuracies[layer_name] = correct / total
    
    return probe_accuracies

# Function to create plots similar to the ones in the image
def create_plots(metrics, dataset_size):
    # Create 5x1 subplot figure (adding 2 new plots for ESD metrics)
    fig = make_subplots(rows=5, cols=1, 
                        subplot_titles=(f"Layer Ranks |D^train|={dataset_size}", 
                                        "Accuracy", 
                                        "Layer Probe Accuracy",
                                        "ESD Alpha Values",
                                        "ESD D Values"),
                        vertical_spacing=0.1,
                        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    steps = list(metrics['train_accuracy'].keys())
    
    # Colors for different layers
    colors = [
        '#DAA520',  # layer_0 - yellow-gold
        '#9ACD32',  # layer_1 - yellowgreen
        '#228B22',  # layer_2 - forest green
        '#20B2AA',  # layer_3 - light sea green
        '#4682B4',  # layer_4 - steel blue
        '#483D8B',  # layer_5 - dark slate blue
        '#8B008B',  # layer_6 - dark magenta
        '#CD5C5C',  # layer_7 - indian red
        '#FF69B4',  # layer_8 - hot pink
        '#FF8C00',  # layer_9 - dark orange
        '#8B4513',  # layer_10 - saddle brown
    ]
    
    # 1. Layer Ranks Plot
    if 'layer_ranks' in metrics and steps and all(step in metrics['layer_ranks'] for step in steps):
        # Find the first step with valid layer_ranks
        first_step = steps[0]
        if metrics['layer_ranks'] and first_step in metrics['layer_ranks']:
            for i, layer_name in enumerate(sorted(metrics['layer_ranks'][first_step].keys())):
                layer_ranks = []
                for step in steps:
                    if step in metrics['layer_ranks'] and layer_name in metrics['layer_ranks'][step]:
                        layer_ranks.append(metrics['layer_ranks'][step][layer_name])
                    else:
                        layer_ranks.append(None)
                
                # Filter out None values
                valid_steps = [s for s, r in zip(steps, layer_ranks) if r is not None]
                valid_ranks = [r for r in layer_ranks if r is not None]
                
                if valid_ranks:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_steps, 
                            y=valid_ranks,
                            mode='lines+markers',
                            name=layer_name,
                            line=dict(color=colors[i % len(colors)]),
                        ),
                        row=1, col=1
                    )
    
    # Add vertical lines to mark phase transitions (at steps 40k and 60k)
    for row in range(1, 6):  # Updated to include 5 rows
        fig.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=40000, y0=0, x1=40000, y1=1, yref="paper", 
            row=row, col=1
        )
        if dataset_size >= 7000:  # Add second line for largest dataset
            fig.add_shape(
                type="line", line=dict(dash="dash", color="gray"),
                x0=60000, y0=0, x1=60000, y1=1, yref="paper", 
                row=row, col=1
            )
    
    # 2. Accuracy Plot
    train_accuracies = [metrics['train_accuracy'].get(step, None) for step in steps]
    test_accuracies = [metrics['test_accuracy'].get(step, None) for step in steps]
    
    # Filter out None values
    valid_train_steps = [s for s, a in zip(steps, train_accuracies) if a is not None]
    valid_train_accs = [a for a in train_accuracies if a is not None]
    
    valid_test_steps = [s for s, a in zip(steps, test_accuracies) if a is not None]
    valid_test_accs = [a for a in test_accuracies if a is not None]
    
    if valid_train_accs:
        fig.add_trace(
            go.Scatter(
                x=valid_train_steps, 
                y=valid_train_accs,
                mode='lines',
                name='Train Accuracy',
                line=dict(color='blue'),
            ),
            row=2, col=1
        )
    
    if valid_test_accs:
        fig.add_trace(
            go.Scatter(
                x=valid_test_steps, 
                y=valid_test_accs,
                mode='lines',
                name='Test Accuracy',
                line=dict(color='orange'),
            ),
            row=2, col=1
        )
    
    # 3. Layer Probe Accuracy Plot
    if 'probe_accuracies' in metrics and steps and any(step in metrics['probe_accuracies'] for step in steps):
        # Find the first step with valid probe_accuracies
        valid_probe_steps = [step for step in steps if step in metrics['probe_accuracies']]
        if valid_probe_steps:
            first_step = valid_probe_steps[0]
            for i, layer_name in enumerate(sorted(metrics['probe_accuracies'][first_step].keys())):
                probe_accs = []
                for step in steps:
                    if step in metrics['probe_accuracies'] and layer_name in metrics['probe_accuracies'][step]:
                        probe_accs.append(metrics['probe_accuracies'][step][layer_name])
                    else:
                        probe_accs.append(None)
                
                # Filter out None values
                valid_steps = [s for s, a in zip(steps, probe_accs) if a is not None]
                valid_accs = [a for a in probe_accs if a is not None]
                
                if valid_accs:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_steps, 
                            y=valid_accs,
                            mode='lines',
                            name=f"{layer_name} probe",
                            line=dict(color=colors[i % len(colors)]),
                            showlegend=False,  # Use same legend as first plot
                        ),
                        row=3, col=1
                    )
    
    # 4. ESD Alpha Values Plot
    if 'esd_metrics' in metrics and any(metrics['esd_metrics'].values()):
        # Find the first step with valid ESD metrics
        valid_steps = [step for step in steps if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None]
        if valid_steps:
            first_valid_step = valid_steps[0]
            
            # Get layer names
            layer_names = metrics['esd_metrics'][first_valid_step]['longname']
            
            for i, layer_name in enumerate(layer_names):
                alpha_values = []
                for step in steps:
                    if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None:
                        if 'alpha' in metrics['esd_metrics'][step] and i < len(metrics['esd_metrics'][step]['alpha']):
                            alpha_values.append(metrics['esd_metrics'][step]['alpha'][i])
                        else:
                            alpha_values.append(None)
                    else:
                        alpha_values.append(None)
                
                # Filter out None values for plotting
                valid_steps_for_layer = [s for s, a in zip(steps, alpha_values) if a is not None]
                valid_alphas = [a for a in alpha_values if a is not None]
                
                if valid_alphas:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_steps_for_layer, 
                            y=valid_alphas,
                            mode='lines+markers',
                            name=f"{layer_name} alpha",
                            line=dict(color=colors[i % len(colors)]),
                        ),
                        row=4, col=1
                    )
    
    # 5. ESD D Values Plot
    if 'esd_metrics' in metrics and any(metrics['esd_metrics'].values()):
        valid_steps = [step for step in steps if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None]
        if valid_steps:
            first_valid_step = valid_steps[0]
            
            # Get layer names
            layer_names = metrics['esd_metrics'][first_valid_step]['longname']
            
            for i, layer_name in enumerate(layer_names):
                d_values = []
                for step in steps:
                    if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None:
                        if 'D' in metrics['esd_metrics'][step] and i < len(metrics['esd_metrics'][step]['D']):
                            d_values.append(metrics['esd_metrics'][step]['D'][i])
                        else:
                            d_values.append(None)
                    else:
                        d_values.append(None)
                
                # Filter out None values for plotting
                valid_steps_for_layer = [s for s, a in zip(steps, d_values) if a is not None]
                valid_d_values = [d for d in d_values if d is not None]
                
                if valid_d_values:
                    fig.add_trace(
                        go.Scatter(
                            x=valid_steps_for_layer, 
                            y=valid_d_values,
                            mode='lines+markers',
                            name=f"{layer_name} D",
                            line=dict(color=colors[i % len(colors)]),
                            showlegend=False,  # Use same legend as first plot
                        ),
                        row=5, col=1
                    )
    
    # Update layout
    fig.update_layout(
        height=1200,  # Increased height for additional plots
        width=800,
        title_text=f"Training Dynamics with |D^train|={dataset_size}",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Rank", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_yaxes(title_text="Probe Acc", row=3, col=1)
    fig.update_yaxes(title_text="Alpha", row=4, col=1)
    fig.update_yaxes(title_text="D Value", row=5, col=1)
    
    # Update x-axis labels
    for row in range(1, 6):
        fig.update_xaxes(title_text="Step", row=row, col=1)
    
    # Set y-axis ranges to match the image
    fig.update_yaxes(range=[0, 400], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_yaxes(range=[0, 1], row=3, col=1)
    # Let the ESD plots auto-scale
    
    return fig

# Main training and visualization function
def train_and_visualize(train_dataset_sizes=[2000, 5000, 7000], max_steps=100000, step_size=5000, save_metrics=True, 
                        init_scale=8.0, weight_decay=0.01):
    """
    Train models with different dataset sizes and visualize the results
    
    Args:
        train_dataset_sizes (list): List of dataset sizes to train on
        max_steps (int): Maximum number of training steps
        step_size (int): Interval for saving metrics
        save_metrics (bool): Whether to save metrics
        init_scale (float): Initialization scaling factor (default: 8.0 as per paper)
        weight_decay (float): Weight decay parameter (default: 0.01 as per paper)
    
    Returns:
        list: List of plotly figures
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    input_size = 28 * 28
    hidden_size = 400
    num_layers = 12  # Total layers including input and output
    output_size = 10
    
    figures = []
    
    for dataset_size in train_dataset_sizes:
        print(f"Training with dataset size: {dataset_size}")
        
        # Create subsets
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_subset = torch.utils.data.Subset(full_train_dataset, range(dataset_size))
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Initialize model, loss function, and optimizer
        model = DeepMLP(input_size, hidden_size, num_layers, output_size, init_scale=init_scale).to(device)
        # Change to MSE loss as per paper
        criterion = nn.MSELoss()
        # Add weight decay as per paper
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
        
        # Metrics tracking
        metrics = {
            'layer_ranks': {},
            'train_accuracy': {},
            'test_accuracy': {},
            'probe_accuracies': {},
            'esd_metrics': {}  # New ESD metrics dictionary
        }
        
        # Create directory for saving metrics if it doesn't exist
        if save_metrics:
            os.makedirs('metrics', exist_ok=True)
        
        # Training loop
        step = 0
        while step < max_steps:
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, torch.zeros_like(output))  # Change to MSE loss
                loss.backward()
                optimizer.step()
                
                step += 1
                
                # Record metrics at specified steps
                if step % step_size == 0 or step == max_steps:
                    print(f"Step {step}/{max_steps}")
                    
                    # Compute layer ranks
                    metrics['layer_ranks'][step] = compute_layer_ranks(model, train_loader, device)
                    
                    # Compute accuracies
                    metrics['train_accuracy'][step] = evaluate_model(model, train_loader, device)
                    metrics['test_accuracy'][step] = evaluate_model(model, test_loader, device)
                    
                    # Train and evaluate probes
                    metrics['probe_accuracies'][step] = train_evaluate_probes(
                        model, train_loader, test_loader, hidden_size, output_size, device
                    )
                    
                    # Compute ESD metrics
                    try:
                        print("Computing ESD metrics...")
                        esd_results = net_esd_estimator(
                            net=model,
                            EVALS_THRESH=0.00001,
                            bins=100,
                            pl_fitting='median',
                            xmin_pos=2,
                            conv_norm=0.5,
                            filter_zeros=True
                        )
                        metrics['esd_metrics'][step] = esd_results
                        print("ESD metrics computed successfully.")
                    except Exception as e:
                        print(f"Error computing ESD metrics: {e}")
                        metrics['esd_metrics'][step] = None
                    
                    print(f"  Train accuracy: {metrics['train_accuracy'][step]:.4f}")
                    print(f"  Test accuracy: {metrics['test_accuracy'][step]:.4f}")
                    
                    # Save metrics to pickle file
                    if save_metrics:
                        metrics_filename = f"metrics/metrics_dataset_{dataset_size}_step_{step}.pkl"
                        with open(metrics_filename, 'wb') as f:
                            pickle.dump(metrics, f)
                        print(f"Metrics saved to {metrics_filename}")
                
                if step >= max_steps:
                    break
            
            if step >= max_steps:
                break
        
        # Create visualization
        fig = create_plots(metrics, dataset_size)
        figures.append(fig)
        
        # Save figure
        pio.write_html(fig, f"mnist_training_dynamics_{dataset_size}.html")
        
        # Save final metrics
        if save_metrics:
            final_metrics_filename = f"metrics/final_metrics_dataset_{dataset_size}.pkl"
            with open(final_metrics_filename, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"Final metrics saved to {final_metrics_filename}")
    
    return figures

# Function to load and visualize saved metrics
def load_and_visualize_metrics(metrics_dir='metrics', dataset_size=None):
    """
    Load metrics from pickle files and create visualizations
    
    Args:
        metrics_dir (str): Directory containing metrics files
        dataset_size (int, optional): If provided, only load metrics for this dataset size
    
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # Check if metrics directory exists
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory '{metrics_dir}' not found. Please run training first.")
        return figures
    
    # Find all final metrics files
    for filename in os.listdir(metrics_dir):
        if filename.startswith('final_metrics_dataset_') and filename.endswith('.pkl'):
            try:
                # Extract dataset size from filename
                file_dataset_size = int(filename.split('_')[-1].split('.')[0])
                
                # Skip if we're only interested in a specific dataset size
                if dataset_size is not None and file_dataset_size != dataset_size:
                    continue
                
                # Load metrics from pickle file
                filepath = os.path.join(metrics_dir, filename)
                print(f"Loading metrics from {filepath}")
                with open(filepath, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Create visualization
                fig = create_plots(metrics, file_dataset_size)
                figures.append(fig)
                
                # Save figure
                output_file = f"mnist_training_dynamics_{file_dataset_size}_from_saved.html"
                pio.write_html(fig, output_file)
                print(f"Visualization saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if not figures:
        print("No metrics files found or could be processed.")
    
    return figures

# Function to create a new visualization with ESD metrics
def create_esd_visualization(metrics_dir='metrics', dataset_size=None):
    """
    Create a specialized visualization focusing on ESD metrics
    
    Args:
        metrics_dir (str): Directory containing metrics files
        dataset_size (int, optional): If provided, only load metrics for this dataset size
    
    Returns:
        list: List of plotly figures
    """
    figures = []
    
    # Check if metrics directory exists
    if not os.path.exists(metrics_dir):
        print(f"Metrics directory '{metrics_dir}' not found. Please run training first.")
        return figures
    
    # Find all final metrics files
    for filename in os.listdir(metrics_dir):
        if filename.startswith('final_metrics_dataset_') and filename.endswith('.pkl'):
            try:
                # Extract dataset size from filename
                file_dataset_size = int(filename.split('_')[-1].split('.')[0])
                
                # Skip if we're only interested in a specific dataset size
                if dataset_size is not None and file_dataset_size != dataset_size:
                    continue
                
                # Load metrics from pickle file
                filepath = os.path.join(metrics_dir, filename)
                print(f"Loading metrics from {filepath}")
                with open(filepath, 'rb') as f:
                    metrics = pickle.load(f)
                
                # Create ESD visualization
                steps = list(metrics['train_accuracy'].keys())
                
                # Check if ESD metrics exist
                if 'esd_metrics' not in metrics or not any(metrics['esd_metrics'].values()):
                    print(f"No ESD metrics found for dataset size {file_dataset_size}")
                    continue
                
                # Find the first step with valid ESD metrics
                valid_steps = [step for step in steps if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None]
                if not valid_steps:
                    print(f"No valid ESD metrics found for dataset size {file_dataset_size}")
                    continue
                
                first_valid_step = valid_steps[0]
                
                # Create figure with 3 subplots for alpha, D, and spectral_norm
                fig = make_subplots(rows=3, cols=1, 
                                    subplot_titles=("Alpha Values", "D Values", "Spectral Norm"),
                                    vertical_spacing=0.1,
                                    row_heights=[0.33, 0.33, 0.33])
                
                # Colors for different layers
                colors = [
                    '#DAA520',  # yellow-gold
                    '#9ACD32',  # yellowgreen
                    '#228B22',  # forest green
                    '#20B2AA',  # light sea green
                    '#4682B4',  # steel blue
                    '#483D8B',  # dark slate blue
                    '#8B008B',  # dark magenta
                    '#CD5C5C',  # indian red
                    '#FF69B4',  # hot pink
                    '#FF8C00',  # dark orange
                    '#8B4513',  # saddle brown
                ]
                
                # Get layer names
                layer_names = metrics['esd_metrics'][first_valid_step]['longname']
                
                # Plot alpha values
                for i, layer_name in enumerate(layer_names):
                    alpha_values = []
                    for step in steps:
                        if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None:
                            if i < len(metrics['esd_metrics'][step]['alpha']):
                                alpha_values.append(metrics['esd_metrics'][step]['alpha'][i])
                            else:
                                alpha_values.append(None)
                        else:
                            alpha_values.append(None)
                    
                    # Filter out None values
                    valid_steps_for_layer = [s for s, a in zip(steps, alpha_values) if a is not None]
                    valid_alphas = [a for a in alpha_values if a is not None]
                    
                    if valid_alphas:
                        fig.add_trace(
                            go.Scatter(
                                x=valid_steps_for_layer, 
                                y=valid_alphas,
                                mode='lines+markers',
                                name=layer_name,
                                line=dict(color=colors[i % len(colors)]),
                            ),
                            row=1, col=1
                        )
                
                # Plot D values
                for i, layer_name in enumerate(layer_names):
                    d_values = []
                    for step in steps:
                        if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None:
                            if i < len(metrics['esd_metrics'][step]['D']):
                                d_values.append(metrics['esd_metrics'][step]['D'][i])
                            else:
                                d_values.append(None)
                        else:
                            d_values.append(None)
                    
                    # Filter out None values
                    valid_steps_for_layer = [s for s, a in zip(steps, d_values) if a is not None]
                    valid_d_values = [d for d in d_values if d is not None]
                    
                    if valid_d_values:
                        fig.add_trace(
                            go.Scatter(
                                x=valid_steps_for_layer, 
                                y=valid_d_values,
                                mode='lines',
                                name=layer_name,
                                line=dict(color=colors[i % len(colors)]),
                                showlegend=False,  # Use same legend as first plot
                            ),
                            row=2, col=1
                        )
                
                # Plot spectral norm values
                for i, layer_name in enumerate(layer_names):
                    norm_values = []
                    for step in steps:
                        if step in metrics['esd_metrics'] and metrics['esd_metrics'][step] is not None:
                            if i < len(metrics['esd_metrics'][step]['spectral_norm']):
                                norm_values.append(metrics['esd_metrics'][step]['spectral_norm'][i])
                            else:
                                norm_values.append(None)
                        else:
                            norm_values.append(None)
                    
                    # Filter out None values
                    valid_steps_for_layer = [s for s, a in zip(steps, norm_values) if a is not None]
                    valid_norm_values = [n for n in norm_values if n is not None]
                    
                    if valid_norm_values:
                        fig.add_trace(
                            go.Scatter(
                                x=valid_steps_for_layer, 
                                y=valid_norm_values,
                                mode='lines',
                                name=layer_name,
                                line=dict(color=colors[i % len(colors)]),
                                showlegend=False,  # Use same legend as first plot
                            ),
                            row=3, col=1
                        )
                
                # Update layout
                fig.update_layout(
                    height=900,
                    width=800,
                    title_text=f"ESD Metrics with |D^train|={file_dataset_size}",
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99
                    )
                )
                
                # Update y-axis labels
                fig.update_yaxes(title_text="Alpha", row=1, col=1)
                fig.update_yaxes(title_text="D Value", row=2, col=1)
                fig.update_yaxes(title_text="Spectral Norm", row=3, col=1)
                
                # Update x-axis labels
                for row in range(1, 4):
                    fig.update_xaxes(title_text="Step", row=row, col=1)
                
                figures.append(fig)
                
                # Save figure
                output_file = f"esd_metrics_{file_dataset_size}.html"
                pio.write_html(fig, output_file)
                print(f"ESD visualization saved to {output_file}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if not figures:
        print("No metrics files found or could be processed.")
    
    return figures

# Run the training and visualization
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and visualize neural network dynamics with ESD metrics")
    parser.add_argument("--dataset_sizes", nargs="+", type=int, default=[2000, 5000, 7000], 
                        help="Dataset sizes to train on")
    parser.add_argument("--max_steps", type=int, default=100000, 
                        help="Maximum number of training steps")
    parser.add_argument("--step_size", type=int, default=5000, 
                        help="Interval for saving metrics")
    parser.add_argument("--save_metrics", type=bool, default=True, 
                        help="Whether to save metrics")
    parser.add_argument("--init_scale", type=float, default=8.0, 
                        help="Initialization scaling factor (default: 8.0 as per paper)")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay parameter (default: 0.01 as per paper)")
    
    args = parser.parse_args()
    
    figures = train_and_visualize(
        train_dataset_sizes=args.dataset_sizes,
        max_steps=args.max_steps,
        step_size=args.step_size,
        save_metrics=args.save_metrics,
        init_scale=args.init_scale,
        weight_decay=args.weight_decay
    )
    print("Training and visualization complete. HTML files saved.")