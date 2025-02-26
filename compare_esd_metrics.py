import os
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from analyze_esd import load_metrics

def create_comparison_directory(output_dir="esd_comparison"):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def compare_metrics_across_datasets(dataset_sizes=[2000, 5000], metrics_dir="metrics", output_dir="esd_comparison"):
    """Compare ESD metrics across different dataset sizes."""
    output_dir = create_comparison_directory(output_dir)
    
    # Load metrics for each dataset size
    all_metrics = {}
    for size in dataset_sizes:
        metrics = load_metrics(metrics_dir, size)
        if metrics:
            all_metrics[size] = metrics
        else:
            print(f"No metrics found for dataset size {size}")
    
    if len(all_metrics) < 2:
        print("Not enough datasets to compare. Need at least 2.")
        return
    
    # Create comparison visualizations
    compare_alpha_across_datasets(all_metrics, output_dir)
    compare_d_across_datasets(all_metrics, output_dir)
    compare_spectral_norm_across_datasets(all_metrics, output_dir)
    compare_final_eigenvalue_distributions(all_metrics, output_dir)
    compare_metrics_vs_accuracy(all_metrics, output_dir)
    
    # Create index.html
    create_index_html(dataset_sizes, output_dir)
    
    print(f"Comparison complete. Results saved to {output_dir}/")

def compare_alpha_across_datasets(all_metrics, output_dir):
    """Compare alpha values across different dataset sizes for each layer."""
    # Get the first dataset size
    first_size = list(all_metrics.keys())[0]
    # Get the first step
    first_step = list(all_metrics[first_size].keys())[0]
    # Get the first step's esd_metrics
    first_step_metrics = list(all_metrics[first_size][first_step]['esd_metrics'].keys())[0]
    # Get the number of layers from the alpha values of the first step
    num_layers = len(all_metrics[first_size][first_step]['esd_metrics'][first_step_metrics]['alpha'])
    
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            alpha_values = []
            step_values = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                alpha_values.append(metrics_by_step[step]['esd_metrics'][step_key]['alpha'][layer])
                step_values.append(step)
            
            fig.add_trace(go.Scatter(
                x=step_values,
                y=alpha_values,
                mode='lines+markers',
                name=f'Dataset Size {size}'
            ))
        
        fig.update_layout(
            title=f'Alpha Values Comparison for Layer {layer}',
            xaxis_title='Training Step',
            yaxis_title='Alpha Value',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_alpha_layer_{layer}.html'))

def compare_d_across_datasets(all_metrics, output_dir):
    """Compare D values across different dataset sizes for each layer."""
    # Get the first dataset size
    first_size = list(all_metrics.keys())[0]
    # Get the first step
    first_step = list(all_metrics[first_size].keys())[0]
    # Get the first step's esd_metrics
    first_step_metrics = list(all_metrics[first_size][first_step]['esd_metrics'].keys())[0]
    # Get the number of layers from the D values of the first step
    num_layers = len(all_metrics[first_size][first_step]['esd_metrics'][first_step_metrics]['D'])
    
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            d_values = []
            step_values = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                d_values.append(metrics_by_step[step]['esd_metrics'][step_key]['D'][layer])
                step_values.append(step)
            
            fig.add_trace(go.Scatter(
                x=step_values,
                y=d_values,
                mode='lines+markers',
                name=f'Dataset Size {size}'
            ))
        
        fig.update_layout(
            title=f'D Values Comparison for Layer {layer}',
            xaxis_title='Training Step',
            yaxis_title='D Value',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_d_layer_{layer}.html'))

def compare_spectral_norm_across_datasets(all_metrics, output_dir):
    """Compare spectral norm values across different dataset sizes for each layer."""
    # Get the first dataset size
    first_size = list(all_metrics.keys())[0]
    # Get the first step
    first_step = list(all_metrics[first_size].keys())[0]
    # Get the first step's esd_metrics
    first_step_metrics = list(all_metrics[first_size][first_step]['esd_metrics'].keys())[0]
    # Get the number of layers from the spectral_norm values of the first step
    num_layers = len(all_metrics[first_size][first_step]['esd_metrics'][first_step_metrics]['spectral_norm'])
    
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            norm_values = []
            step_values = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                norm_values.append(metrics_by_step[step]['esd_metrics'][step_key]['spectral_norm'][layer])
                step_values.append(step)
            
            fig.add_trace(go.Scatter(
                x=step_values,
                y=norm_values,
                mode='lines+markers',
                name=f'Dataset Size {size}'
            ))
        
        fig.update_layout(
            title=f'Spectral Norm Comparison for Layer {layer}',
            xaxis_title='Training Step',
            yaxis_title='Spectral Norm',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_spectral_norm_layer_{layer}.html'))

def compare_final_eigenvalue_distributions(all_metrics, output_dir):
    """Compare final eigenvalue distributions across different dataset sizes."""
    # Get the first dataset size
    first_size = list(all_metrics.keys())[0]
    # Get the first step
    first_step = list(all_metrics[first_size].keys())[0]
    # Get the first step's esd_metrics
    first_step_metrics = list(all_metrics[first_size][first_step]['esd_metrics'].keys())[0]
    # Get the number of layers from the eigenvalues of the first step
    num_layers = len(all_metrics[first_size][first_step]['esd_metrics'][first_step_metrics]['eigs'])
    
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            # Get the final step
            final_step = max(metrics_by_step.keys())
            # The step is stored as a key in the esd_metrics dictionary
            step_key = list(metrics_by_step[final_step]['esd_metrics'].keys())[0]
            eigenvalues = metrics_by_step[final_step]['esd_metrics'][step_key]['eigs'][layer]
            
            # Create histogram
            fig.add_trace(go.Histogram(
                x=eigenvalues,
                histnorm='probability density',
                name=f'Dataset Size {size}',
                opacity=0.7,
                nbinsx=50
            ))
        
        fig.update_layout(
            title=f'Final Eigenvalue Distribution Comparison for Layer {layer}',
            xaxis_title='Eigenvalue',
            yaxis_title='Probability Density',
            barmode='overlay',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_eigenvalue_dist_layer_{layer}.html'))

def compare_metrics_vs_accuracy(all_metrics, output_dir):
    """Compare how metrics correlate with test accuracy across different dataset sizes."""
    # Get the first dataset size
    first_size = list(all_metrics.keys())[0]
    # Get the first step
    first_step = list(all_metrics[first_size].keys())[0]
    # Get the first step's esd_metrics
    first_step_metrics = list(all_metrics[first_size][first_step]['esd_metrics'].keys())[0]
    # Get the number of layers from the alpha values of the first step
    num_layers = len(all_metrics[first_size][first_step]['esd_metrics'][first_step_metrics]['alpha'])
    
    # Compare alpha vs test accuracy
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            alpha_values = []
            test_acc = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                alpha_values.append(metrics_by_step[step]['esd_metrics'][step_key]['alpha'][layer])
                test_acc.append(metrics_by_step[step]['test_accuracy'])
            
            fig.add_trace(go.Scatter(
                x=alpha_values,
                y=test_acc,
                mode='markers',
                name=f'Dataset Size {size}',
                marker=dict(
                    size=10,
                    opacity=0.7,
                    line=dict(width=1)
                )
            ))
        
        fig.update_layout(
            title=f'Alpha vs Test Accuracy Comparison for Layer {layer}',
            xaxis_title='Alpha Value',
            yaxis_title='Test Accuracy',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_alpha_vs_accuracy_layer_{layer}.html'))
    
    # Compare D vs test accuracy
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            d_values = []
            test_acc = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                d_values.append(metrics_by_step[step]['esd_metrics'][step_key]['D'][layer])
                test_acc.append(metrics_by_step[step]['test_accuracy'])
            
            fig.add_trace(go.Scatter(
                x=d_values,
                y=test_acc,
                mode='markers',
                name=f'Dataset Size {size}',
                marker=dict(
                    size=10,
                    opacity=0.7,
                    line=dict(width=1)
                )
            ))
        
        fig.update_layout(
            title=f'D vs Test Accuracy Comparison for Layer {layer}',
            xaxis_title='D Value',
            yaxis_title='Test Accuracy',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_d_vs_accuracy_layer_{layer}.html'))
    
    # Compare spectral norm vs test accuracy
    for layer in range(num_layers):
        fig = go.Figure()
        
        for size, metrics_by_step in all_metrics.items():
            steps = sorted(metrics_by_step.keys())
            norm_values = []
            test_acc = []
            
            for step in steps:
                # The step is stored as a key in the esd_metrics dictionary
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                norm_values.append(metrics_by_step[step]['esd_metrics'][step_key]['spectral_norm'][layer])
                test_acc.append(metrics_by_step[step]['test_accuracy'])
            
            fig.add_trace(go.Scatter(
                x=norm_values,
                y=test_acc,
                mode='markers',
                name=f'Dataset Size {size}',
                marker=dict(
                    size=10,
                    opacity=0.7,
                    line=dict(width=1)
                )
            ))
        
        fig.update_layout(
            title=f'Spectral Norm vs Test Accuracy Comparison for Layer {layer}',
            xaxis_title='Spectral Norm',
            yaxis_title='Test Accuracy',
            legend_title='Dataset Size',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'compare_spectral_norm_vs_accuracy_layer_{layer}.html'))

def create_index_html(dataset_sizes, output_dir):
    """Create an index.html file for the comparison visualizations."""
    num_layers = 12  # Assuming 12 layers based on previous code
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ESD Metrics Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
        }}
        .metric-group {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-item {{
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }}
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .description {{
            margin-bottom: 20px;
            background-color: #f0f7ff;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #0066cc;
        }}
        .comparison {{
            background-color: #fff8e1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffa000;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>ESD Metrics Comparison Dashboard</h1>
    
    <div class="description">
        <p>This dashboard provides direct comparisons of Empirical Spectral Density (ESD) metrics across different dataset sizes.
        These comparisons can help understand how dataset size affects the learning dynamics and generalization properties of neural networks.</p>
        
        <p><strong>Dataset Sizes Compared:</strong> {', '.join(map(str, dataset_sizes))}</p>
        
        <p><strong>Key Metrics:</strong></p>
        <ul>
            <li><strong>Alpha:</strong> Power-law exponent of the eigenvalue distribution. Lower values indicate more complex representations.</li>
            <li><strong>D:</strong> Kolmogorov-Smirnov statistic measuring the goodness of fit of the power-law distribution.</li>
            <li><strong>Spectral Norm:</strong> The largest eigenvalue of the weight matrix, indicating the maximum amplification the layer can apply.</li>
        </ul>
    </div>
    
    <div class="comparison">
        <p><strong>View Individual Dataset Analyses:</strong></p>
        <ul>
    """
    
    for size in dataset_sizes:
        html_content += f'            <li><a href="../esd_analysis{"_" + str(size) if size != 2000 else ""}/index.html">Dataset Size {size}</a></li>\n'
    
    html_content += f"""        </ul>
        <p><a href="../index.html">Back to Main Page</a></p>
    </div>
    
    <div class="section">
        <h2>Alpha Values Comparison</h2>
        <p>These plots compare how the alpha values (power-law exponents) change during training across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_alpha_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>D Values Comparison</h2>
        <p>These plots compare how the D values (goodness of fit) change during training across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_d_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>Spectral Norm Comparison</h2>
        <p>These plots compare how the spectral norm values change during training across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_spectral_norm_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>Final Eigenvalue Distributions Comparison</h2>
        <p>These plots compare the distribution of eigenvalues at the final training step across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_eigenvalue_dist_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>Alpha vs Test Accuracy Comparison</h2>
        <p>These plots compare the relationship between alpha values and test accuracy across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_alpha_vs_accuracy_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>D vs Test Accuracy Comparison</h2>
        <p>These plots compare the relationship between D values and test accuracy across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_d_vs_accuracy_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>Spectral Norm vs Test Accuracy Comparison</h2>
        <p>These plots compare the relationship between spectral norm values and test accuracy across different dataset sizes for each layer.</p>
        <div class="metric-group">
"""
    
    for layer in range(num_layers):
        html_content += f'            <div class="metric-item"><a href="compare_spectral_norm_vs_accuracy_layer_{layer}.html">Layer {layer}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <footer>
        <p>Generated on February 25, 2025</p>
    </footer>
</body>
</html>"""
    
    with open(os.path.join(output_dir, 'index.html'), 'w') as f:
        f.write(html_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare ESD metrics across different dataset sizes')
    parser.add_argument('--dataset_sizes', type=int, nargs='+', default=[2000, 5000],
                        help='List of dataset sizes to compare')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='esd_comparison',
                        help='Directory to save comparison visualizations')
    
    args = parser.parse_args()
    
    compare_metrics_across_datasets(args.dataset_sizes, args.metrics_dir, args.output_dir)
