import os
import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from analyze_esd import load_metrics

def create_htsr_directory(output_dir="htsr_analysis"):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def analyze_htsr_metrics(dataset_sizes=[2000, 5000], metrics_dir="metrics", output_dir="htsr_analysis"):
    """Analyze metrics through the lens of Heavy-Tailed Self Regularization theory."""
    output_dir = create_htsr_directory(output_dir)
    
    # Load metrics for each dataset size
    all_metrics = {}
    for size in dataset_sizes:
        metrics = load_metrics(metrics_dir, size)
        if metrics:
            all_metrics[size] = metrics
        else:
            print(f"No metrics found for dataset size {size}")
    
    if len(all_metrics) == 0:
        print("No metrics found for any dataset size.")
        return
    
    # Create HTSR visualizations
    analyze_alpha_distribution(all_metrics, output_dir)
    analyze_alpha_vs_generalization(all_metrics, output_dir)
    create_htsr_summary(all_metrics, output_dir)
    
    # Create index.html
    create_index_html(dataset_sizes, output_dir)
    
    print(f"HTSR analysis complete. Results saved to {output_dir}/")

def analyze_alpha_distribution(all_metrics, output_dir):
    """Analyze the distribution of alpha values across layers and training steps."""
    # Create histogram of alpha values for each dataset size
    for size, metrics_by_step in all_metrics.items():
        fig = go.Figure()
        
        # Collect alpha values from all layers and steps
        all_alpha_values = []
        
        for step in metrics_by_step.keys():
            step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
            alpha_values = metrics_by_step[step]['esd_metrics'][step_key]['alpha']
            all_alpha_values.extend(alpha_values)
        
        # Create histogram
        fig.add_trace(go.Histogram(
            x=all_alpha_values,
            nbinsx=30,
            name=f'Dataset Size {size}'
        ))
        
        # Add vertical lines for HTSR theory thresholds
        fig.add_vline(x=2, line_dash="dash", line_color="green", 
                      annotation_text="Optimal (α ≈ 2)", annotation_position="top right")
        fig.add_vline(x=4, line_dash="dash", line_color="orange", 
                      annotation_text="Upper Good Fit (α = 4)", annotation_position="top right")
        fig.add_vline(x=6, line_dash="dash", line_color="red", 
                      annotation_text="Overfit Threshold (α = 6)", annotation_position="top right")
        
        fig.update_layout(
            title=f'Distribution of Alpha Values for Dataset Size {size}',
            xaxis_title='Alpha Value',
            yaxis_title='Count',
            template='plotly_white'
        )
        
        pio.write_html(fig, file=os.path.join(output_dir, f'alpha_distribution_size_{size}.html'))

def analyze_alpha_vs_generalization(all_metrics, output_dir):
    """Analyze the relationship between alpha values and generalization gap."""
    # For each dataset size, plot alpha vs generalization gap (train_acc - test_acc)
    for size, metrics_by_step in all_metrics.items():
        # Get the first step to determine number of layers
        first_step = list(metrics_by_step.keys())[0]
        first_step_metrics = list(metrics_by_step[first_step]['esd_metrics'].keys())[0]
        num_layers = len(metrics_by_step[first_step]['esd_metrics'][first_step_metrics]['alpha'])
        
        # Create a figure for each layer
        for layer in range(num_layers):
            fig = go.Figure()
            
            # Collect data points
            alpha_values = []
            gen_gap_values = []
            step_values = []
            
            for step in sorted(metrics_by_step.keys()):
                step_key = list(metrics_by_step[step]['esd_metrics'].keys())[0]
                alpha = metrics_by_step[step]['esd_metrics'][step_key]['alpha'][layer]
                
                # Extract train and test accuracy values correctly
                # Check if they are dictionaries and extract the values
                train_acc = metrics_by_step[step]['train_accuracy']
                test_acc = metrics_by_step[step]['test_accuracy']
                
                if isinstance(train_acc, dict):
                    # If it's a dictionary, get the value for the current step
                    train_acc = train_acc.get(step, 0)
                
                if isinstance(test_acc, dict):
                    # If it's a dictionary, get the value for the current step
                    test_acc = test_acc.get(step, 0)
                
                # Calculate generalization gap
                gen_gap = train_acc - test_acc
                
                alpha_values.append(alpha)
                gen_gap_values.append(gen_gap)
                step_values.append(step)
            
            # Create scatter plot with color based on training step
            fig.add_trace(go.Scatter(
                x=alpha_values,
                y=gen_gap_values,
                mode='markers',
                marker=dict(
                    size=10,
                    color=step_values,
                    colorscale='Viridis',
                    colorbar=dict(title='Training Step'),
                    showscale=True
                ),
                text=[f'Step: {step}' for step in step_values],
                name=f'Layer {layer}'
            ))
            
            # Add horizontal line at zero generalization gap
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Add vertical lines for HTSR theory thresholds
            fig.add_vline(x=2, line_dash="dash", line_color="green", 
                        annotation_text="Optimal (α ≈ 2)", annotation_position="top right")
            fig.add_vline(x=4, line_dash="dash", line_color="orange", 
                        annotation_text="Upper Good Fit (α = 4)", annotation_position="top right")
            fig.add_vline(x=6, line_dash="dash", line_color="red", 
                        annotation_text="Overfit Threshold (α = 6)", annotation_position="top right")
            
            fig.update_layout(
                title=f'Alpha vs Generalization Gap for Layer {layer} (Dataset Size {size})',
                xaxis_title='Alpha Value',
                yaxis_title='Generalization Gap (Train Acc - Test Acc)',
                template='plotly_white'
            )
            
            pio.write_html(fig, file=os.path.join(output_dir, f'alpha_vs_gen_gap_size_{size}_layer_{layer}.html'))

def create_htsr_summary(all_metrics, output_dir):
    """Create a summary visualization of HTSR metrics across dataset sizes."""
    # Create a figure with subplots for each dataset size
    fig = make_subplots(rows=len(all_metrics), cols=1, 
                        subplot_titles=[f'Dataset Size {size}' for size in all_metrics.keys()])
    
    row = 1
    for size, metrics_by_step in all_metrics.items():
        # Get the final step
        final_step = max(metrics_by_step.keys())
        step_key = list(metrics_by_step[final_step]['esd_metrics'].keys())[0]
        
        # Get alpha values for the final step
        alpha_values = metrics_by_step[final_step]['esd_metrics'][step_key]['alpha']
        layers = list(range(len(alpha_values)))
        
        # Create bar chart
        fig.add_trace(go.Bar(
            x=layers,
            y=alpha_values,
            name=f'Size {size}',
            marker_color=['green' if a <= 2.5 else 'orange' if a <= 4 else 'red' for a in alpha_values]
        ), row=row, col=1)
        
        # Add horizontal lines for HTSR theory thresholds
        fig.add_hline(y=2, line_dash="dash", line_color="green", row=row, col=1)
        fig.add_hline(y=4, line_dash="dash", line_color="orange", row=row, col=1)
        fig.add_hline(y=6, line_dash="dash", line_color="red", row=row, col=1)
        
        row += 1
    
    fig.update_layout(
        title='Final Alpha Values Across Layers by Dataset Size',
        xaxis_title='Layer',
        yaxis_title='Alpha Value',
        template='plotly_white',
        height=300 * len(all_metrics)
    )
    
    pio.write_html(fig, file=os.path.join(output_dir, 'htsr_summary.html'))

def create_index_html(dataset_sizes, output_dir):
    """Create an index.html file for the HTSR analysis visualizations."""
    # Get the first dataset size to determine number of layers
    first_size = dataset_sizes[0]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HTSR Analysis</title>
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
        .theory {{
            background-color: #fff8e1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffa000;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>Heavy-Tailed Self Regularization (HTSR) Analysis</h1>
    
    <div class="description">
        <p>This dashboard provides analysis of the neural network training results through the lens of Heavy-Tailed Self Regularization (HTSR) theory.</p>
    </div>
    
    <div class="theory">
        <h2>HTSR Theory</h2>
        <p>According to Heavy-Tailed Self Regularization theory by Charles Martin, PhD, the power-law exponent (alpha) of the eigenvalue distribution of weight matrices provides insights into the generalization capabilities of neural networks:</p>
        <ul>
            <li><strong>Alpha ≈ 2:</strong> Optimal value for quality layers with good generalization</li>
            <li><strong>Alpha between 2-4:</strong> Good fit to the data</li>
            <li><strong>Alpha > 6:</strong> Indication of overfitting</li>
        </ul>
        <p>This analysis examines how alpha values relate to generalization performance across different dataset sizes and training steps.</p>
    </div>
    
    <div class="section">
        <h2>HTSR Summary</h2>
        <p>This visualization shows the final alpha values across all layers for each dataset size, color-coded according to HTSR theory thresholds.</p>
        <div class="metric-item"><a href="htsr_summary.html">View HTSR Summary</a></div>
    </div>
    
    <div class="section">
        <h2>Alpha Value Distributions</h2>
        <p>These visualizations show the distribution of alpha values across all layers and training steps for each dataset size.</p>
        <div class="metric-group">
"""
    
    for size in dataset_sizes:
        html_content += f'            <div class="metric-item"><a href="alpha_distribution_size_{size}.html">Dataset Size {size}</a></div>\n'
    
    html_content += """        </div>
    </div>
    
    <div class="section">
        <h2>Alpha vs Generalization Gap</h2>
        <p>These visualizations show the relationship between alpha values and the generalization gap (train accuracy - test accuracy) for each layer and dataset size.</p>
"""
    
    for size in dataset_sizes:
        html_content += f'        <h3>Dataset Size {size}</h3>\n'
        html_content += '        <div class="metric-group">\n'
        
        # Assume 12 layers based on previous code
        for layer in range(12):
            html_content += f'            <div class="metric-item"><a href="alpha_vs_gen_gap_size_{size}_layer_{layer}.html">Layer {layer}</a></div>\n'
        
        html_content += '        </div>\n'
    
    html_content += """    </div>
    
    <div class="section">
        <h2>Navigation</h2>
        <p><a href="../index.html">Back to Main Dashboard</a></p>
        <p><a href="../esd_comparison/index.html">View ESD Comparison Dashboard</a></p>
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
    
    parser = argparse.ArgumentParser(description='Analyze metrics through the lens of HTSR theory')
    parser.add_argument('--dataset_sizes', type=int, nargs='+', default=[2000, 5000],
                        help='List of dataset sizes to analyze')
    parser.add_argument('--metrics_dir', type=str, default='metrics',
                        help='Directory containing metrics files')
    parser.add_argument('--output_dir', type=str, default='htsr_analysis',
                        help='Directory to save HTSR analysis visualizations')
    
    args = parser.parse_args()
    
    analyze_htsr_metrics(args.dataset_sizes, args.metrics_dir, args.output_dir)