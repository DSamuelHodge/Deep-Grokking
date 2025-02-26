# Deep Grokking ESD Analysis

This project explores the relationship between Empirical Spectral Density (ESD) metrics and neural network learning dynamics. It provides tools for training neural networks, computing ESD metrics, and visualizing these metrics to gain insights into how networks learn and generalize.

## Theoretical Background

This project is based on reproducing the results found in the research paper [Deep Grokking: Would Deep Neural Networks Generalize Better](https://arxiv.org/html/2405.19454v1) and analyzing them through the lens of Heavy-Tailed Self Regularization (HTSR) by Charles Martin, PhD.

The HTSR theory predicts specific patterns in the eigenvalue distributions of weight matrices in neural networks:
- Power-law alpha ≈ 2 for quality layers with good generalization
- Good fit is typically between alpha values of 2-4
- Overfit is indicated by alpha values over 6

By analyzing these ESD metrics across different dataset sizes and training steps, we can gain insights into the generalization capabilities of neural networks and the conditions under which "grokking" (delayed generalization) occurs.

## Paper Reproduction

The original Deep Grokking paper shows distinct learning phases across different dataset sizes (2000, 5000, and 7000 examples):

1. **Dataset Size 2000**: Shows clear overfitting followed by generalization phases, with a dramatic drop in loss and increase in test accuracy around step 40K.
2. **Dataset Size 5000**: Similar pattern but with the generalization phase starting earlier, around step 35K.
3. **Dataset Size 7000**: Shows three distinct phases - overfitting, memorization, and compression, with alpha values showing more complex patterns.

The paper demonstrates that larger dataset sizes lead to earlier generalization and more complex learning dynamics. The alpha values (power-law exponents) of the weight matrices correlate strongly with these learning phases, supporting the HTSR theory.

To reproduce these results, we use the same model architecture and training parameters as in the paper, with training for up to 100,000 steps.

### Default Configuration Parameters

Following the paper specifications, our implementation uses these default parameters:

- **Network Architecture**: MLP with width 400 and ReLU activation
- **Initialization Scale**: α = 8.0 (scaling factor for standard PyTorch initialization)
- **Weight Decay**: γ = 0.01
- **Loss Function**: Mean Square Error (MSE)
- **Optimizer**: Adam with learning rate 1e-3
- **Training Steps**: 100,000

These parameters are critical for reproducing the grokking phenomenon as described in the paper.

## Overview

The project consists of several components:

1. **Training Framework**: A neural network training framework that tracks various metrics during training, including ESD metrics.
2. **ESD Metrics Computation**: Functions for computing ESD metrics such as alpha (power-law exponent), D (goodness of fit), and spectral norm.
3. **Visualization Tools**: Scripts for generating interactive visualizations of ESD metrics and their relationship to model performance.
4. **Comparison Analysis**: Tools for comparing ESD metrics across different dataset sizes.

## Key Files

- `grokking.py`: Contains the training framework and functions for training neural networks and tracking metrics.
- `esd.py`: Implements the ESD metrics computation.
- `analyze_esd.py`: Generates visualizations of ESD metrics for a specific dataset size.
- `compare_esd_metrics.py`: Compares ESD metrics across different dataset sizes.
- `paper_visualization.html`: Displays the key visualizations from the Deep Grokking paper with explanations.
- `reproduce_paper.py`: Script for reproducing the results from the Deep Grokking paper with configurable parameters.
- `analyze_htsr.py`: Analyzes the Heavy-Tailed Self-Regularization metrics and generates visualizations.

## Running the Analysis

### Training a Model

To train a model with a specific dataset size:

```bash
python -c "from grokking import train_and_visualize; train_and_visualize(train_dataset_sizes=[SIZE], max_steps=10000, step_size=1000)"
```

Replace `SIZE` with the desired dataset size (e.g., 2000, 5000).

You can also customize the training parameters:

```bash
python -c "from grokking import train_and_visualize; train_and_visualize(train_dataset_sizes=[SIZE], max_steps=10000, step_size=1000, init_scale=8.0, weight_decay=0.01)"
```

Available parameters:
- `train_dataset_sizes`: List of dataset sizes to train on
- `max_steps`: Maximum number of training steps
- `step_size`: Interval for saving metrics
- `save_metrics`: Whether to save metrics (default: True)
- `init_scale`: Initialization scaling factor (default: 8.0)
- `weight_decay`: Weight decay parameter (default: 0.01)

### Analyzing ESD Metrics

To analyze ESD metrics for a specific dataset size:

```bash
python analyze_esd.py --dataset_size SIZE --metrics_dir metrics --output_dir esd_analysis_SIZE
```

Replace `SIZE` with the dataset size you want to analyze.

### Analyzing HTSR Metrics

To analyze Heavy-Tailed Self-Regularization metrics:

```bash
python3 analyze_htsr.py --dataset_sizes 2000 5000 --metrics_dir metrics --output_dir htsr_analysis
```

This will generate visualizations that help understand the relationship between alpha values (power-law exponents) and generalization gap, providing insights into how HTSR theory explains the grokking phenomenon.

### Comparing ESD Metrics Across Dataset Sizes

To compare ESD metrics across different dataset sizes:

```bash
python compare_esd_metrics.py --dataset_sizes SIZE1 SIZE2 --metrics_dir metrics --output_dir esd_comparison
```

Replace `SIZE1` and `SIZE2` with the dataset sizes you want to compare.

### Reproducing Paper Results

To reproduce the results from the Deep Grokking paper:

```bash
python3 reproduce_paper.py --dataset_sizes 2000 5000 7000 --max_steps 100000 --step_size 5000
```

You can customize the parameters:
- `--dataset_sizes`: List of dataset sizes to train on (default: 2000 5000 7000)
- `--max_steps`: Maximum number of training steps (default: 100000)
- `--step_size`: Interval for saving metrics (default: 5000)
- `--save_metrics`: Whether to save metrics (default: True)
- `--init_scale`: Initialization scaling factor (default: 8.0)
- `--weight_decay`: Weight decay parameter (default: 0.01)

Note: Training for 100,000 steps can be computationally intensive. You can reduce the max_steps parameter for quicker results.

## Visualization Dashboard

After running the analysis, you can view the visualizations by opening the following files in a web browser:

- `index.html`: Main dashboard with links to individual analyses and comparisons.
- `paper_visualization.html`: Visualizations from the original Deep Grokking paper with explanations.
- `esd_analysis/index.html`: Visualizations for the 2000 dataset size.
- `esd_analysis_5000/index.html`: Visualizations for the 5000 dataset size.
- `esd_comparison/index.html`: Comparisons of ESD metrics across different dataset sizes.
- `htsr_analysis/index.html`: Analysis of Heavy-Tailed Self-Regularization metrics.

## ESD Metrics Explained

- **Alpha**: Power-law exponent of the eigenvalue distribution. Lower values indicate more complex representations.
- **D**: Kolmogorov-Smirnov statistic measuring the goodness of fit of the power-law distribution.
- **Spectral Norm**: The largest eigenvalue of the weight matrix, indicating the maximum amplification the layer can apply.

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- SciPy
- Plotly
- Matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.
