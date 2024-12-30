# Grokking Research Framework

## Overview
This project implements a comprehensive framework for studying the "grokking" phenomenon in neural networks, with a particular focus on modular arithmetic tasks. Grokking refers to the phenomenon where neural networks suddenly achieve good generalization after a long period of apparent overfitting.

## Key Features
- Modular arithmetic task implementation ((a+b) mod p)
- Transformer-based architecture using HookedTransformer
- Dynamic weight decay optimization
- Comprehensive mechanistic analysis tools
- Curriculum learning support
- Early grokking detection
- Advanced visualization capabilities

## Requirements
```
torch
einops
tqdm
matplotlib
numpy
seaborn
transformer_lens
pandas
```

## Project Structure
```
.
├── experiments/
│   └── grokking_{timestamp}/
│       ├── weights/
│       ├── heatmaps/
│       ├── metrics/
│       ├── checkpoints/
│       └── analysis/
├── grokking_research.py
└── README.md
```

## Core Components

### 1. Base Experiment Class (`BaseGrokkingExperiment`)
- Handles basic experiment setup and configuration
- Manages data generation and model initialization
- Implements core training loops and evaluation

### 2. Mechanistic Analysis (`MechanisticAnalysis`)
Provides tools for analyzing:
- Attention patterns
- Weight evolution
- Activation statistics
- Gradient flow
- Feature attribution

### 3. Optimization Components (`GrokkingOptimizer`)
Features:
- Dynamic weight decay scheduling
- Curriculum learning implementation
- Early grokking detection
- Batch size scheduling
- Learning rate adaptation

### 4. Enhanced Experiment Class (`EnhancedGrokkingExperiment`)
Combines all components for comprehensive experiments with:
- Advanced training loops
- Integrated analysis
- Visualization generation
- Metric tracking
- Checkpoint management

## Usage

### Basic Example
```python
# Configure the experiment
base_config = {
    'num_epochs': 25000,
    'batch_size': 2048,
    'lr': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create optimization strategy
strategy = OptimizationStrategy(
    name="adaptive_curriculum",
    learning_rate_schedule="adaptive",
    weight_decay_schedule="increasing",
    batch_size_schedule="curriculum",
    early_grok_detection=True,
    curriculum_learning=True
)

# Initialize and run experiment
experiment = EnhancedGrokkingExperiment(base_config)
experiment.generate_data()
experiment.create_model()
experiment.setup_training()
experiment.run_enhanced_training(strategy)
```

### Custom Task Configuration
```python
task_config = TaskConfig(
    task_type=TaskType.MODULAR_ARITHMETIC,
    input_size=113,
    output_size=113,
    sequence_length=3
)

experiment = EnhancedGrokkingExperiment(base_config, task_config)
```

## Configuration Options

### Task Types
- `MODULAR_ARITHMETIC`
- `MATRIX_MULTIPLICATION`
- `SEQUENCE_PREDICTION`
- `SYMBOLIC_REASONING`

### Optimization Strategies
1. Learning Rate Schedules:
   - `constant`
   - `cyclic`
   - `adaptive`

2. Weight Decay Schedules:
   - `constant`
   - `increasing`
   - `adaptive`

3. Batch Size Schedules:
   - `constant`
   - `increasing`
   - `curriculum`

## Analysis Tools

### 1. Training Dynamics
```python
experiment.create_training_visualizations(save_dir)
```
Generates plots for:
- Training/test loss trajectories
- Accuracy progression
- Learning rate changes
- Weight decay adaptation

### 2. Mechanistic Analysis
```python
experiment.mechanistic_analyzer.create_visualizations(save_dir)
```
Provides visualizations of:
- Attention patterns
- Weight evolution
- Activation statistics
- Gradient flow

## Experimental Results
Results are automatically saved in the `experiments/grokking_{timestamp}/` directory:
- Training metrics (JSON)
- Visualizations (PNG)
- Model checkpoints (PT)
- Analysis reports

## Contributing
This is a research framework designed to be extended. Key areas for contribution:
1. New task types
2. Additional optimization strategies
3. Enhanced analysis tools
4. Improved visualization methods

## References
1. Power, A., Abnar, S., & Sutskever, I. (2022). "Grokking: Generalization Beyond Memorization in Neural Networks."
2. Liu, D., & Ritter, S. (2023). "Omnigrok: Grokking Beyond Algorithmic Data."
3. Stander, D., Yu, Q., Fan, H., & Biderman, S. (2023). "Grokking Group Multiplication with Cosets."
