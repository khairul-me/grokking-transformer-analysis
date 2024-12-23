# Grokking Transformer Analysis

A comprehensive framework for investigating and visualizing the grokking phenomenon in transformer models.

## Architecture
```mermaid
graph TD
    A[Data Generation] --> B[Transformer Model]
    B --> C[Training Loop]
    C --> D[Mechanistic Analysis]
    D --> E[Visualization Suite]
    
    C -->|Early Detection| F[Grokking Detection]
    C -->|Checkpoints| G[Model Snapshots]
    
    D -->|Patterns| H[Attention Analysis]
    D -->|Statistics| I[Activation Stats]
    D -->|Gradients| J[Gradient Flow]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px


Project Flow
mermaidCopysequenceDiagram
    participant D as Data
    participant M as Model
    participant T as Training
    participant A as Analysis
    
    D->>M: Generate Modular Arithmetic Data
    M->>T: Initialize Training
    loop Training Process
        T->>T: Execute Epoch
        T->>A: Track Metrics
        A->>T: Early Grokking Check
    end
    T->>A: Generate Visualizations
Overview
This project provides a comprehensive framework for studying the grokking phenomenon in transformer models through modular arithmetic tasks. It implements sophisticated training and analysis tools to detect and visualize sudden improvements in model generalization.
Features
Core Components

🔄 Enhanced Training Framework

Curriculum learning
Adaptive optimization
Early grokking detection
Automated checkpointing


🔍 Mechanistic Analysis

Attention pattern tracking
Activation statistics
Gradient flow analysis
Weight evolution monitoring


📊 Visualization Suite

Real-time metrics
Attention heatmaps
Learning dynamics
Performance analytics



Requirements
pythonCopytorch>=1.9.0
transformer_lens
numpy
pandas
matplotlib
seaborn
einops
tqdm
Installation

Clone the repository:

bashCopygit clone https://github.com/khairul-me/grokking-transformer-analysis.git
cd grokking-transformer-analysis

Install dependencies:

bashCopypip install -r requirements.txt
Quick Start
pythonCopyfrom grokking_research import EnhancedGrokkingExperiment

# Default configuration
experiment = EnhancedGrokkingExperiment()
experiment.run_enhanced_training()

# Custom configuration
config = {
    'num_epochs': 25000,
    'batch_size': 2048,
    'lr': 1e-3,
    'wd': 1.0
}
experiment = EnhancedGrokkingExperiment(config)
Project Structure
Copygrokking-transformer-analysis/
├── grokking_research.py      # Main implementation
├── experiments/             # Experiment outputs
│   └── grokking_{timestamp}/
│       ├── checkpoints/    # Model checkpoints
│       ├── analysis/       # Analysis outputs
│       ├── metrics/        # Training metrics
│       └── config.json     # Experiment config
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
Implementation Details
Training Process
mermaidCopygraph LR
    A[Data] --> B[Model]
    B --> C[Train]
    C --> D[Evaluate]
    D -->|Early Grokking| C
    D -->|Complete| E[Analysis]
Analysis Components
The project includes comprehensive analysis tools:

Attention pattern visualization
Training dynamics plots
Gradient flow analysis
Activation statistics

Results
The experiment generates various visualizations:

Training/test loss curves
Accuracy progression
Attention pattern heatmaps
Gradient norm evolution

Example output:
CopyTraining: 100%|██████████| 25000/25000 [12:14<00:00, 4.95it/s]
Early grokking detected at epoch 19996 with score 11367.7024
Training complete! Generating final analysis...
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Citation
If you use this code in your research, please cite:
bibtexCopy@software{grokking_transformer_analysis,
  title = {Grokking Transformer Analysis},
  author = {Islam, Khairul},
  year = {2024},
  url = {https://github.com/khairul-me/grokking-transformer-analysis}
}
Contact

Khairul Islam
Email: khairul.islam@hws.edu
GitHub: @khairul-me

Copy
This README includes:
1. Mermaid diagrams for visual explanation
2. Clear structure and organization
3. Your contact information
4. Code examples and installation instructions
5. Comprehensive project overview
You can directly copy this into a README.md file, and GitHub will render the mermaid diagrams automatically. Would you like me to explain any section in more detail or make any adjustments?
