# grokking_research.py
import torch
import einops
import tqdm.auto as tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import time
from transformer_lens import HookedTransformer, HookedTransformerConfig
import pandas as pd
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum


# ======== Base Configurations and Enums ========
class TaskType(Enum):
    MODULAR_ARITHMETIC = "modular_arithmetic"
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    SEQUENCE_PREDICTION = "sequence_prediction"
    SYMBOLIC_REASONING = "symbolic_reasoning"


class DomainType(Enum):
    MATHEMATICAL = "mathematical"
    LANGUAGE = "language"
    SYMBOLIC = "symbolic"
    VISUAL = "visual"


@dataclass
class TaskConfig:
    task_type: TaskType
    input_size: int
    output_size: int
    sequence_length: int
    extra_params: Dict[str, Any] = None


@dataclass
class DomainConfig:
    domain_type: DomainType
    task_complexity: int  # 1-10 scale
    input_modality: str
    output_modality: str
    extra_params: Dict = None


@dataclass
class OptimizationStrategy:
    name: str
    learning_rate_schedule: str  # 'constant', 'cyclic', 'adaptive'
    weight_decay_schedule: str  # 'constant', 'increasing', 'adaptive'
    batch_size_schedule: str  # 'constant', 'increasing', 'curriculum'
    early_grok_detection: bool
    curriculum_learning: bool


# ======== Base Experiment Class ========
class BaseGrokkingExperiment:
    def __init__(self, config=None):
        """Initialize base experiment"""
        self.default_config = {
            'p': 113,  # Modulo base
            'frac_train': 0.3,  # Training data fraction
            'lr': 1e-3,  # Learning rate
            'wd': 1.0,  # Weight decay
            'betas': (0.9, 0.98),  # Adam optimizer betas
            'num_epochs': 25000,  # Training epochs
            'checkpoint_every': 5000,  # Checkpoint frequency
            'weight_track_every': 5000,  # Weight tracking frequency
            'batch_size': 2048,  # Batch size
            'data_seed': 598,  # Data generation seed
            'model_seed': 999,  # Model initialization seed
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_config': {
                'n_layers': 1,
                'n_heads': 4,
                'd_model': 64,
                'd_head': 16,
                'd_mlp': 256,
                'act_fn': "relu",
                'normalization_type': None
            }
        }

        # Update config with any overrides
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)

        # Initialize metrics storage
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        # Set random seeds
        self._set_random_seeds()

        # Setup directories
        self.setup_directories()

    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.config['model_seed'])
        np.random.seed(self.config['data_seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['model_seed'])

    def setup_directories(self):
        """Setup experiment directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f'experiments/grokking_{timestamp}')

        # Create subdirectories
        self.dirs = {
            'weights': self.exp_dir / 'weights',
            'heatmaps': self.exp_dir / 'heatmaps',
            'metrics': self.exp_dir / 'metrics',
            'checkpoints': self.exp_dir / 'checkpoints',
            'analysis': self.exp_dir / 'analysis'
        }

        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def generate_data(self):
        """Generate modular arithmetic dataset"""
        p = self.config['p']

        # Create input vectors
        a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        equals_vector = einops.repeat(torch.tensor(p), " -> (i j)", i=p, j=p)

        # Create dataset and labels
        self.dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)
        self.labels = (self.dataset[:, 0] + self.dataset[:, 1]) % p

        # Split into train and test
        indices = torch.randperm(p * p)
        cutoff = int(p * p * self.config['frac_train'])

        self.train_indices = indices[:cutoff]
        self.test_indices = indices[cutoff:]

        # Move to device
        self.train_data = self.dataset[self.train_indices].to(self.config['device'])
        self.train_labels = self.labels[self.train_indices].to(self.config['device']).long()
        self.test_data = self.dataset[self.test_indices].to(self.config['device'])
        self.test_labels = self.labels[self.test_indices].to(self.config['device']).long()

    def create_model(self):
        """Create and initialize the transformer model"""
        p = self.config['p']

        # Create model configuration
        cfg = HookedTransformerConfig(
            d_vocab=p + 1,
            d_vocab_out=p,
            n_ctx=3,
            seed=self.config['model_seed'],
            device=self.config['device'],
            **self.config['model_config']
        )

        # Initialize model
        self.model = HookedTransformer(cfg)

        # Freeze bias parameters
        for name, param in self.model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        self.model.to(self.config['device'])

    def setup_training(self):
        """Setup optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['wd'],
            betas=self.config['betas']
        )

    def loss_fn(self, logits, labels):
        """Calculate cross-entropy loss"""
        if len(logits.shape) == 3:
            logits = logits[:, -1]  # Take the last token's predictions
        return torch.nn.functional.cross_entropy(logits, labels)

    def calculate_accuracy(self, logits, labels):
        """Calculate accuracy"""
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == labels).float().mean().item()


# ======== Mechanistic Analysis Class ========

class MechanisticAnalysis:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.attention_patterns = []
        self.weight_evolution = {}
        self.activation_stats = []
        self.gradient_stats = []
        self.feature_attribution = []

    def analyze_feature_attribution(self, input_data: torch.Tensor, labels: torch.Tensor):
        """Analyze feature attribution for inputs"""
        with torch.no_grad():
            # Basic implementation - just store the input data importance
            attributions = torch.sum(torch.abs(input_data), dim=-1)
            self.feature_attribution.append(attributions.cpu().numpy())

    def track_training_step(self, epoch: int, input_data: torch.Tensor, labels: torch.Tensor):
        """Track all relevant metrics for a training step"""
        if epoch % self.config['weight_track_every'] == 0:
            self.compute_attention_patterns(input_data)
            self.track_weight_evolution()
            self.compute_activation_statistics(input_data)
            self.compute_gradient_flow()
            if epoch % (self.config['weight_track_every'] * 5) == 0:
                self.analyze_feature_attribution(input_data, labels)

    def compute_attention_patterns(self, input_data: torch.Tensor):
        """Compute and store attention patterns"""
        self.model.eval()
        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_data)
            pattern = cache['pattern', 0]
            self.attention_patterns.append(pattern.cpu().numpy())

    def track_weight_evolution(self):
        """Track weight evolution during training"""
        weights = {
            'embedding': self.model.embed.W_E.detach().cpu(),
            'attention': {
                'query': self.model.blocks[0].attn.W_Q.detach().cpu(),
                'key': self.model.blocks[0].attn.W_K.detach().cpu(),
                'value': self.model.blocks[0].attn.W_V.detach().cpu(),
                'output': self.model.blocks[0].attn.W_O.detach().cpu(),
            },
            'mlp': {
                'in': self.model.blocks[0].mlp.W_in.detach().cpu(),
                'out': self.model.blocks[0].mlp.W_out.detach().cpu(),
            },
            'unembedding': self.model.unembed.W_U.detach().cpu()
        }

        for name, weight in weights.items():
            if isinstance(weight, dict):
                if name not in self.weight_evolution:
                    self.weight_evolution[name] = {}
                for sub_name, sub_weight in weight.items():
                    if sub_name not in self.weight_evolution[name]:
                        self.weight_evolution[name][sub_name] = []
                    self.weight_evolution[name][sub_name].append(sub_weight.numpy())
            else:
                if name not in self.weight_evolution:
                    self.weight_evolution[name] = []
                self.weight_evolution[name].append(weight.numpy())

    def compute_activation_statistics(self, input_data: torch.Tensor):
        """Track activation statistics"""
        self.model.eval()
        with torch.no_grad():
            _, cache = self.model.run_with_cache(input_data)

            stats = {
                'resid_pre': cache['resid_pre', 0].cpu().numpy(),
                'resid_post': cache['resid_post', 0].cpu().numpy(),
                'mlp_out': cache['mlp_out', 0].cpu().numpy(),
                'attn_out': cache['attn_out', 0].cpu().numpy()
            }

            self.activation_stats.append({
                k: {
                    'mean': v.mean(),
                    'std': v.std(),
                    'sparsity': (v == 0).mean(),
                    'magnitude': np.linalg.norm(v)
                } for k, v in stats.items()
            })

    def compute_gradient_flow(self):
        """Analyze gradient flow through the network"""
        grad_norms = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norms.append({
                    'name': name,
                    'grad_norm': param.grad.norm().item(),
                    'param_norm': param.norm().item(),
                    'grad_to_param_ratio': param.grad.norm().item() / param.norm().item()
                })
        self.gradient_stats.append(grad_norms)

    def create_visualizations(self, save_dir: Path):
        """Create visualizations for mechanistic analysis"""
        try:
            print(f"Starting visualization creation in directory: {save_dir}")
            save_dir.mkdir(exist_ok=True)

            # Plot attention patterns
            if self.attention_patterns:
                try:
                    plt.figure(figsize=(10, 6))
                    attention_patterns = np.array(self.attention_patterns)
                    print(f"Debug - Attention patterns shape before averaging: {attention_patterns.shape}")

                    # Average over all dimensions except the last two (which are the 3x3 attention matrix)
                    attention_mean = np.mean(attention_patterns, axis=(0, 1, 2))  # Average over batch, time, and heads
                    print(f"Debug - Attention mean shape after averaging: {attention_mean.shape}")

                    # Create heatmap
                    sns.heatmap(attention_mean, cmap='viridis', annot=True, fmt='.2f', square=True)
                    plt.title('Average Attention Patterns')
                    plt.xlabel('Key Position')
                    plt.ylabel('Query Position')
                    plt.tight_layout()
                    plt.savefig(save_dir / 'attention_patterns.png')
                    plt.close()
                except Exception as e:
                    print(f"Warning: Failed to plot attention patterns: {e}")
                    print(f"Attention patterns shape: {attention_patterns.shape}")

            # Plot activation statistics over time
            if self.activation_stats:
                try:
                    print("Starting activation statistics plotting")
                    plt.figure(figsize=(15, 5))
                    for key in ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']:
                        values = [stats[key]['magnitude'] for stats in self.activation_stats]
                        plt.plot(values, label=key)
                    plt.title('Activation Magnitudes Over Training')
                    plt.xlabel('Training Step')
                    plt.ylabel('Magnitude')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(save_dir / 'activation_magnitudes.png')
                    plt.close()
                    print("Completed activation statistics plotting")
                except Exception as e:
                    print(f"Warning: Failed to plot activation statistics: {e}")

            # Plot gradient statistics
            if self.gradient_stats:
                try:
                    print("Starting gradient statistics plotting")
                    # Convert gradient stats to a more robust format
                    grad_data = []
                    for epoch_stats in self.gradient_stats:
                        epoch_dict = {}
                        for stat in epoch_stats:
                            param_name = stat['name']
                            epoch_dict[param_name] = stat['grad_norm']
                        grad_data.append(epoch_dict)

                    plt.figure(figsize=(15, 5))
                    grad_df = pd.DataFrame(grad_data)
                    if not grad_df.empty:
                        grad_df.plot()
                        plt.title('Gradient Norms Over Training')
                        plt.xlabel('Training Step')
                        plt.ylabel('Gradient Norm')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        plt.savefig(save_dir / 'gradient_norms.png')
                    plt.close()
                    print("Completed gradient statistics plotting")
                except Exception as e:
                    print(f"Warning: Failed to plot gradient statistics: {e}")

        except Exception as e:
            print(f"Warning: Error during visualization creation: {e}")
            print("Continuing with execution despite visualization error")


# ======== Optimization Components ========
class GrokkingOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.base_config = config
        self.optimization_history = []
        self.grokking_indicators = []

    def configure_optimization_strategy(self, strategy: OptimizationStrategy) -> dict:
        """Configure optimization parameters based on strategy"""
        config = self.base_config.copy()

        if strategy.learning_rate_schedule == 'cyclic':
            config['lr_scheduler'] = {
                'type': 'cyclic',
                'base_lr': config['lr'] / 10,
                'max_lr': config['lr'],
                'step_size_up': 2000,
                'cycle_momentum': False
            }
        elif strategy.learning_rate_schedule == 'adaptive':
            config['lr_scheduler'] = {
                'type': 'reduce_on_plateau',
                'factor': 0.5,
                'patience': 1000,
                'min_lr': config['lr'] / 100
            }

        if strategy.weight_decay_schedule == 'increasing':
            config['wd_scheduler'] = {
                'type': 'linear',
                'final_value': config['wd'] * 10,
                'warmup_steps': 5000
            }

        if strategy.batch_size_schedule == 'increasing':
            config['batch_size_schedule'] = {
                'initial_size': config['batch_size'],
                'final_size': config['batch_size'] * 4,
                'rampup_epochs': 5000
            }

        return config

    def create_scheduler(self, optimizer: torch.optim.Optimizer, config: dict):
        """Create learning rate scheduler based on configuration"""
        if config.get('lr_scheduler', {}).get('type') == 'cyclic':
            return torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=config['lr_scheduler']['base_lr'],
                max_lr=config['lr_scheduler']['max_lr'],
                step_size_up=config['lr_scheduler']['step_size_up'],
                cycle_momentum=config['lr_scheduler']['cycle_momentum']
            )
        elif config.get('lr_scheduler', {}).get('type') == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config['lr_scheduler']['factor'],
                patience=config['lr_scheduler']['patience'],
                min_lr=config['lr_scheduler']['min_lr']
            )
        return None

    def detect_early_grokking(self, train_losses: List[float],
                              test_losses: List[float],
                              window_size: int = 100) -> Tuple[bool, float]:
        """Detect early signs of grokking"""
        if len(train_losses) < window_size:
            return False, 0.0

        recent_train = train_losses[-window_size:]
        recent_test = test_losses[-window_size:]

        train_stability = np.std(recent_train)
        test_improvement = (recent_test[0] - recent_test[-1]) / recent_test[0]
        loss_ratio = np.mean(recent_test) / np.mean(recent_train)

        grokking_score = test_improvement * loss_ratio / (train_stability + 1e-8)
        self.grokking_indicators.append(grokking_score)

        return grokking_score > 0.1, grokking_score

    def apply_curriculum_learning(self, epoch: int, data: torch.Tensor,
                                  labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply curriculum learning strategy"""
        if epoch < 1000:
            difficulty_threshold = 0.3
        elif epoch < 5000:
            difficulty_threshold = 0.6
        else:
            difficulty_threshold = 1.0

        difficulties = self.calculate_example_difficulties(data, labels)
        mask = difficulties <= difficulty_threshold
        return data[mask], labels[mask]

    def calculate_example_difficulties(self, data: torch.Tensor,
                                       labels: torch.Tensor) -> torch.Tensor:
        """Calculate difficulty scores for examples"""
        # For modular arithmetic: difficulty based on size of numbers
        max_val = self.base_config['p'] - 1
        input_vals = data[:, :2]  # First two elements are the numbers to add
        normalized_vals = input_vals.float() / max_val
        difficulties = torch.mean(normalized_vals, dim=1)
        return difficulties

    def update_batch_size(self, epoch: int, config: dict) -> int:
        """Update batch size according to schedule"""
        if 'batch_size_schedule' not in config:
            return config['batch_size']

        schedule = config['batch_size_schedule']
        progress = min(epoch / schedule['rampup_epochs'], 1.0)
        new_size = int(schedule['initial_size'] + progress *
                       (schedule['final_size'] - schedule['initial_size']))
        return new_size

    def update_weight_decay(self, epoch: int, config: dict) -> float:
        """Update weight decay according to schedule"""
        if 'wd_scheduler' not in config:
            return config['wd']

        schedule = config['wd_scheduler']
        if schedule['type'] == 'linear':
            progress = min(epoch / schedule['warmup_steps'], 1.0)
            return config['wd'] + progress * (schedule['final_value'] - config['wd'])
        return config['wd']


class EnhancedGrokkingExperiment(BaseGrokkingExperiment):
    def __init__(self, config=None, task_config: TaskConfig = None):
        """Initialize enhanced experiment"""
        super().__init__(config)

        # Initialize task configuration
        self.task_config = task_config or TaskConfig(
            task_type=TaskType.MODULAR_ARITHMETIC,
            input_size=113,
            output_size=113,
            sequence_length=3
        )

        # Initialize analysis components
        self.mechanistic_analyzer = None
        self.grokking_optimizer = None
        self.scheduler = None

        # Additional metrics storage
        self.optimization_metrics = {
            'learning_rates': [],
            'weight_decays': [],
            'batch_sizes': [],
            'grokking_indicators': []
        }

    def initialize_components(self):
        """Initialize analysis and optimization components"""
        self.mechanistic_analyzer = MechanisticAnalysis(self.model, self.config)
        self.grokking_optimizer = GrokkingOptimizer(self.model, self.config)

    def enhanced_train_epoch(self, epoch: int, strategy: OptimizationStrategy):
        """Enhanced training for one epoch"""
        self.model.train()

        # Apply curriculum learning if enabled
        if strategy.curriculum_learning:
            curr_data, curr_labels = self.grokking_optimizer.apply_curriculum_learning(
                epoch, self.train_data, self.train_labels
            )
        else:
            curr_data, curr_labels = self.train_data, self.train_labels

        # Training step with automatic mixed precision
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            train_logits = self.model(curr_data)
            if len(train_logits.shape) == 3:
                train_logits = train_logits[:, -1]
            train_loss = self.loss_fn(train_logits, curr_labels)
            train_acc = self.calculate_accuracy(train_logits, curr_labels)

        # Optimization step
        train_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # Update schedulers if they exist
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

        return train_loss.item(), train_acc

    def run_enhanced_training(self, strategy: OptimizationStrategy):
        """Run enhanced training loop"""
        print("Initializing enhanced training...")
        self.initialize_components()

        # Configure optimization strategy
        enhanced_config = self.grokking_optimizer.configure_optimization_strategy(strategy)

        # Setup scheduler if specified in strategy
        self.scheduler = self.grokking_optimizer.create_scheduler(self.optimizer, enhanced_config)

        print("Starting training loop...")
        pbar = tqdm.tqdm(range(enhanced_config['num_epochs']), desc="Training")

        for epoch in pbar:
            # Training step
            train_loss, train_acc = self.enhanced_train_epoch(epoch, strategy)
            test_loss, test_acc = self.evaluate()

            # Update metrics
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)

            # Mechanistic analysis
            self.mechanistic_analyzer.track_training_step(
                epoch, self.train_data, self.train_labels
            )

            # Check for early grokking
            if strategy.early_grok_detection:
                is_grokking, grokking_score = self.grokking_optimizer.detect_early_grokking(
                    self.train_losses, self.test_losses
                )
                if is_grokking:
                    print(f"\nEarly grokking detected at epoch {epoch} with score {grokking_score:.4f}")

            # Update and track optimization metrics
            self.update_optimization_metrics(epoch, enhanced_config)

            # Update progress bar
            pbar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Test Loss": f"{test_loss:.4f}",
                "Train Acc": f"{train_acc:.4f}",
                "Test Acc": f"{test_acc:.4f}"
            })

            # Checkpointing
            if epoch % self.config['checkpoint_every'] == 0:
                self.save_checkpoint(epoch)

        print("\nTraining complete! Generating final analysis...")
        self.generate_final_analysis()

    def update_optimization_metrics(self, epoch: int, config: dict):
        """Update optimization-related metrics"""
        self.optimization_metrics['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
        self.optimization_metrics['weight_decays'].append(
            self.grokking_optimizer.update_weight_decay(epoch, config)
        )
        self.optimization_metrics['batch_sizes'].append(
            self.grokking_optimizer.update_batch_size(epoch, config)
        )

    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            test_logits = self.model(self.test_data)
            if len(test_logits.shape) == 3:
                test_logits = test_logits[:, -1]
            test_loss = self.loss_fn(test_logits, self.test_labels)
            test_acc = self.calculate_accuracy(test_logits, self.test_labels)
        return test_loss.item(), test_acc

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'config': self.config
        }
        torch.save(checkpoint, self.dirs['checkpoints'] / f'checkpoint_epoch_{epoch}.pt')

    def generate_final_analysis(self):
        """Generate comprehensive final analysis"""
        print("Generating comprehensive analysis...")

        # Create analysis directory
        analysis_dir = self.dirs['analysis']
        analysis_dir.mkdir(exist_ok=True)

        # Generate visualizations
        self.create_training_visualizations(analysis_dir)
        self.mechanistic_analyzer.create_visualizations(analysis_dir / 'mechanistic')

        # Save metrics
        self.save_metrics(analysis_dir)

        print(f"Analysis complete! Results saved in {analysis_dir}")

    def create_training_visualizations(self, save_dir: Path):
        """Create training visualizations"""
        # Training dynamics plot
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.semilogy(self.train_losses, label='Train Loss', alpha=0.8)
        plt.semilogy(self.test_losses, label='Test Loss', alpha=0.8)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.title('Training and Test Loss')

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', alpha=0.8)
        plt.plot(self.test_accuracies, label='Test Accuracy', alpha=0.8)
        plt.grid(True, alpha=0.2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Test Accuracy')

        plt.tight_layout()
        plt.savefig(save_dir / 'training_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_metrics(self, save_dir: Path):
        """Save all metrics to JSON"""
        metrics = {
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'optimization_metrics': self.optimization_metrics
        }

        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
    # ======== Main Execution Script ========


def main():
    # Configuration
    base_config = {
        'num_epochs': 25000,
        'checkpoint_every': 5000,
        'weight_track_every': 5000,
        'batch_size': 2048,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_config': {
            'n_layers': 1,
            'n_heads': 4,
            'd_model': 64,
            'd_head': 16,
            'd_mlp': 256,
            'act_fn': "relu",
            'normalization_type': None
        }
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

    # Create and run experiment
    print("Initializing experiment...")
    experiment = EnhancedGrokkingExperiment(base_config)

    # Enable GPU optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Setup and run training
    print("Setting up experiment...")
    experiment.generate_data()
    experiment.create_model()
    experiment.setup_training()

    print("Starting enhanced training...")
    experiment.run_enhanced_training(strategy)


if __name__ == "__main__":
    main()
