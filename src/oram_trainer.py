"""
ORAM-integrated trainer for PyTorch CIFAR-10 training.

Trains ResNet-18 with ORAM-backed data loading to measure
the overhead of oblivious data access patterns.
"""

import os
import time
from typing import Dict, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm

from .oram_storage import ORAMStorage, load_cifar10_to_oram, BLOCK_SIZE
from .oram_dataloader import create_oram_dataloader, get_cifar10_transforms
from .profiler import Profiler, ProfilerContext, get_profiler


class ORAMTrainer:
    """
    ORAM-integrated trainer for ResNet-18 on CIFAR-10.
    
    Uses ORAM-backed data loading to hide access patterns,
    with comprehensive profiling of overhead sources.
    """
    
    def __init__(
        self,
        data_dir: str = './data',
        oram_storage_path: Optional[str] = None,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        device: Optional[str] = None,
        num_samples: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.oram_storage_path = oram_storage_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.profiler = get_profiler()
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # ORAM components
        self.oram_storage = None
        self.train_loader = None
        self.test_loader = None
        self.num_train_samples = num_samples if num_samples is not None else 50000
        self.num_test_samples = 10000
        
    def setup(self):
        """Initialize ORAM storage, model, and training components."""
        print("Setting up ORAM-integrated training...")
        
        with self.profiler.track('setup'):
            # Initialize ORAM storage for training data
            print(f"Initializing ORAM storage for {self.num_train_samples} samples...")
            self.oram_storage = ORAMStorage(
                num_samples=self.num_train_samples,
                storage_path=self.oram_storage_path,
            )
            
            # Load CIFAR-10 into ORAM (with optional sample limit)
            limit = self.num_train_samples if self.num_train_samples < 50000 else None
            print(f"Loading CIFAR-10 training data into ORAM ({self.num_train_samples} samples)...")
            load_cifar10_to_oram(
                self.oram_storage,
                data_dir=self.data_dir,
                train=True,
                progress=True,
                limit=limit
            )
            
            # Create ORAM-backed training data loader
            # Note: num_workers must be 0 for ORAM
            self.train_loader = create_oram_dataloader(
                oram_storage=self.oram_storage,
                num_samples=self.num_train_samples,
                batch_size=self.batch_size,
                shuffle=True,
                train=True
            )
            
            # For testing, we use standard loader (test set access pattern
            # leakage is less critical, and allows faster evaluation)
            # In a full deployment, test loader would also be ORAM-backed
            self._setup_test_loader()
            
            # Create model (ResNet-18 adapted for CIFAR-10)
            self._setup_model()
            
        print(f"Setup complete. Device: {self.device}")
        print(f"ORAM storage stats: {self.oram_storage.get_stats()}")
        
    def _setup_test_loader(self):
        """Set up standard test data loader."""
        import torchvision.transforms as transforms
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ),
        ])
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=test_transform
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    def _setup_model(self):
        """Initialize ResNet-18 model and optimizer."""
        self.model = torchvision.models.resnet18(weights=None)
        # Modify first conv layer for CIFAR-10 (32x32 images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
        # Modify final FC for 10 classes
        self.model.fc = nn.Linear(512, 10)
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        # Learning rate schedule: decay at epochs 50, 75, 90
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[50, 75, 90],
            gamma=0.1
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with ORAM-backed data loading.
        
        Each batch access goes through ORAM, incurring oblivious
        access overhead that we measure and decompose.
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        self.profiler.start_epoch(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start = time.perf_counter()
            
            # Data is already loaded via ORAM in the DataLoader
            # Move to device
            with self.profiler.track('transfer'):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            with self.profiler.track('compute'):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            with self.profiler.track('compute'):
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Record batch timing
            batch_time = time.perf_counter() - batch_start
            self.profiler.record_time('batch', batch_time)
            
            # Record batch metrics
            self.profiler.record_batch(batch_idx, {
                'loss': loss.item(),
                'batch_time': batch_time
            })
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # Step scheduler
        self.scheduler.step()
        
        metrics = {
            'train_loss': running_loss / len(self.train_loader),
            'train_acc': 100. * correct / total,
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        self.profiler.end_epoch(epoch, metrics)
        self.profiler.record_memory()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set (using standard loader)."""
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'test_loss': test_loss / len(self.test_loader),
            'test_acc': 100. * correct / total
        }
    
    def train(
        self,
        num_epochs: int = 100,
        eval_every: int = 10,
        save_dir: str = 'results/oram'
    ) -> Dict:
        """
        Run full ORAM-integrated training loop.
        
        Args:
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_dir: Directory to save results
            
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        
        history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
            'oram_config': self.oram_storage.get_stats()
        }
        
        best_acc = 0.0
        total_start = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train (with ORAM data loading)
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate periodically
            if epoch % eval_every == 0 or epoch == num_epochs:
                test_metrics = self.evaluate()
            else:
                test_metrics = {'test_loss': 0, 'test_acc': 0}
            
            epoch_time = time.time() - epoch_start
            self.profiler.record_time('epoch', epoch_time)
            
            # Record history
            history['epochs'].append(epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['test_loss'].append(test_metrics['test_loss'])
            history['test_acc'].append(test_metrics['test_acc'])
            history['lr'].append(train_metrics['lr'])
            
            print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                  f"train_acc={train_metrics['train_acc']:.2f}%, "
                  f"test_acc={test_metrics['test_acc']:.2f}%, "
                  f"time={epoch_time:.2f}s")
            
            # Save best model
            if test_metrics['test_acc'] > best_acc:
                best_acc = test_metrics['test_acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(save_dir, 'best_model.pth'))
        
        total_time = time.time() - total_start
        
        # Save final results
        history['total_time'] = total_time
        history['best_acc'] = best_acc
        
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nORAM Training complete. Total time: {total_time:.2f}s")
        print(f"Best test accuracy: {best_acc:.2f}%")
        
        return history
    
    def cleanup(self):
        """Clean up ORAM storage."""
        if self.oram_storage is not None:
            self.oram_storage.close()


def run_oram_training(
    num_epochs: int = 100,
    batch_size: int = 128,
    output_dir: str = 'results/oram',
    device: Optional[str] = None,
    num_samples: Optional[int] = None
) -> Dict:
    """
    Convenience function to run ORAM-integrated training.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        output_dir: Directory for outputs
        device: Device to use (auto-detected if None)
        num_samples: Number of training samples (None = full 50k)
        
    Returns:
        Training history
    """
    with ProfilerContext('oram', output_dir=output_dir) as profiler:
        trainer = ORAMTrainer(
            batch_size=batch_size,
            device=device,
            num_samples=num_samples
        )
        try:
            trainer.setup()
            history = trainer.train(
                num_epochs=num_epochs,
                save_dir=output_dir
            )
        finally:
            trainer.cleanup()
    
    return history
