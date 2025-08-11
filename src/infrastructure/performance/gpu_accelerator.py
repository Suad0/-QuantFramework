"""
GPU acceleration utilities for ML models using PyTorch/CUDA.

This module provides GPU acceleration capabilities for machine learning models
and computationally intensive operations.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration."""
    device: Optional[str] = None  # 'cuda', 'cpu', or None for auto-detect
    memory_fraction: float = 0.8  # Fraction of GPU memory to use
    mixed_precision: bool = True  # Use automatic mixed precision
    benchmark: bool = True  # Enable cudnn benchmark for consistent input sizes


class GPUAccelerator:
    """
    Handles GPU acceleration for ML models and computations.
    
    Provides utilities for GPU device management, memory optimization,
    and accelerated computations using PyTorch/CUDA.
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        self.config = config or GPUConfig()
        self._setup_device()
        self._setup_optimization()
        
    def _setup_device(self):
        """Setup GPU device and check availability."""
        if self.config.device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.device_name = torch.cuda.get_device_name()
                self.memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"Using CUDA GPU: {self.device_name} ({self.memory_total:.1f}GB)")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.device_name = "Apple Metal Performance Shaders"
                self.memory_total = None  # MPS doesn't expose memory info directly
                logger.info(f"Using MPS: {self.device_name}")
            else:
                self.device = torch.device('cpu')
                self.device_name = "CPU"
                logger.info("No GPU acceleration available, using CPU")
        else:
            self.device = torch.device(self.config.device)
            self.device_name = self.config.device
            
        self.is_cuda = self.device.type == 'cuda'
        self.is_mps = self.device.type == 'mps'
        self.is_gpu = self.is_cuda or self.is_mps
        
    def _setup_optimization(self):
        """Setup optimization settings."""
        if self.is_cuda:
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.memory_fraction)
                
            # Enable benchmark mode for consistent input sizes
            torch.backends.cudnn.benchmark = self.config.benchmark
            
            # Enable mixed precision if supported
            self.use_amp = self.config.mixed_precision and hasattr(torch.cuda, 'amp')
            if self.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()
                logger.info("CUDA Automatic mixed precision enabled")
        elif self.is_mps:
            # MPS-specific optimizations
            # Note: MPS doesn't support all CUDA features yet
            self.use_amp = False  # MPS doesn't support AMP yet
            logger.info("MPS optimizations enabled (AMP not supported)")
        else:
            self.use_amp = False
    
    def to_device(self, tensor_or_model: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """Move tensor or model to configured device."""
        return tensor_or_model.to(self.device)
    
    def from_numpy(self, array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert numpy array to tensor on device."""
        tensor = torch.from_numpy(array).type(dtype)
        return self.to_device(tensor)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if self.is_cuda or self.is_mps:
            return tensor.cpu().detach().numpy()
        return tensor.detach().numpy()
    
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.use_amp and self.is_cuda:
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def accelerate_model(self, model: nn.Module) -> nn.Module:
        """
        Accelerate model for GPU training/inference.
        
        Args:
            model: PyTorch model
            
        Returns:
            Accelerated model
        """
        model = self.to_device(model)
        
        if self.is_cuda and torch.cuda.device_count() > 1:
            # Use DataParallel for multiple GPUs
            model = nn.DataParallel(model)
            logger.info(f"Using {torch.cuda.device_count()} CUDA GPUs with DataParallel")
        elif self.is_mps:
            # MPS doesn't support DataParallel yet
            logger.info("Using single MPS device (DataParallel not supported)")
            
        return model
    
    def accelerated_matrix_operations(
        self,
        operation: str,
        *tensors: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform accelerated matrix operations.
        
        Args:
            operation: Operation name ('matmul', 'solve', 'svd', 'eig', etc.)
            *tensors: Input tensors
            **kwargs: Additional arguments
            
        Returns:
            Result tensor
        """
        # Move tensors to device
        device_tensors = [self.to_device(t) for t in tensors]
        
        with self.autocast_context():
            if operation == 'matmul':
                return torch.matmul(device_tensors[0], device_tensors[1])
            elif operation == 'solve':
                return torch.linalg.solve(device_tensors[0], device_tensors[1])
            elif operation == 'svd':
                return torch.linalg.svd(device_tensors[0], **kwargs)
            elif operation == 'eig':
                return torch.linalg.eig(device_tensors[0])
            elif operation == 'inv':
                return torch.linalg.inv(device_tensors[0])
            elif operation == 'cholesky':
                return torch.linalg.cholesky(device_tensors[0])
            else:
                raise ValueError(f"Unsupported operation: {operation}")
    
    def accelerated_optimization(
        self,
        objective_func: callable,
        initial_params: torch.Tensor,
        optimizer_class: type = torch.optim.Adam,
        lr: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        **optimizer_kwargs
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        GPU-accelerated optimization.
        
        Args:
            objective_func: Function to minimize
            initial_params: Initial parameter values
            optimizer_class: PyTorch optimizer class
            lr: Learning rate
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Tuple of (optimal_params, loss_history)
        """
        params = self.to_device(initial_params.clone().detach().requires_grad_(True))
        optimizer = optimizer_class([params], lr=lr, **optimizer_kwargs)
        
        loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            with self.autocast_context():
                loss = objective_func(params)
                
            if self.use_amp and self.is_cuda:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Check convergence
            if abs(prev_loss - current_loss) < tolerance:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
            prev_loss = current_loss
            
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: loss = {current_loss:.6f}")
        
        return params.detach(), loss_history
    
    def accelerated_monte_carlo(
        self,
        simulation_func: callable,
        n_simulations: int,
        batch_size: int = 10000,
        **kwargs
    ) -> torch.Tensor:
        """
        GPU-accelerated Monte Carlo simulations.
        
        Args:
            simulation_func: Function that generates random samples
            n_simulations: Total number of simulations
            batch_size: Batch size for GPU processing
            **kwargs: Arguments for simulation function
            
        Returns:
            Simulation results
        """
        results = []
        n_batches = (n_simulations + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            current_batch_size = min(batch_size, n_simulations - batch_idx * batch_size)
            
            with self.autocast_context():
                # Generate random samples on GPU
                batch_results = simulation_func(current_batch_size, device=self.device, **kwargs)
                
            results.append(batch_results)
            
            if batch_idx % 10 == 0:
                logger.debug(f"Completed batch {batch_idx + 1}/{n_batches}")
        
        return torch.cat(results, dim=0)
    
    def accelerated_covariance_estimation(
        self,
        returns: torch.Tensor,
        method: str = 'sample',
        **kwargs
    ) -> torch.Tensor:
        """
        GPU-accelerated covariance matrix estimation.
        
        Args:
            returns: Return data (n_samples x n_assets)
            method: Estimation method ('sample', 'shrinkage', 'robust')
            **kwargs: Additional arguments
            
        Returns:
            Covariance matrix
        """
        returns = self.to_device(returns)
        
        with self.autocast_context():
            if method == 'sample':
                # Sample covariance
                centered = returns - returns.mean(dim=0, keepdim=True)
                cov_matrix = torch.matmul(centered.T, centered) / (returns.shape[0] - 1)
                
            elif method == 'shrinkage':
                # Ledoit-Wolf shrinkage
                sample_cov = self.accelerated_covariance_estimation(returns, method='sample')
                target = torch.eye(sample_cov.shape[0], device=self.device) * torch.trace(sample_cov) / sample_cov.shape[0]
                
                # Simplified shrinkage intensity
                shrinkage = kwargs.get('shrinkage', 0.1)
                cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
                
            elif method == 'robust':
                # Robust covariance (simplified Huber estimator)
                threshold = kwargs.get('threshold', 2.0)
                centered = returns - returns.median(dim=0, keepdim=True)[0]
                
                # Apply Huber weights
                distances = torch.norm(centered, dim=1)
                weights = torch.where(distances <= threshold, 
                                    torch.ones_like(distances),
                                    threshold / distances)
                
                weighted_centered = centered * weights.unsqueeze(1)
                cov_matrix = torch.matmul(weighted_centered.T, weighted_centered) / weights.sum()
                
            else:
                raise ValueError(f"Unsupported covariance method: {method}")
        
        return cov_matrix
    
    def accelerated_portfolio_optimization(
        self,
        expected_returns: torch.Tensor,
        covariance_matrix: torch.Tensor,
        risk_aversion: float = 1.0,
        constraints: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        GPU-accelerated portfolio optimization.
        
        Args:
            expected_returns: Expected returns vector
            covariance_matrix: Covariance matrix
            risk_aversion: Risk aversion parameter
            constraints: Portfolio constraints
            
        Returns:
            Optimal portfolio weights
        """
        expected_returns = self.to_device(expected_returns)
        covariance_matrix = self.to_device(covariance_matrix)
        
        with self.autocast_context():
            # Solve quadratic optimization problem
            # min: 0.5 * w^T * Σ * w - λ * μ^T * w
            # subject to: 1^T * w = 1 (budget constraint)
            
            n_assets = len(expected_returns)
            ones = torch.ones(n_assets, device=self.device)
            
            # Build KKT system
            A = torch.cat([
                torch.cat([risk_aversion * covariance_matrix, ones.unsqueeze(1)], dim=1),
                torch.cat([ones.unsqueeze(0), torch.zeros(1, 1, device=self.device)], dim=1)
            ], dim=0)
            
            b = torch.cat([expected_returns, torch.ones(1, device=self.device)])
            
            # Solve system
            solution = self.accelerated_matrix_operations('solve', A, b)
            weights = solution[:-1]  # Exclude Lagrange multiplier
            
            # Apply constraints if provided
            if constraints:
                # Simple box constraints
                if 'lower_bounds' in constraints:
                    weights = torch.max(weights, constraints['lower_bounds'])
                if 'upper_bounds' in constraints:
                    weights = torch.min(weights, constraints['upper_bounds'])
                    
                # Renormalize to satisfy budget constraint
                weights = weights / weights.sum()
        
        return weights
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get GPU memory statistics."""
        base_stats = {
            'device': self.device_name,
            'device_type': self.device.type,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        if self.is_cuda:
            base_stats.update({
                'memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'memory_total_gb': self.memory_total,
                'memory_utilization': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory,
                'device_count': torch.cuda.device_count()
            })
        elif self.is_mps:
            # MPS doesn't expose detailed memory stats yet
            base_stats.update({
                'memory_allocated_gb': None,  # Not available for MPS
                'memory_reserved_gb': None,   # Not available for MPS
                'memory_total_gb': None,      # Not available for MPS
                'memory_utilization': None,   # Not available for MPS
                'device_count': 1
            })
        else:
            base_stats.update({
                'memory_allocated_gb': 0,
                'memory_reserved_gb': 0,
                'memory_total_gb': 0,
                'memory_utilization': 0,
                'device_count': 0
            })
            
        return base_stats
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")
        elif self.is_mps:
            # MPS uses torch.mps.empty_cache() if available
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                logger.info("MPS memory cache cleared")
            else:
                logger.info("MPS cache clearing not available in this PyTorch version")


class CUDAKernels:
    """
    Custom CUDA kernels for specialized financial computations.
    """
    
    @staticmethod
    def rolling_statistics(
        data: torch.Tensor,
        window: int,
        operation: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute rolling statistics using GPU acceleration.
        
        Args:
            data: Input time series data
            window: Rolling window size
            operation: Statistical operation ('mean', 'std', 'min', 'max')
            
        Returns:
            Rolling statistics
        """
        if operation == 'mean':
            return data.unfold(0, window, 1).mean(dim=1)
        elif operation == 'std':
            return data.unfold(0, window, 1).std(dim=1)
        elif operation == 'min':
            return data.unfold(0, window, 1).min(dim=1)[0]
        elif operation == 'max':
            return data.unfold(0, window, 1).max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    @staticmethod
    def technical_indicators_batch(
        price_data: torch.Tensor,
        indicators: List[str],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multiple technical indicators in batch on GPU.
        
        Args:
            price_data: OHLCV price data
            indicators: List of indicator names
            **kwargs: Parameters for indicators
            
        Returns:
            Dictionary of indicator results
        """
        results = {}
        
        # Extract price components
        if price_data.shape[1] >= 4:  # OHLC data
            open_prices = price_data[:, 0]
            high_prices = price_data[:, 1]
            low_prices = price_data[:, 2]
            close_prices = price_data[:, 3]
        else:
            close_prices = price_data[:, 0]
            
        for indicator in indicators:
            if indicator == 'sma':
                window = kwargs.get('sma_window', 20)
                results['sma'] = CUDAKernels.rolling_statistics(close_prices, window, 'mean')
                
            elif indicator == 'rsi':
                window = kwargs.get('rsi_window', 14)
                delta = close_prices[1:] - close_prices[:-1]
                gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
                loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
                
                avg_gain = CUDAKernels.rolling_statistics(gain, window, 'mean')
                avg_loss = CUDAKernels.rolling_statistics(loss, window, 'mean')
                
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                results['rsi'] = rsi
                
            elif indicator == 'bollinger_bands':
                window = kwargs.get('bb_window', 20)
                std_mult = kwargs.get('bb_std', 2.0)
                
                sma = CUDAKernels.rolling_statistics(close_prices, window, 'mean')
                std = CUDAKernels.rolling_statistics(close_prices, window, 'std')
                
                results['bb_upper'] = sma + std_mult * std
                results['bb_lower'] = sma - std_mult * std
                results['bb_middle'] = sma
        
        return results