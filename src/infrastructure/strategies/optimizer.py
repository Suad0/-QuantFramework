"""
Strategy parameter optimization using genetic algorithms and other methods.

This module provides sophisticated parameter optimization for trading strategies
using genetic algorithms, grid search, and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ...domain.interfaces import IStrategy
from ...domain.exceptions import OptimizationError, ValidationError
from ...infrastructure.logging.logger import get_logger


@dataclass
class ParameterRange:
    """Defines the range and type for a strategy parameter."""
    name: str
    min_value: float
    max_value: float
    param_type: str = "float"  # "float", "int", "bool", "choice"
    choices: Optional[List[Any]] = None
    step: Optional[float] = None


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_parameters: Dict[str, Any]
    best_fitness: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    convergence_generation: Optional[int]
    execution_time: float
    metadata: Dict[str, Any]


class GeneticAlgorithmOptimizer:
    """
    Genetic algorithm optimizer for strategy parameters.
    
    Implements a sophisticated genetic algorithm with various selection,
    crossover, and mutation strategies for parameter optimization.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        logger=None
    ):
        """
        Initialize genetic algorithm optimizer.
        
        Args:
            population_size: Size of the population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best individuals to preserve
            tournament_size: Size of tournament for selection
            logger: Logger instance
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.logger = logger or get_logger(__name__)
        
        self.optimization_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
    def optimize(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable[[IStrategy], float],
        maximize: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using genetic algorithm.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_ranges: List of parameter ranges to optimize
            fitness_function: Function to evaluate strategy fitness
            maximize: Whether to maximize or minimize fitness
            parallel: Whether to use parallel evaluation
            max_workers: Maximum number of worker threads
            
        Returns:
            Optimization results
            
        Raises:
            OptimizationError: If optimization fails
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(
                f"Starting genetic algorithm optimization",
                population_size=self.population_size,
                generations=self.generations,
                parameters=len(parameter_ranges)
            )
            
            # Initialize population
            population = self._initialize_population(parameter_ranges)
            
            # Evolution loop
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = self._evaluate_population(
                    population, strategy_class, fitness_function, 
                    parameter_ranges, parallel, max_workers
                )
                
                # Update best individual
                generation_best_idx = np.argmax(fitness_scores) if maximize else np.argmin(fitness_scores)
                generation_best_fitness = fitness_scores[generation_best_idx]
                
                if (maximize and generation_best_fitness > self.best_fitness) or \
                   (not maximize and generation_best_fitness < self.best_fitness):
                    self.best_fitness = generation_best_fitness
                    self.best_individual = population[generation_best_idx].copy()
                
                # Record generation statistics
                self._record_generation_stats(generation, fitness_scores, maximize)
                
                # Check convergence
                if self._check_convergence(generation):
                    self.logger.info(f"Converged at generation {generation}")
                    break
                
                # Create next generation
                population = self._create_next_generation(
                    population, fitness_scores, parameter_ranges, maximize
                )
                
                self.logger.info(
                    f"Generation {generation + 1}/{self.generations} completed",
                    best_fitness=self.best_fitness,
                    avg_fitness=np.mean(fitness_scores)
                )
            
            # Prepare results
            execution_time = (datetime.now() - start_time).total_seconds()
            best_parameters = self._decode_individual(self.best_individual, parameter_ranges)
            
            result = OptimizationResult(
                best_parameters=best_parameters,
                best_fitness=self.best_fitness,
                optimization_history=self.optimization_history.copy(),
                total_evaluations=len(self.optimization_history) * self.population_size,
                convergence_generation=self._find_convergence_generation(),
                execution_time=execution_time,
                metadata={
                    "algorithm": "genetic_algorithm",
                    "population_size": self.population_size,
                    "generations": self.generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate
                }
            )
            
            self.logger.info(
                f"Optimization completed",
                best_fitness=self.best_fitness,
                execution_time=execution_time,
                total_evaluations=result.total_evaluations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Genetic algorithm optimization failed: {str(e)}")
            raise OptimizationError(
                f"Genetic algorithm optimization failed: {str(e)}",
                error_code="GA_OPTIMIZATION_FAILED"
            ) from e
    
    def _initialize_population(self, parameter_ranges: List[ParameterRange]) -> List[List[float]]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for param_range in parameter_ranges:
                if param_range.param_type == "choice":
                    # For choice parameters, use index
                    individual.append(random.randint(0, len(param_range.choices) - 1))
                elif param_range.param_type == "bool":
                    individual.append(random.randint(0, 1))
                elif param_range.param_type == "int":
                    individual.append(random.randint(int(param_range.min_value), int(param_range.max_value)))
                else:  # float
                    individual.append(random.uniform(param_range.min_value, param_range.max_value))
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(
        self,
        population: List[List[float]],
        strategy_class: type,
        fitness_function: Callable[[IStrategy], float],
        parameter_ranges: List[ParameterRange],
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> List[float]:
        """Evaluate fitness for entire population."""
        if parallel:
            return self._evaluate_population_parallel(
                population, strategy_class, fitness_function, parameter_ranges, max_workers
            )
        else:
            return self._evaluate_population_sequential(
                population, strategy_class, fitness_function, parameter_ranges
            )
    
    def _evaluate_population_parallel(
        self,
        population: List[List[float]],
        strategy_class: type,
        fitness_function: Callable[[IStrategy], float],
        parameter_ranges: List[ParameterRange],
        max_workers: Optional[int] = None
    ) -> List[float]:
        """Evaluate population fitness in parallel."""
        fitness_scores = [0.0] * len(population)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluations
            future_to_index = {
                executor.submit(
                    self._evaluate_individual,
                    individual, strategy_class, fitness_function, parameter_ranges
                ): i
                for i, individual in enumerate(population)
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    fitness_scores[index] = future.result()
                except Exception as e:
                    self.logger.warning(f"Individual evaluation failed: {str(e)}")
                    fitness_scores[index] = float('-inf')  # Penalize failed evaluations
        
        return fitness_scores
    
    def _evaluate_population_sequential(
        self,
        population: List[List[float]],
        strategy_class: type,
        fitness_function: Callable[[IStrategy], float],
        parameter_ranges: List[ParameterRange]
    ) -> List[float]:
        """Evaluate population fitness sequentially."""
        fitness_scores = []
        
        for individual in population:
            try:
                fitness = self._evaluate_individual(individual, strategy_class, fitness_function, parameter_ranges)
                fitness_scores.append(fitness)
            except Exception as e:
                self.logger.warning(f"Individual evaluation failed: {str(e)}")
                fitness_scores.append(float('-inf'))
        
        return fitness_scores
    
    def _evaluate_individual(
        self,
        individual: List[float],
        strategy_class: type,
        fitness_function: Callable[[IStrategy], float],
        parameter_ranges: List[ParameterRange]
    ) -> float:
        """Evaluate fitness for a single individual."""
        try:
            # Decode individual to parameters
            parameters = self._decode_individual(individual, parameter_ranges)
            
            # Create strategy instance with decoded parameters
            strategy = strategy_class(**parameters)
            
            # Evaluate fitness
            fitness = fitness_function(strategy)
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Individual evaluation error: {str(e)}")
            return float('-inf')
    
    def _decode_individual(
        self,
        individual: List[float],
        parameter_ranges: List[ParameterRange]
    ) -> Dict[str, Any]:
        """Decode individual genes to parameter values."""
        parameters = {}
        
        for i, param_range in enumerate(parameter_ranges):
            gene_value = individual[i]
            
            if param_range.param_type == "choice":
                parameters[param_range.name] = param_range.choices[int(gene_value)]
            elif param_range.param_type == "bool":
                parameters[param_range.name] = bool(int(gene_value))
            elif param_range.param_type == "int":
                parameters[param_range.name] = int(gene_value)
            else:  # float
                parameters[param_range.name] = float(gene_value)
        
        return parameters
    
    def _create_next_generation(
        self,
        population: List[List[float]],
        fitness_scores: List[float],
        parameter_ranges: List[ParameterRange],
        maximize: bool
    ) -> List[List[float]]:
        """Create next generation through selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism - preserve best individuals
        elite_count = int(self.population_size * self.elitism_rate)
        if elite_count > 0:
            elite_indices = np.argsort(fitness_scores)
            if maximize:
                elite_indices = elite_indices[-elite_count:]
            else:
                elite_indices = elite_indices[:elite_count]
            
            for idx in elite_indices:
                next_generation.append(population[idx].copy())
        
        # Generate remaining individuals
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population, fitness_scores, maximize)
            parent2 = self._tournament_selection(population, fitness_scores, maximize)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, parameter_ranges)
            child2 = self._mutate(child2, parameter_ranges)
            
            next_generation.extend([child1, child2])
        
        # Trim to exact population size
        return next_generation[:self.population_size]
    
    def _tournament_selection(
        self,
        population: List[List[float]],
        fitness_scores: List[float],
        maximize: bool
    ) -> List[float]:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        if maximize:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[winner_idx].copy()
    
    def _crossover(
        self,
        parent1: List[float],
        parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Perform crossover between two parents."""
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(
        self,
        individual: List[float],
        parameter_ranges: List[ParameterRange]
    ) -> List[float]:
        """Mutate individual genes."""
        mutated = individual.copy()
        
        for i, param_range in enumerate(parameter_ranges):
            if random.random() < self.mutation_rate:
                if param_range.param_type == "choice":
                    mutated[i] = random.randint(0, len(param_range.choices) - 1)
                elif param_range.param_type == "bool":
                    mutated[i] = 1 - mutated[i]  # Flip boolean
                elif param_range.param_type == "int":
                    # Gaussian mutation with bounds
                    mutation_strength = (param_range.max_value - param_range.min_value) * 0.1
                    new_value = mutated[i] + random.gauss(0, mutation_strength)
                    mutated[i] = max(param_range.min_value, 
                                   min(param_range.max_value, int(new_value)))
                else:  # float
                    # Gaussian mutation with bounds
                    mutation_strength = (param_range.max_value - param_range.min_value) * 0.1
                    new_value = mutated[i] + random.gauss(0, mutation_strength)
                    mutated[i] = max(param_range.min_value, 
                                   min(param_range.max_value, new_value))
        
        return mutated
    
    def _record_generation_stats(
        self,
        generation: int,
        fitness_scores: List[float],
        maximize: bool
    ) -> None:
        """Record statistics for current generation."""
        stats = {
            "generation": generation,
            "best_fitness": max(fitness_scores) if maximize else min(fitness_scores),
            "worst_fitness": min(fitness_scores) if maximize else max(fitness_scores),
            "avg_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(stats)
    
    def _check_convergence(self, generation: int, patience: int = 10) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < patience:
            return False
        
        # Check if best fitness hasn't improved in last 'patience' generations
        recent_best = [gen["best_fitness"] for gen in self.optimization_history[-patience:]]
        return len(set(recent_best)) == 1  # All values are the same
    
    def _find_convergence_generation(self) -> Optional[int]:
        """Find the generation where convergence occurred."""
        if len(self.optimization_history) < 2:
            return None
        
        best_fitness = self.optimization_history[-1]["best_fitness"]
        
        for i in range(len(self.optimization_history) - 1, -1, -1):
            if self.optimization_history[i]["best_fitness"] != best_fitness:
                return i + 1
        
        return 0


class StrategyOptimizer:
    """
    Main strategy optimizer that supports multiple optimization algorithms.
    
    Provides a unified interface for strategy parameter optimization using
    various algorithms including genetic algorithms, grid search, and Bayesian optimization.
    """
    
    def __init__(self, logger=None):
        """
        Initialize strategy optimizer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.optimization_results = []
        
    def optimize_strategy(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable[[IStrategy], float],
        method: str = "genetic_algorithm",
        maximize: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize strategy parameters using specified method.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_ranges: List of parameter ranges
            fitness_function: Function to evaluate strategy fitness
            method: Optimization method ("genetic_algorithm", "grid_search", "bayesian")
            maximize: Whether to maximize or minimize fitness
            **kwargs: Additional parameters for optimization method
            
        Returns:
            Optimization results
            
        Raises:
            OptimizationError: If optimization fails
            ValidationError: If inputs are invalid
        """
        try:
            self.logger.info(
                f"Starting strategy optimization",
                method=method,
                strategy_class=strategy_class.__name__,
                parameters=len(parameter_ranges)
            )
            
            # Validate inputs
            self._validate_optimization_inputs(
                strategy_class, parameter_ranges, fitness_function
            )
            
            # Run optimization based on method
            if method == "genetic_algorithm":
                result = self._optimize_genetic_algorithm(
                    strategy_class, parameter_ranges, fitness_function, maximize, **kwargs
                )
            elif method == "grid_search":
                result = self._optimize_grid_search(
                    strategy_class, parameter_ranges, fitness_function, maximize, **kwargs
                )
            elif method == "bayesian":
                result = self._optimize_bayesian(
                    strategy_class, parameter_ranges, fitness_function, maximize, **kwargs
                )
            else:
                raise ValidationError(f"Unsupported optimization method: {method}")
            
            # Store result
            self.optimization_results.append(result)
            
            self.logger.info(
                f"Strategy optimization completed",
                method=method,
                best_fitness=result.best_fitness,
                total_evaluations=result.total_evaluations
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Strategy optimization failed: {str(e)}")
            raise OptimizationError(
                f"Strategy optimization failed: {str(e)}",
                error_code="OPTIMIZATION_FAILED"
            ) from e
    
    def _validate_optimization_inputs(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable
    ) -> None:
        """Validate optimization inputs."""
        if not parameter_ranges:
            raise ValidationError("Parameter ranges cannot be empty")
        
        if not callable(fitness_function):
            raise ValidationError("Fitness function must be callable")
        
        # Validate parameter ranges
        for param_range in parameter_ranges:
            if param_range.param_type not in ["float", "int", "bool", "choice"]:
                raise ValidationError(f"Invalid parameter type: {param_range.param_type}")
            
            if param_range.param_type == "choice" and not param_range.choices:
                raise ValidationError(f"Choice parameter {param_range.name} must have choices")
            
            if param_range.param_type in ["float", "int"]:
                if param_range.min_value >= param_range.max_value:
                    raise ValidationError(
                        f"Invalid range for {param_range.name}: "
                        f"min ({param_range.min_value}) >= max ({param_range.max_value})"
                    )
    
    def _optimize_genetic_algorithm(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable[[IStrategy], float],
        maximize: bool,
        **kwargs
    ) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        ga_optimizer = GeneticAlgorithmOptimizer(
            population_size=kwargs.get("population_size", 50),
            generations=kwargs.get("generations", 100),
            mutation_rate=kwargs.get("mutation_rate", 0.1),
            crossover_rate=kwargs.get("crossover_rate", 0.8),
            elitism_rate=kwargs.get("elitism_rate", 0.1),
            tournament_size=kwargs.get("tournament_size", 3),
            logger=self.logger
        )
        
        return ga_optimizer.optimize(
            strategy_class=strategy_class,
            parameter_ranges=parameter_ranges,
            fitness_function=fitness_function,
            maximize=maximize,
            parallel=kwargs.get("parallel", True),
            max_workers=kwargs.get("max_workers", None)
        )
    
    def _optimize_grid_search(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable[[IStrategy], float],
        maximize: bool,
        **kwargs
    ) -> OptimizationResult:
        """Optimize using grid search (placeholder implementation)."""
        # This would implement a comprehensive grid search
        # For now, returning a placeholder result
        self.logger.warning("Grid search optimization not fully implemented")
        
        return OptimizationResult(
            best_parameters={},
            best_fitness=0.0,
            optimization_history=[],
            total_evaluations=0,
            convergence_generation=None,
            execution_time=0.0,
            metadata={"algorithm": "grid_search", "status": "not_implemented"}
        )
    
    def _optimize_bayesian(
        self,
        strategy_class: type,
        parameter_ranges: List[ParameterRange],
        fitness_function: Callable[[IStrategy], float],
        maximize: bool,
        **kwargs
    ) -> OptimizationResult:
        """Optimize using Bayesian optimization (placeholder implementation)."""
        # This would implement Bayesian optimization using libraries like scikit-optimize
        # For now, returning a placeholder result
        self.logger.warning("Bayesian optimization not fully implemented")
        
        return OptimizationResult(
            best_parameters={},
            best_fitness=0.0,
            optimization_history=[],
            total_evaluations=0,
            convergence_generation=None,
            execution_time=0.0,
            metadata={"algorithm": "bayesian", "status": "not_implemented"}
        )
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get history of all optimization runs."""
        return self.optimization_results.copy()
    
    def export_results(self, file_path: str) -> None:
        """Export optimization results to file."""
        try:
            export_data = {
                "optimization_results": [
                    {
                        "best_parameters": result.best_parameters,
                        "best_fitness": result.best_fitness,
                        "total_evaluations": result.total_evaluations,
                        "execution_time": result.execution_time,
                        "metadata": result.metadata
                    }
                    for result in self.optimization_results
                ],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported optimization results to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {str(e)}")
            raise OptimizationError(
                f"Results export failed: {str(e)}",
                error_code="EXPORT_FAILED"
            ) from e