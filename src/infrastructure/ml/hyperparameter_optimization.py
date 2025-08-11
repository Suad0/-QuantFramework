"""
Hyperparameter optimization framework for ML models.

This module provides comprehensive hyperparameter optimization capabilities
including grid search, random search, Bayesian optimization, and genetic algorithms.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
import warnings
from datetime import datetime
from abc import ABC, abstractmethod

from src.domain.exceptions import ValidationError


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(
        self,
        cv_folds: int = 5,
        scoring: str = 'neg_mean_squared_error',
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        """
        Initialize base optimizer.
        
        Args:
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Verbosity level
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.best_params_ = None
        self.best_score_ = None
        self.optimization_history_ = []
    
    @abstractmethod
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters.
        
        Args:
            model: Base model to optimize
            param_space: Parameter search space
            X: Feature data
            y: Target data
            
        Returns:
            Tuple of (best_params, best_score)
        """
        pass


class GridSearchOptimizer(BaseOptimizer):
    """Grid search hyperparameter optimizer."""
    
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize using grid search."""
        search = GridSearchCV(
            model,
            param_space,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        # Store optimization history
        self.optimization_history_ = [
            {
                'params': params,
                'score': score,
                'std': std
            }
            for params, score, std in zip(
                search.cv_results_['params'],
                search.cv_results_['mean_test_score'],
                search.cv_results_['std_test_score']
            )
        ]
        
        return self.best_params_, self.best_score_


class RandomSearchOptimizer(BaseOptimizer):
    """Random search hyperparameter optimizer."""
    
    def __init__(self, n_iter: int = 100, **kwargs):
        """
        Initialize random search optimizer.
        
        Args:
            n_iter: Number of parameter settings sampled
            **kwargs: Base optimizer arguments
        """
        super().__init__(**kwargs)
        self.n_iter = n_iter
    
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize using random search."""
        search = RandomizedSearchCV(
            model,
            param_space,
            n_iter=self.n_iter,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        search.fit(X, y)
        
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        
        # Store optimization history
        self.optimization_history_ = [
            {
                'params': params,
                'score': score,
                'std': std
            }
            for params, score, std in zip(
                search.cv_results_['params'],
                search.cv_results_['mean_test_score'],
                search.cv_results_['std_test_score']
            )
        ]
        
        return self.best_params_, self.best_score_


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(
        self,
        n_calls: int = 100,
        n_initial_points: int = 10,
        acquisition_function: str = 'gp_hedge',
        **kwargs
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            n_calls: Number of calls to objective function
            n_initial_points: Number of initial random points
            acquisition_function: Acquisition function to use
            **kwargs: Base optimizer arguments
        """
        super().__init__(**kwargs)
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        
        # Try to import scikit-optimize
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            self.skopt_available = True
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.Integer = Integer
            self.Categorical = Categorical
            self.use_named_args = use_named_args
        except ImportError:
            self.skopt_available = False
            warnings.warn("scikit-optimize not available. Install with: pip install scikit-optimize")
    
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize using Bayesian optimization."""
        if not self.skopt_available:
            warnings.warn("Falling back to random search")
            fallback = RandomSearchOptimizer(
                n_iter=self.n_calls,
                cv_folds=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
            return fallback.optimize(model, param_space, X, y)
        
        # Convert parameter space to skopt format
        dimensions = []
        param_names = []
        
        for param_name, param_values in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    # Numeric range
                    if all(isinstance(v, int) for v in param_values):
                        dimensions.append(self.Integer(min(param_values), max(param_values)))
                    else:
                        dimensions.append(self.Real(min(param_values), max(param_values)))
                else:
                    # Categorical
                    dimensions.append(self.Categorical(param_values))
            else:
                raise ValueError(f"Unsupported parameter type for {param_name}")
        
        # Define objective function
        @self.use_named_args(dimensions)
        def objective(**params):
            model_copy = type(model)(**model.get_params())
            model_copy.set_params(**params)
            
            scores = cross_val_score(
                model_copy, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=1  # Avoid nested parallelism
            )
            
            # Return negative score for minimization
            return -np.mean(scores)
        
        # Run optimization
        result = self.gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            random_state=self.random_state
        )
        
        # Extract best parameters
        self.best_params_ = {name: value for name, value in zip(param_names, result.x)}
        self.best_score_ = -result.fun  # Convert back to positive
        
        # Store optimization history
        self.optimization_history_ = [
            {
                'params': {name: value for name, value in zip(param_names, x)},
                'score': -y,  # Convert back to positive
                'iteration': i
            }
            for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals))
        ]
        
        return self.best_params_, self.best_score_


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """Genetic algorithm hyperparameter optimizer."""
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        **kwargs
    ):
        """
        Initialize genetic algorithm optimizer.
        
        Args:
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            tournament_size: Tournament selection size
            **kwargs: Base optimizer arguments
        """
        super().__init__(**kwargs)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        # Try to import DEAP
        try:
            from deap import base, creator, tools, algorithms
            self.deap_available = True
            self.deap_base = base
            self.deap_creator = creator
            self.deap_tools = tools
            self.deap_algorithms = algorithms
        except ImportError:
            self.deap_available = False
            warnings.warn("DEAP not available. Install with: pip install deap")
    
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize using genetic algorithm."""
        if not self.deap_available:
            warnings.warn("Falling back to random search")
            fallback = RandomSearchOptimizer(
                n_iter=self.population_size * self.generations // 10,
                cv_folds=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose
            )
            return fallback.optimize(model, param_space, X, y)
        
        # Set up DEAP
        if not hasattr(self.deap_creator, "FitnessMax"):
            self.deap_creator.create("FitnessMax", self.deap_base.Fitness, weights=(1.0,))
        if not hasattr(self.deap_creator, "Individual"):
            self.deap_creator.create("Individual", list, fitness=self.deap_creator.FitnessMax)
        
        toolbox = self.deap_base.Toolbox()
        
        # Convert parameter space to genetic representation
        param_names = list(param_space.keys())
        param_bounds = []
        
        for param_name in param_names:
            param_values = param_space[param_name]
            if isinstance(param_values, list):
                param_bounds.append((0, len(param_values) - 1))
            else:
                raise ValueError(f"Unsupported parameter type for {param_name}")
        
        # Define individual creation
        def create_individual():
            return [np.random.randint(low, high + 1) for low, high in param_bounds]
        
        toolbox.register("individual", self.deap_tools.initIterate, 
                        self.deap_creator.Individual, create_individual)
        toolbox.register("population", self.deap_tools.initRepeat, 
                        list, toolbox.individual)
        
        # Define evaluation function
        def evaluate(individual):
            params = {}
            for i, param_name in enumerate(param_names):
                param_values = param_space[param_name]
                params[param_name] = param_values[individual[i]]
            
            model_copy = type(model)(**model.get_params())
            model_copy.set_params(**params)
            
            scores = cross_val_score(
                model_copy, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=1
            )
            
            return (np.mean(scores),)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", self.deap_tools.cxTwoPoint)
        toolbox.register("mutate", self.deap_tools.mutUniformInt, 
                        low=[low for low, _ in param_bounds],
                        up=[high for _, high in param_bounds],
                        indpb=self.mutation_rate)
        toolbox.register("select", self.deap_tools.selTournament, 
                        tournsize=self.tournament_size)
        
        # Run genetic algorithm
        population = toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        for generation in range(self.generations):
            # Selection
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if np.random.random() < self.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
        
        # Find best individual
        best_individual = self.deap_tools.selBest(population, 1)[0]
        
        # Convert back to parameters
        self.best_params_ = {}
        for i, param_name in enumerate(param_names):
            param_values = param_space[param_name]
            self.best_params_[param_name] = param_values[best_individual[i]]
        
        self.best_score_ = best_individual.fitness.values[0]
        
        return self.best_params_, self.best_score_


class HyperparameterOptimizer:
    """
    Main hyperparameter optimization class that coordinates different optimizers.
    """
    
    def __init__(self):
        """Initialize hyperparameter optimizer."""
        self.optimizers = {
            'grid': GridSearchOptimizer,
            'random': RandomSearchOptimizer,
            'bayesian': BayesianOptimizer,
            'genetic': GeneticAlgorithmOptimizer
        }
    
    def optimize(
        self,
        model: BaseEstimator,
        param_space: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'random',
        **optimizer_kwargs
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """
        Optimize hyperparameters using specified method.
        
        Args:
            model: Base model to optimize
            param_space: Parameter search space
            X: Feature data
            y: Target data
            method: Optimization method
            **optimizer_kwargs: Additional optimizer arguments
            
        Returns:
            Tuple of (best_params, best_score, optimization_info)
        """
        if method not in self.optimizers:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        optimizer_class = self.optimizers[method]
        optimizer = optimizer_class(**optimizer_kwargs)
        
        start_time = datetime.now()
        best_params, best_score = optimizer.optimize(model, param_space, X, y)
        end_time = datetime.now()
        
        optimization_info = {
            'method': method,
            'optimization_time': (end_time - start_time).total_seconds(),
            'best_score': best_score,
            'optimization_history': optimizer.optimization_history_
        }
        
        return best_params, best_score, optimization_info
    
    def get_default_param_space(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameter space for common model types.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default parameter space
        """
        param_spaces = {
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0.0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.7]
            },
            'svm': {
                'C': [0.1, 1, 10, 100, 1000],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['linear', 'rbf', 'poly']
            },
            'lstm': {
                'lstm_units': [32, 50, 64, 100, 128],
                'dropout_rate': [0.1, 0.2, 0.3, 0.4],
                'learning_rate': [0.001, 0.005, 0.01, 0.05],
                'batch_size': [16, 32, 64, 128],
                'sequence_length': [30, 60, 90, 120]
            }
        }
        
        return param_spaces.get(model_type, {})


class RegularizationManager:
    """
    Manager for regularization techniques to prevent overfitting.
    """
    
    def __init__(self):
        """Initialize regularization manager."""
        self.techniques = {
            'early_stopping': self._apply_early_stopping,
            'dropout': self._apply_dropout,
            'l1_l2': self._apply_l1_l2_regularization,
            'batch_normalization': self._apply_batch_normalization,
            'data_augmentation': self._apply_data_augmentation
        }
    
    def apply_regularization(
        self,
        model: BaseEstimator,
        techniques: List[str],
        **kwargs
    ) -> BaseEstimator:
        """
        Apply regularization techniques to model.
        
        Args:
            model: Base model
            techniques: List of regularization techniques
            **kwargs: Additional arguments for techniques
            
        Returns:
            Regularized model
        """
        regularized_model = model
        
        for technique in techniques:
            if technique in self.techniques:
                regularized_model = self.techniques[technique](regularized_model, **kwargs)
            else:
                warnings.warn(f"Unknown regularization technique: {technique}")
        
        return regularized_model
    
    def _apply_early_stopping(self, model: BaseEstimator, **kwargs) -> BaseEstimator:
        """Apply early stopping regularization."""
        # This would be implemented based on the specific model type
        return model
    
    def _apply_dropout(self, model: BaseEstimator, **kwargs) -> BaseEstimator:
        """Apply dropout regularization."""
        # This would be implemented for neural network models
        return model
    
    def _apply_l1_l2_regularization(self, model: BaseEstimator, **kwargs) -> BaseEstimator:
        """Apply L1/L2 regularization."""
        # This would be implemented based on the specific model type
        return model
    
    def _apply_batch_normalization(self, model: BaseEstimator, **kwargs) -> BaseEstimator:
        """Apply batch normalization."""
        # This would be implemented for neural network models
        return model
    
    def _apply_data_augmentation(self, model: BaseEstimator, **kwargs) -> BaseEstimator:
        """Apply data augmentation."""
        # This would be implemented based on the data type
        return model