#!/usr/bin/env python3
"""
Test script to verify the professional project structure is working correctly.
This script tests configuration management, logging, error handling, and clean architecture.
"""

import sys
import os
import tempfile
import asyncio
import importlib.util
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.config.config_manager import initialize_config, get_config_manager
from src.infrastructure.logging.logger import configure_logging, get_logger
from src.infrastructure.container_config import initialize_application
from src.infrastructure.middleware.error_handler import get_error_handler, handle_errors
from src.domain.exceptions import ValidationError, ConfigurationError


def test_configuration_management():
    """Test configuration management system."""
    print("Testing configuration management...")
    
    try:
        # Initialize configuration
        config_manager = initialize_config()
        
        # Test getting configuration values
        app_config = config_manager.get_app_config()
        print(f"✓ Environment: {app_config.environment}")
        print(f"✓ Debug mode: {app_config.debug}")
        print(f"✓ Database host: {app_config.database.host}")
        print(f"✓ Log level: {app_config.logging.level}")
        
        # Test configuration validation
        config_manager.validate()
        print("✓ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_logging_framework():
    """Test logging framework."""
    print("\nTesting logging framework...")
    
    try:
        # Configure logging
        configure_logging({
            'level': 'DEBUG',
            'console_output': True,
            'file_path': 'test_logs/test.log'
        })
        
        # Get logger
        logger = get_logger('test_logger')
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Test structured logging with context
        logger.info("Structured log message", 
                   user_id="test_user", 
                   operation="test_operation",
                   duration=0.123)
        
        print("✓ Logging framework working")
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_error_handling():
    """Test error handling middleware."""
    print("\nTesting error handling...")
    
    try:
        error_handler = get_error_handler()
        
        # Test handling different error types
        try:
            raise ValidationError("Test validation error", error_code="TEST_ERROR")
        except Exception as e:
            error_info = error_handler.handle_error(e, reraise=False)
            print(f"✓ Handled ValidationError: {error_info.get('category', 'unknown')}")
        
        # Test error decorator
        @handle_errors("test_operation", reraise=False)
        def test_function():
            raise ValueError("Test value error")
        
        result = test_function()
        print("✓ Error decorator working")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_dependency_injection():
    """Test dependency injection container."""
    print("\nTesting dependency injection...")
    
    try:
        # Initialize container
        container = initialize_application()
        
        # Test resolving services
        from src.domain.interfaces import IConfigManager, ILogger
        
        config_manager = container.resolve(IConfigManager)
        logger = container.resolve(ILogger)
        
        print("✓ Successfully resolved IConfigManager")
        print("✓ Successfully resolved ILogger")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency injection test failed: {e}")
        return False


def test_domain_entities():
    """Test domain entities and value objects."""
    print("\nTesting domain entities...")
    
    try:
        from src.domain.entities import Portfolio, Position
        from src.domain.value_objects import Price, Signal, SignalType
        from decimal import Decimal
        from datetime import datetime
        
        # Test creating a position
        position = Position(
            symbol="AAPL",
            quantity=Decimal('100'),
            average_cost=Decimal('150.00'),
            current_price=Decimal('155.00')
        )
        
        print(f"✓ Created position: {position.symbol}")
        print(f"✓ Market value: ${position.market_value}")
        print(f"✓ Unrealized P&L: ${position.unrealized_pnl}")
        
        # Test creating a portfolio
        portfolio = Portfolio(
            id="test_portfolio",
            name="Test Portfolio",
            cash=Decimal('10000')
        )
        
        portfolio.add_position(position)
        weights = portfolio.get_weights()
        
        print(f"✓ Created portfolio: {portfolio.name}")
        print(f"✓ Total value: ${portfolio.total_value}")
        print(f"✓ Position weights: {weights}")
        
        return True
        
    except Exception as e:
        print(f"✗ Domain entities test failed: {e}")
        return False


def test_project_structure():
    """Test that the project follows clean architecture structure."""
    print("\nTesting project structure...")
    
    # Define expected structure
    expected_structure = {
        'src': {
            'domain': ['entities.py', 'value_objects.py', 'interfaces.py', 'exceptions.py'],
            'application': {
                'services': ['data_service.py', 'portfolio_service.py']
            },
            'infrastructure': {
                'config': ['config_manager.py'],
                'logging': ['logger.py'],
                'middleware': ['error_handler.py']
            },
            'presentation': {
                'cli': ['main.py']
            }
        },
        'config': ['base.yaml', 'development.yaml', 'production.yaml']
    }
    
    def check_structure(current_path, expected, level=0):
        """Recursively check directory structure."""
        indent = "  " * level
        all_good = True
        
        for item, content in expected.items():
            item_path = current_path / item
            
            if isinstance(content, list):
                # It's a directory with files
                if not item_path.is_dir():
                    print(f"{indent}✗ Missing directory: {item_path}")
                    all_good = False
                    continue
                
                print(f"{indent}✓ Directory: {item}")
                
                # Check files in directory
                for file in content:
                    file_path = item_path / file
                    if file_path.is_file():
                        print(f"{indent}  ✓ File: {file}")
                    else:
                        print(f"{indent}  ✗ Missing file: {file}")
                        all_good = False
                        
            elif isinstance(content, dict):
                # It's a directory with subdirectories
                if not item_path.is_dir():
                    print(f"{indent}✗ Missing directory: {item_path}")
                    all_good = False
                    continue
                
                print(f"{indent}✓ Directory: {item}")
                all_good &= check_structure(item_path, content, level + 1)
        
        return all_good
    
    try:
        project_root = Path(__file__).parent.parent
        structure_valid = check_structure(project_root, expected_structure)
        
        if structure_valid:
            print("✓ Project structure is valid!")
            return True
        else:
            print("✗ Project structure has issues!")
            return False
            
    except Exception as e:
        print(f"✗ Project structure test failed: {e}")
        return False


def test_clean_architecture_dependencies():
    """Test that clean architecture dependency rules are followed."""
    print("\nTesting clean architecture dependencies...")
    
    try:
        # Domain layer should not import from other layers
        domain_files = [
            'src/domain/entities.py',
            'src/domain/value_objects.py', 
            'src/domain/interfaces.py',
            'src/domain/exceptions.py'
        ]
        
        forbidden_imports = [
            'src.application',
            'src.infrastructure', 
            'src.presentation'
        ]
        
        all_good = True
        
        for file_path in domain_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                for forbidden in forbidden_imports:
                    if forbidden in content:
                        print(f"✗ {file_path} imports from {forbidden} (violates clean architecture)")
                        all_good = False
        
        if all_good:
            print("✓ Clean architecture dependencies are correct!")
        
        return all_good
        
    except Exception as e:
        print(f"✗ Clean architecture test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING PROFESSIONAL PROJECT STRUCTURE")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_clean_architecture_dependencies,
        test_configuration_management,
        test_logging_framework,
        test_error_handling,
        test_dependency_injection,
        test_domain_entities
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Project structure is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the output above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)