"""
Main CLI entry point for the quantitative framework.
Provides command-line interface for framework operations.
"""

import argparse
import asyncio
import sys
from typing import Optional
from pathlib import Path

from ...infrastructure.config.config_manager import get_config_manager, initialize_config
from ...infrastructure.logging.logger import get_logger, configure_logging
from ...infrastructure.container_config import initialize_application
from ...domain.exceptions import QuantFrameworkError


class QuantFrameworkCLI:
    """Main CLI application class."""
    
    def __init__(self):
        """Initialize CLI application."""
        self.config_manager = None
        self.logger = None
        self.container = None
    
    def initialize(self, config_dir: Optional[str] = None) -> None:
        """Initialize the application components."""
        try:
            # Initialize configuration
            self.config_manager = initialize_config(config_dir)
            
            # Configure logging
            logging_config = self.config_manager.get_app_config().logging
            configure_logging({
                'level': logging_config.level,
                'format': logging_config.format,
                'file_path': logging_config.file_path,
                'structured_file_path': getattr(logging_config, 'structured_file_path', None),
                'max_file_size': logging_config.max_file_size,
                'backup_count': logging_config.backup_count,
                'console_output': logging_config.console_output
            })
            
            # Get logger
            self.logger = get_logger(__name__)
            
            # Initialize dependency injection container
            self.container = initialize_application()
            
            self.logger.info("Quantitative Framework CLI initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize application: {str(e)}")
            sys.exit(1)
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser."""
        parser = argparse.ArgumentParser(
            description="Quantitative Framework CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument(
            '--config-dir',
            type=str,
            help='Configuration directory path'
        )
        
        parser.add_argument(
            '--environment',
            type=str,
            choices=['development', 'production', 'testing'],
            help='Environment to run in'
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_parser.add_argument('action', choices=['show', 'validate'], help='Config action')
        
        # Data command
        data_parser = subparsers.add_parser('data', help='Data management')
        data_parser.add_argument('action', choices=['fetch', 'validate'], help='Data action')
        data_parser.add_argument('--symbols', nargs='+', help='Stock symbols')
        data_parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
        data_parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
        
        # Portfolio command
        portfolio_parser = subparsers.add_parser('portfolio', help='Portfolio management')
        portfolio_parser.add_argument('action', choices=['create', 'list', 'optimize'], help='Portfolio action')
        portfolio_parser.add_argument('--name', help='Portfolio name')
        
        # Strategy command
        strategy_parser = subparsers.add_parser('strategy', help='Strategy management')
        strategy_parser.add_argument('action', choices=['list', 'run', 'backtest'], help='Strategy action')
        strategy_parser.add_argument('--name', help='Strategy name')
        
        return parser
    
    async def handle_config_command(self, args) -> None:
        """Handle configuration commands."""
        if args.action == 'show':
            config = self.config_manager.get_app_config()
            print(f"Environment: {config.environment}")
            print(f"Debug: {config.debug}")
            print(f"Database Host: {config.database.host}")
            print(f"Log Level: {config.logging.level}")
            
        elif args.action == 'validate':
            try:
                self.config_manager.validate()
                print("Configuration is valid")
            except QuantFrameworkError as e:
                print(f"Configuration validation failed: {e}")
                sys.exit(1)
    
    async def handle_data_command(self, args) -> None:
        """Handle data commands."""
        if args.action == 'fetch':
            if not args.symbols:
                print("Error: --symbols is required for fetch action")
                sys.exit(1)
            
            print(f"Fetching data for symbols: {args.symbols}")
            # This will be implemented when data management is available
            print("Data fetching not yet implemented - will be available in future tasks")
            
        elif args.action == 'validate':
            print("Data validation not yet implemented - will be available in future tasks")
    
    async def handle_portfolio_command(self, args) -> None:
        """Handle portfolio commands."""
        if args.action == 'create':
            if not args.name:
                print("Error: --name is required for create action")
                sys.exit(1)
            
            print(f"Creating portfolio: {args.name}")
            print("Portfolio creation not yet implemented - will be available in future tasks")
            
        elif args.action == 'list':
            print("Listing portfolios not yet implemented - will be available in future tasks")
            
        elif args.action == 'optimize':
            print("Portfolio optimization not yet implemented - will be available in future tasks")
    
    async def handle_strategy_command(self, args) -> None:
        """Handle strategy commands."""
        if args.action == 'list':
            print("Strategy listing not yet implemented - will be available in future tasks")
            
        elif args.action == 'run':
            if not args.name:
                print("Error: --name is required for run action")
                sys.exit(1)
            
            print(f"Running strategy: {args.name}")
            print("Strategy execution not yet implemented - will be available in future tasks")
            
        elif args.action == 'backtest':
            if not args.name:
                print("Error: --name is required for backtest action")
                sys.exit(1)
            
            print(f"Backtesting strategy: {args.name}")
            print("Strategy backtesting not yet implemented - will be available in future tasks")
    
    async def run(self, args) -> None:
        """Run the CLI application."""
        try:
            if args.command == 'config':
                await self.handle_config_command(args)
            elif args.command == 'data':
                await self.handle_data_command(args)
            elif args.command == 'portfolio':
                await self.handle_portfolio_command(args)
            elif args.command == 'strategy':
                await self.handle_strategy_command(args)
            else:
                print("No command specified. Use --help for available commands.")
                
        except QuantFrameworkError as e:
            self.logger.error(f"Application error: {e}")
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            print(f"Unexpected error: {e}")
            sys.exit(1)


def main():
    """Main entry point for CLI application."""
    cli = QuantFrameworkCLI()
    parser = cli.create_parser()
    args = parser.parse_args()
    
    # Set environment if specified
    if args.environment:
        import os
        os.environ['ENVIRONMENT'] = args.environment
    
    # Initialize application
    cli.initialize(args.config_dir)
    
    # Run the command
    asyncio.run(cli.run(args))


if __name__ == '__main__':
    main()