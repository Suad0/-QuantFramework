"""
Centralized DataManager implementation that orchestrates all data management components.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import logging

from ...domain.interfaces import IDataManager
from ...domain.exceptions import DataError, ValidationError
from .providers import (
    MultiSourceDataProvider, 
    YahooFinanceProvider,
    AlphaVantageProvider,
    IEXCloudProvider,
    QuandlProvider,
    DataProviderConfig
)
from .cache import DataCache, CacheConfig
from .storage import DataStorage, StorageConfig, DataMetadata, CompressionType
from .quality import DataQualityValidator, DataQualityReport
from .versioning import DataVersionManager, ChangeType, DataSource


@dataclass
class DataManagerConfig:
    """Configuration for DataManager."""
    # Provider configurations
    providers: Dict[str, DataProviderConfig]
    
    # Cache configuration
    cache_config: CacheConfig
    
    # Storage configuration
    storage_config: StorageConfig
    
    # Quality validation configuration
    quality_config: Dict[str, Any]
    
    # Versioning configuration
    versioning_enabled: bool = True
    versioning_path: str = "data_versions"
    
    # General settings
    default_provider: str = "yahoo"
    failover_enabled: bool = True
    auto_quality_check: bool = True
    auto_versioning: bool = True
    
    @classmethod
    def create_default(cls) -> 'DataManagerConfig':
        """Create default configuration."""
        return cls(
            providers={
                'yahoo': DataProviderConfig(
                    name='yahoo',
                    enabled=True,
                    rate_limit=2000,
                    timeout=30,
                    retry_attempts=3
                ),
                'alphavantage': DataProviderConfig(
                    name='alphavantage',
                    enabled=False,  # Requires API key
                    rate_limit=5,
                    timeout=30,
                    retry_attempts=3
                ),
                'iex': DataProviderConfig(
                    name='iex',
                    enabled=False,  # Requires API key
                    rate_limit=100,
                    timeout=30,
                    retry_attempts=3
                ),
                'quandl': DataProviderConfig(
                    name='quandl',
                    enabled=False,  # Limited without API key
                    rate_limit=50,
                    timeout=30,
                    retry_attempts=3
                )
            },
            cache_config=CacheConfig(
                default_ttl_seconds=3600,
                max_size=1000,
                max_memory_mb=500,
                persistence_enabled=True
            ),
            storage_config=StorageConfig(
                base_path="data_storage",
                compression=CompressionType.SNAPPY,
                metadata_enabled=True
            ),
            quality_config={
                'enabled': True,
                'auto_fix': False,
                'strict_mode': False
            }
        )


class DataManager(IDataManager):
    """Centralized data manager that orchestrates all data management components."""
    
    def __init__(self, config: Optional[DataManagerConfig] = None):
        self.config = config or DataManagerConfig.create_default()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self._stats = {
            'requests_total': 0,
            'requests_cached': 0,
            'requests_failed': 0,
            'data_quality_issues': 0,
            'versions_created': 0
        }
    
    def _initialize_components(self) -> None:
        """Initialize all data management components."""
        try:
            # Initialize data providers
            self._providers = []
            
            for provider_name, provider_config in self.config.providers.items():
                if not provider_config.enabled:
                    continue
                
                if provider_name == 'yahoo':
                    provider = YahooFinanceProvider(provider_config)
                elif provider_name == 'alphavantage':
                    provider = AlphaVantageProvider(provider_config)
                elif provider_name == 'iex':
                    provider = IEXCloudProvider(provider_config)
                elif provider_name == 'quandl':
                    provider = QuandlProvider(provider_config)
                else:
                    self.logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                self._providers.append(provider)
            
            # Initialize multi-source provider
            self._data_provider = MultiSourceDataProvider(
                providers=self._providers,
                failover_enabled=self.config.failover_enabled
            )
            
            # Initialize cache
            self._cache = DataCache(self.config.cache_config)
            self._cache.initialize_sync()  # Initialize synchronously
            
            # Initialize storage
            self._storage = DataStorage(self.config.storage_config)
            
            # Initialize quality validator
            self._quality_validator = DataQualityValidator(self.config.quality_config)
            
            # Initialize version manager
            if self.config.versioning_enabled:
                self._version_manager = DataVersionManager(self.config.versioning_path)
            else:
                self._version_manager = None
            
            self.logger.info("DataManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DataManager: {str(e)}")
            raise
    
    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        provider: Optional[str] = None,
        use_cache: bool = True,
        store_data: bool = True,
        create_version: bool = None
    ) -> pd.DataFrame:
        """Fetch market data with full data management pipeline."""
        
        self._stats['requests_total'] += 1
        
        try:
            # Validate inputs
            if not symbols:
                raise ValidationError("Symbols list cannot be empty")
            
            if start_date >= end_date:
                raise ValidationError("Start date must be before end date")
            
            # Generate cache key
            cache_key = self._generate_cache_key(symbols, start_date, end_date, provider)
            
            # Try cache first
            if use_cache:
                cached_data = await self.get_cached_data(cache_key)
                if cached_data is not None:
                    self._stats['requests_cached'] += 1
                    self.logger.info(f"Retrieved data from cache for {len(symbols)} symbols")
                    return cached_data
            
            # Fetch fresh data
            self.logger.info(f"Fetching market data for {len(symbols)} symbols from {start_date} to {end_date}")
            
            data = await self._data_provider.fetch_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                preferred_provider=provider or self.config.default_provider
            )
            
            if data.empty:
                raise DataError("No data returned from providers")
            
            # Quality validation
            if self.config.auto_quality_check:
                quality_report = await self.validate_data_quality(data)
                
                if quality_report.has_critical_issues():
                    self._stats['data_quality_issues'] += 1
                    if self.config.quality_config.get('strict_mode', False):
                        raise DataError(f"Data quality validation failed: {len(quality_report.get_issues_by_severity('critical'))} critical issues")
                    else:
                        self.logger.warning(f"Data quality issues detected: {len(quality_report.issues)} total issues")
                
                # Auto-fix if enabled
                if self.config.quality_config.get('auto_fix', False):
                    data = await self._apply_quality_fixes(data, quality_report)
            
            # Store data
            if store_data:
                dataset_id = self._generate_dataset_id(symbols, start_date, end_date)
                
                metadata = await self.store_data(data, {
                    'dataset_id': dataset_id,
                    'symbols': symbols,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'provider': provider,
                    'fetch_timestamp': datetime.now().isoformat()
                })
                
                # Create version if enabled
                if (create_version is True) or (create_version is None and self.config.auto_versioning):
                    if self._version_manager:
                        await self._version_manager.create_version(
                            dataset_id=dataset_id,
                            data=data,
                            created_by="system",
                            change_type=ChangeType.CREATE,
                            change_description=f"Fetched market data for {len(symbols)} symbols",
                            metadata={
                                'symbols': symbols,
                                'date_range': f"{start_date.date()} to {end_date.date()}",
                                'provider': provider
                            }
                        )
                        self._stats['versions_created'] += 1
            
            # Cache the result
            if use_cache:
                self._cache.set_sync(
                    key=cache_key,
                    data=data,
                    ttl_seconds=self.config.cache_config.default_ttl_seconds,
                    metadata={
                        'symbols': symbols,
                        'date_range': f"{start_date} to {end_date}",
                        'provider': provider
                    }
                )
            
            self.logger.info(f"Successfully fetched {len(data)} rows for {len(symbols)} symbols")
            
            return data
            
        except Exception as e:
            self._stats['requests_failed'] += 1
            self.logger.error(f"Failed to fetch market data: {str(e)}")
            raise DataError(f"Market data fetch failed: {str(e)}")
    
    async def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data by key."""
        try:
            return self._cache.get_sync(cache_key)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {str(e)}")
            return None
    
    async def store_data(self, data: pd.DataFrame, metadata: Dict[str, Any]) -> DataMetadata:
        """Store data with associated metadata."""
        try:
            dataset_id = metadata.get('dataset_id') or self._generate_dataset_id_from_data(data)
            
            storage_metadata = await self._storage.store_data(
                data=data,
                dataset_id=dataset_id,
                metadata=None  # Will be created by storage
            )
            
            self.logger.info(f"Stored dataset '{dataset_id}' with {len(data)} rows")
            
            return storage_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to store data: {str(e)}")
            raise DataError(f"Data storage failed: {str(e)}")
    
    async def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Validate data quality and return quality report."""
        try:
            return await self._quality_validator.validate_data_quality(data)
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {str(e)}")
            # Return a basic report indicating validation failure
            from .quality import DataQualityReport, ValidationIssue, ValidationSeverity
            
            return DataQualityReport(
                is_valid=False,
                total_rows=len(data) if not data.empty else 0,
                total_columns=len(data.columns) if not data.empty else 0,
                issues=[ValidationIssue(
                    rule_name='validation_error',
                    severity=ValidationSeverity.ERROR,
                    message=f"Quality validation failed: {str(e)}"
                )]
            )
    
    async def load_stored_data(
        self,
        dataset_id: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None
    ) -> Optional[pd.DataFrame]:
        """Load previously stored data."""
        try:
            return await self._storage.load_data(
                dataset_id=dataset_id,
                columns=columns,
                filters=filters
            )
        except Exception as e:
            self.logger.error(f"Failed to load stored data: {str(e)}")
            return None
    
    async def get_data_metadata(self, dataset_id: str) -> Optional[DataMetadata]:
        """Get metadata for stored dataset."""
        try:
            return await self._storage.get_metadata(dataset_id)
        except Exception as e:
            self.logger.error(f"Failed to get data metadata: {str(e)}")
            return None
    
    async def list_stored_datasets(self) -> List[str]:
        """List all stored datasets."""
        try:
            return await self._storage.list_datasets()
        except Exception as e:
            self.logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    async def delete_stored_data(self, dataset_id: str) -> bool:
        """Delete stored dataset."""
        try:
            # Delete from storage
            storage_deleted = await self._storage.delete_data(dataset_id)
            
            # Invalidate cache
            self._cache.invalidate_pattern_sync(f"*{dataset_id}*")
            
            return storage_deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete stored data: {str(e)}")
            return False
    
    async def get_available_providers(self) -> List[str]:
        """Get list of available data providers."""
        return self._data_provider.get_available_providers()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            return self._cache.get_stats_sync()
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            return await self._storage.get_storage_stats()
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """Get overall data manager statistics."""
        stats = self._stats.copy()
        
        # Add component stats
        try:
            stats['cache'] = await self.get_cache_stats()
            stats['storage'] = await self.get_storage_stats()
            stats['providers'] = {
                'available': await self.get_available_providers(),
                'total_configured': len(self.config.providers),
                'enabled': len([p for p in self.config.providers.values() if p.enabled])
            }
            
            if self._version_manager:
                stats['versioning'] = {
                    'enabled': True,
                    'versions_created': self._stats['versions_created']
                }
            else:
                stats['versioning'] = {'enabled': False}
                
        except Exception as e:
            self.logger.warning(f"Failed to collect some statistics: {str(e)}")
        
        return stats
    
    async def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up old data based on retention policy."""
        cleanup_stats = {
            'storage_cleaned': 0,
            'cache_cleaned': 0,
            'versions_cleaned': 0
        }
        
        try:
            # Clean up storage
            cleanup_stats['storage_cleaned'] = await self._storage.cleanup_old_data(retention_days)
            
            # Clean up cache (expired entries)
            self._cache.cleanup_expired_entries_sync()
            
            # Clean up versions if enabled
            if self._version_manager:
                datasets = await self.list_stored_datasets()
                for dataset_id in datasets:
                    cleaned = await self._version_manager.cleanup_old_versions(
                        dataset_id=dataset_id,
                        keep_versions=10,
                        keep_days=retention_days
                    )
                    cleanup_stats['versions_cleaned'] += cleaned
            
            self.logger.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
        
        return cleanup_stats
    
    def _generate_cache_key(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        provider: Optional[str] = None
    ) -> str:
        """Generate cache key for data request."""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        provider_str = provider or "default"
        
        return f"market_data_{symbols_str}_{start_str}_{end_str}_{provider_str}"
    
    def _generate_dataset_id(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Generate dataset ID for storage."""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"market_data_{symbols_str}_{start_str}_{end_str}_{timestamp}"
    
    def _generate_dataset_id_from_data(self, data: pd.DataFrame) -> str:
        """Generate dataset ID from data content."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if 'symbol' in data.columns:
            symbols = sorted(data['symbol'].unique())
            symbols_str = "_".join(symbols[:5])  # Limit to first 5 symbols
            return f"dataset_{symbols_str}_{timestamp}"
        else:
            return f"dataset_{timestamp}"
    
    async def _apply_quality_fixes(
        self,
        data: pd.DataFrame,
        quality_report: DataQualityReport
    ) -> pd.DataFrame:
        """Apply automatic fixes for data quality issues."""
        
        fixed_data = data.copy()
        
        try:
            # Apply basic fixes based on quality issues
            for issue in quality_report.issues:
                if issue.rule_name == 'missing_values' and issue.affected_columns:
                    for column in issue.affected_columns:
                        if column in fixed_data.columns:
                            # Forward fill for price data
                            if column in ['open', 'high', 'low', 'close', 'adj_close']:
                                fixed_data[column] = fixed_data[column].fillna(method='ffill')
                            # Zero fill for volume
                            elif column == 'volume':
                                fixed_data[column] = fixed_data[column].fillna(0)
                
                elif issue.rule_name == 'duplicate_records' and issue.affected_rows:
                    # Remove duplicates
                    if 'symbol' in fixed_data.columns and 'date' in fixed_data.columns:
                        fixed_data = fixed_data.drop_duplicates(subset=['symbol', 'date'], keep='first')
                
                elif issue.rule_name == 'negative_prices' and issue.affected_rows:
                    # Set negative prices to NaN (will be forward filled if needed)
                    for column in issue.affected_columns:
                        if column in fixed_data.columns:
                            fixed_data.loc[fixed_data[column] < 0, column] = None
            
            self.logger.info(f"Applied automatic quality fixes to data")
            
        except Exception as e:
            self.logger.warning(f"Failed to apply quality fixes: {str(e)}")
            return data  # Return original data if fixes fail
        
        return fixed_data
    
    def close(self) -> None:
        """Close data manager and cleanup resources."""
        try:
            if hasattr(self, '_cache'):
                self._cache.close_sync()
            
            # Note: Storage close is async, but we'll skip it in sync mode for now
            # if hasattr(self, '_storage'):
            #     await self._storage.close()
            
            self.logger.info("DataManager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing DataManager: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, '_cache') or hasattr(self, '_storage'):
                # Try to close gracefully if event loop is available
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.close())
                    else:
                        loop.run_until_complete(self.close())
                except:
                    pass  # Ignore errors during cleanup
        except:
            pass