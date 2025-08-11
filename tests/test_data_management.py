"""
Tests for data management infrastructure.
"""

import pytest
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.infrastructure.data.data_manager import DataManager, DataManagerConfig
from src.infrastructure.data.providers import DataProviderConfig
from src.infrastructure.data.cache import CacheConfig
from src.infrastructure.data.storage import StorageConfig
from src.infrastructure.data.quality import DataQualityValidator
from src.infrastructure.data.versioning import DataVersionManager, ChangeType


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'date': date,
                'symbol': symbol,
                'open': 100.0 + (hash(f"{symbol}{date}") % 20),
                'high': 105.0 + (hash(f"{symbol}{date}") % 20),
                'low': 95.0 + (hash(f"{symbol}{date}") % 20),
                'close': 102.0 + (hash(f"{symbol}{date}") % 20),
                'volume': 1000000 + (hash(f"{symbol}{date}") % 500000),
                'adj_close': 102.0 + (hash(f"{symbol}{date}") % 20),
                'provider': 'test',
                'timestamp': datetime.now()
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return DataManagerConfig(
        providers={
            'yahoo': DataProviderConfig(
                name='yahoo',
                enabled=True,
                rate_limit=100,
                timeout=10
            )
        },
        cache_config=CacheConfig(
            default_ttl_seconds=300,
            max_size=100,
            persistence_enabled=False  # Disable for tests
        ),
        storage_config=StorageConfig(
            base_path=str(Path(temp_dir) / "storage"),
            metadata_enabled=True
        ),
        quality_config={
            'enabled': True,
            'auto_fix': False,
            'strict_mode': False
        },
        versioning_enabled=True,
        versioning_path=str(Path(temp_dir) / "versions"),
        auto_quality_check=True,
        auto_versioning=False  # Disable for tests
    )


class TestDataQualityValidator:
    """Test data quality validation."""
    
    def test_quality_validator_initialization(self):
        """Test quality validator initialization."""
        validator = DataQualityValidator()
        assert validator is not None
        assert len(validator.rules) > 0
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, sample_data):
        """Test data quality validation."""
        validator = DataQualityValidator()
        
        report = await validator.validate_data_quality(sample_data)
        
        assert report is not None
        assert report.total_rows == len(sample_data)
        assert report.total_columns == len(sample_data.columns)
        assert isinstance(report.is_valid, bool)
    
    @pytest.mark.asyncio
    async def test_empty_data_validation(self):
        """Test validation of empty data."""
        validator = DataQualityValidator()
        empty_data = pd.DataFrame()
        
        report = await validator.validate_data_quality(empty_data)
        
        assert report is not None
        assert not report.is_valid
        assert len(report.issues) > 0


class TestDataVersionManager:
    """Test data versioning system."""
    
    def test_version_manager_initialization(self, temp_dir):
        """Test version manager initialization."""
        version_path = str(Path(temp_dir) / "versions")
        manager = DataVersionManager(version_path)
        
        assert manager is not None
        assert Path(version_path).exists()
    
    @pytest.mark.asyncio
    async def test_create_version(self, sample_data, temp_dir):
        """Test version creation."""
        version_path = str(Path(temp_dir) / "versions")
        manager = DataVersionManager(version_path)
        
        version = await manager.create_version(
            dataset_id="test_dataset",
            data=sample_data,
            created_by="test_user",
            change_type=ChangeType.CREATE,
            change_description="Test version creation"
        )
        
        assert version is not None
        assert version.dataset_id == "test_dataset"
        assert version.version_number == 1
        assert version.row_count == len(sample_data)
        assert version.column_count == len(sample_data.columns)
    
    @pytest.mark.asyncio
    async def test_load_version_data(self, sample_data, temp_dir):
        """Test loading version data."""
        version_path = str(Path(temp_dir) / "versions")
        manager = DataVersionManager(version_path)
        
        # Create version
        version = await manager.create_version(
            dataset_id="test_dataset",
            data=sample_data,
            created_by="test_user",
            change_type=ChangeType.CREATE,
            change_description="Test version"
        )
        
        # Load version data
        loaded_data = await manager.load_version_data(version.version_id)
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)


class TestDataManager:
    """Test centralized data manager."""
    
    def test_data_manager_initialization(self, test_config):
        """Test data manager initialization."""
        manager = DataManager(test_config)
        
        assert manager is not None
        assert manager.config == test_config
    
    @pytest.mark.asyncio
    async def test_store_and_load_data(self, sample_data, test_config):
        """Test data storage and loading."""
        manager = DataManager(test_config)
        
        # Store data
        metadata = await manager.store_data(
            data=sample_data,
            metadata={'test': 'metadata'}
        )
        
        assert metadata is not None
        assert metadata.row_count == len(sample_data)
        
        # Load data
        loaded_data = await manager.load_stored_data(metadata.dataset_id)
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, sample_data, test_config):
        """Test data quality validation through manager."""
        manager = DataManager(test_config)
        
        report = await manager.validate_data_quality(sample_data)
        
        assert report is not None
        assert report.total_rows == len(sample_data)
        assert isinstance(report.is_valid, bool)
    
    @pytest.mark.asyncio
    async def test_manager_stats(self, test_config):
        """Test manager statistics."""
        manager = DataManager(test_config)
        
        stats = await manager.get_manager_stats()
        
        assert stats is not None
        assert 'requests_total' in stats
        assert 'cache' in stats
        assert 'storage' in stats
        assert 'providers' in stats
    
    @pytest.mark.asyncio
    async def test_list_datasets(self, sample_data, test_config):
        """Test listing stored datasets."""
        manager = DataManager(test_config)
        
        # Store some data
        await manager.store_data(
            data=sample_data,
            metadata={'dataset_id': 'test_dataset_1'}
        )
        
        # List datasets
        datasets = await manager.list_stored_datasets()
        
        assert isinstance(datasets, list)
        assert len(datasets) >= 1
    
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, test_config):
        """Test resource cleanup."""
        manager = DataManager(test_config)
        
        # This should not raise an exception
        await manager.close()


@pytest.mark.asyncio
async def test_integration_workflow(sample_data, test_config):
    """Test complete data management workflow."""
    manager = DataManager(test_config)
    
    try:
        # 1. Validate data quality
        quality_report = await manager.validate_data_quality(sample_data)
        assert quality_report is not None
        
        # 2. Store data
        metadata = await manager.store_data(
            data=sample_data,
            metadata={'source': 'integration_test'}
        )
        assert metadata is not None
        
        # 3. Load data back
        loaded_data = await manager.load_stored_data(metadata.dataset_id)
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        
        # 4. Get statistics
        stats = await manager.get_manager_stats()
        assert stats is not None
        
        # 5. List datasets
        datasets = await manager.list_stored_datasets()
        assert metadata.dataset_id in [d for d in datasets if metadata.dataset_id in d]
        
        # 6. Clean up
        deleted = await manager.delete_stored_data(metadata.dataset_id)
        assert deleted
        
    finally:
        await manager.close()


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        # Create sample data
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'symbol': ['AAPL'] * 5,
            'close': [150, 151, 149, 152, 153],
            'volume': [1000000] * 5
        })
        
        # Test quality validation
        validator = DataQualityValidator()
        report = await validator.validate_data_quality(data)
        print(f"Quality validation: {report.is_valid}, Issues: {len(report.issues)}")
        
        # Test version manager
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DataVersionManager(temp_dir)
            version = await manager.create_version(
                dataset_id="test",
                data=data,
                created_by="test",
                change_type=ChangeType.CREATE,
                change_description="Test"
            )
            print(f"Created version: {version.version_id}")
        
        print("Simple test completed successfully!")
    
    asyncio.run(simple_test())