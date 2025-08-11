"""
Efficient data storage using Parquet/HDF5 with compression.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import hashlib
import aiofiles
from concurrent.futures import ThreadPoolExecutor


class StorageFormat(Enum):
    """Supported storage formats."""
    PARQUET = "parquet"
    HDF5 = "hdf5"
    CSV = "csv"
    FEATHER = "feather"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = None
    SNAPPY = "snappy"
    GZIP = "gzip"
    BROTLI = "brotli"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class StorageConfig:
    """Configuration for data storage."""
    base_path: str = "data"
    format: StorageFormat = StorageFormat.PARQUET
    compression: CompressionType = CompressionType.SNAPPY
    partition_by: Optional[List[str]] = None  # Columns to partition by
    max_file_size_mb: int = 100  # Maximum file size before splitting
    enable_indexing: bool = True
    metadata_enabled: bool = True
    backup_enabled: bool = False
    backup_retention_days: int = 30
    thread_pool_size: int = 4


@dataclass
class DataMetadata:
    """Metadata for stored data."""
    dataset_id: str
    created_at: datetime
    updated_at: datetime
    row_count: int
    column_count: int
    file_size_bytes: int
    schema: Dict[str, str]
    partitions: List[str] = field(default_factory=list)
    compression: Optional[str] = None
    checksum: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'dataset_id': self.dataset_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'row_count': self.row_count,
            'column_count': self.column_count,
            'file_size_bytes': self.file_size_bytes,
            'schema': self.schema,
            'partitions': self.partitions,
            'compression': self.compression,
            'checksum': self.checksum,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetadata':
        """Create metadata from dictionary."""
        return cls(
            dataset_id=data['dataset_id'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            row_count=data['row_count'],
            column_count=data['column_count'],
            file_size_bytes=data['file_size_bytes'],
            schema=data['schema'],
            partitions=data.get('partitions', []),
            compression=data.get('compression'),
            checksum=data.get('checksum'),
            tags=data.get('tags', {})
        )


class DataStorage:
    """Efficient data storage system with multiple format support."""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(config.base_path)
        self.metadata_path = self.base_path / "metadata"
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize storage directories and metadata."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.metadata_path.mkdir(parents=True, exist_ok=True)
            
            # Create format-specific directories
            for format_type in StorageFormat:
                format_dir = self.base_path / format_type.value
                format_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Data storage initialized at {self.base_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize storage: {str(e)}")
            raise
    
    async def store_data(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        metadata: Optional[DataMetadata] = None,
        format_override: Optional[StorageFormat] = None,
        compression_override: Optional[CompressionType] = None,
        **kwargs
    ) -> DataMetadata:
        """Store DataFrame with specified format and compression."""
        
        if data.empty:
            raise ValueError("Cannot store empty DataFrame")
        
        format_type = format_override or self.config.format
        compression = compression_override or self.config.compression
        
        try:
            # Generate file path
            file_path = self._generate_file_path(dataset_id, format_type)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Store data using thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                self._store_data_sync,
                data, file_path, format_type, compression, kwargs
            )
            
            # Calculate file size and checksum
            file_size = file_path.stat().st_size
            checksum = await self._calculate_checksum(file_path)
            
            # Create or update metadata
            if metadata is None:
                metadata = DataMetadata(
                    dataset_id=dataset_id,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    row_count=len(data),
                    column_count=len(data.columns),
                    file_size_bytes=file_size,
                    schema={col: str(dtype) for col, dtype in data.dtypes.items()},
                    compression=compression.value if compression else None,
                    checksum=checksum
                )
            else:
                metadata.updated_at = datetime.now()
                metadata.row_count = len(data)
                metadata.column_count = len(data.columns)
                metadata.file_size_bytes = file_size
                metadata.checksum = checksum
            
            # Save metadata
            if self.config.metadata_enabled:
                await self._save_metadata(metadata)
            
            # Create backup if enabled
            if self.config.backup_enabled:
                await self._create_backup(file_path, dataset_id)
            
            self.logger.info(f"Stored dataset '{dataset_id}' ({len(data)} rows, {file_size} bytes)")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to store dataset '{dataset_id}': {str(e)}")
            raise
    
    def _store_data_sync(
        self,
        data: pd.DataFrame,
        file_path: Path,
        format_type: StorageFormat,
        compression: CompressionType,
        kwargs: Dict[str, Any]
    ) -> None:
        """Synchronous data storage (runs in thread pool)."""
        
        if format_type == StorageFormat.PARQUET:
            # Parquet storage
            compression_map = {
                CompressionType.SNAPPY: 'snappy',
                CompressionType.GZIP: 'gzip',
                CompressionType.BROTLI: 'brotli',
                CompressionType.LZ4: 'lz4',
                CompressionType.ZSTD: 'zstd',
                CompressionType.NONE: None
            }
            
            data.to_parquet(
                file_path,
                compression=compression_map.get(compression),
                index=kwargs.get('index', False),
                partition_cols=self.config.partition_by,
                **{k: v for k, v in kwargs.items() if k not in ['index']}
            )
            
        elif format_type == StorageFormat.HDF5:
            # HDF5 storage
            compression_map = {
                CompressionType.GZIP: 'gzip',
                CompressionType.LZ4: 'lz4',
                CompressionType.ZSTD: 'zstd',
                CompressionType.NONE: None
            }
            
            comp = compression_map.get(compression)
            comp_opts = {'compression': comp} if comp else {}
            
            with pd.HDFStore(file_path, mode='w', **comp_opts) as store:
                store.put(
                    'data',
                    data,
                    format='table',
                    index=kwargs.get('index', False),
                    **{k: v for k, v in kwargs.items() if k not in ['index']}
                )
                
        elif format_type == StorageFormat.FEATHER:
            # Feather storage
            compression_map = {
                CompressionType.GZIP: 'gzip',
                CompressionType.LZ4: 'lz4',
                CompressionType.ZSTD: 'zstd',
                CompressionType.NONE: None
            }
            
            data.to_feather(
                file_path,
                compression=compression_map.get(compression),
                **kwargs
            )
            
        elif format_type == StorageFormat.CSV:
            # CSV storage
            if compression and compression != CompressionType.NONE:
                compression_ext = {
                    CompressionType.GZIP: '.gz',
                    CompressionType.BROTLI: '.bz2'
                }.get(compression, '')
                
                if compression_ext:
                    file_path = file_path.with_suffix(file_path.suffix + compression_ext)
            
            data.to_csv(
                file_path,
                compression=compression.value if compression != CompressionType.NONE else None,
                index=kwargs.get('index', False),
                **{k: v for k, v in kwargs.items() if k not in ['index']}
            )
        
        else:
            raise ValueError(f"Unsupported storage format: {format_type}")
    
    async def load_data(
        self,
        dataset_id: str,
        format_override: Optional[StorageFormat] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load DataFrame from storage."""
        
        format_type = format_override or self.config.format
        
        try:
            # Find file path
            file_path = self._find_file_path(dataset_id, format_type)
            
            if not file_path or not file_path.exists():
                raise FileNotFoundError(f"Dataset '{dataset_id}' not found")
            
            # Load data using thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.thread_pool,
                self._load_data_sync,
                file_path, format_type, columns, filters, kwargs
            )
            
            self.logger.info(f"Loaded dataset '{dataset_id}' ({len(data)} rows)")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset '{dataset_id}': {str(e)}")
            raise
    
    def _load_data_sync(
        self,
        file_path: Path,
        format_type: StorageFormat,
        columns: Optional[List[str]],
        filters: Optional[List[Tuple]],
        kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """Synchronous data loading (runs in thread pool)."""
        
        if format_type == StorageFormat.PARQUET:
            return pd.read_parquet(
                file_path,
                columns=columns,
                filters=filters,
                **kwargs
            )
            
        elif format_type == StorageFormat.HDF5:
            with pd.HDFStore(file_path, mode='r') as store:
                return store.select(
                    'data',
                    columns=columns,
                    where=self._convert_filters_to_hdf_where(filters) if filters else None,
                    **kwargs
                )
                
        elif format_type == StorageFormat.FEATHER:
            return pd.read_feather(
                file_path,
                columns=columns,
                **kwargs
            )
            
        elif format_type == StorageFormat.CSV:
            return pd.read_csv(
                file_path,
                usecols=columns,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported storage format: {format_type}")
    
    async def delete_data(self, dataset_id: str) -> bool:
        """Delete dataset from storage."""
        try:
            deleted = False
            
            # Try to find and delete files in all formats
            for format_type in StorageFormat:
                file_path = self._find_file_path(dataset_id, format_type)
                if file_path and file_path.exists():
                    if file_path.is_dir():
                        # Partitioned dataset
                        import shutil
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                    deleted = True
            
            # Delete metadata
            if self.config.metadata_enabled:
                metadata_file = self.metadata_path / f"{dataset_id}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
            
            if deleted:
                self.logger.info(f"Deleted dataset '{dataset_id}'")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete dataset '{dataset_id}': {str(e)}")
            return False
    
    async def list_datasets(self) -> List[str]:
        """List all available datasets."""
        try:
            datasets = set()
            
            # Scan all format directories
            for format_type in StorageFormat:
                format_dir = self.base_path / format_type.value
                if format_dir.exists():
                    for file_path in format_dir.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith('.'):
                            # Extract dataset ID from filename
                            dataset_id = file_path.stem
                            datasets.add(dataset_id)
            
            return sorted(list(datasets))
            
        except Exception as e:
            self.logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    async def get_metadata(self, dataset_id: str) -> Optional[DataMetadata]:
        """Get metadata for dataset."""
        if not self.config.metadata_enabled:
            return None
        
        try:
            metadata_file = self.metadata_path / f"{dataset_id}.json"
            
            if not metadata_file.exists():
                return None
            
            async with aiofiles.open(metadata_file, 'r') as f:
                data = json.loads(await f.read())
                return DataMetadata.from_dict(data)
                
        except Exception as e:
            self.logger.error(f"Failed to get metadata for '{dataset_id}': {str(e)}")
            return None
    
    async def update_metadata(self, metadata: DataMetadata) -> bool:
        """Update metadata for dataset."""
        if not self.config.metadata_enabled:
            return False
        
        try:
            metadata.updated_at = datetime.now()
            await self._save_metadata(metadata)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {str(e)}")
            return False
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage system statistics."""
        try:
            stats = {
                'total_datasets': 0,
                'total_size_bytes': 0,
                'format_breakdown': {},
                'compression_breakdown': {},
                'oldest_dataset': None,
                'newest_dataset': None
            }
            
            datasets = await self.list_datasets()
            stats['total_datasets'] = len(datasets)
            
            oldest_date = None
            newest_date = None
            
            for dataset_id in datasets:
                metadata = await self.get_metadata(dataset_id)
                if metadata:
                    stats['total_size_bytes'] += metadata.file_size_bytes
                    
                    # Format breakdown
                    format_key = self._detect_format(dataset_id)
                    if format_key:
                        stats['format_breakdown'][format_key] = stats['format_breakdown'].get(format_key, 0) + 1
                    
                    # Compression breakdown
                    if metadata.compression:
                        comp_key = metadata.compression
                        stats['compression_breakdown'][comp_key] = stats['compression_breakdown'].get(comp_key, 0) + 1
                    
                    # Date tracking
                    if oldest_date is None or metadata.created_at < oldest_date:
                        oldest_date = metadata.created_at
                        stats['oldest_dataset'] = dataset_id
                    
                    if newest_date is None or metadata.created_at > newest_date:
                        newest_date = metadata.created_at
                        stats['newest_dataset'] = dataset_id
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage stats: {str(e)}")
            return {}
    
    async def cleanup_old_data(self, retention_days: int) -> int:
        """Clean up old data based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            deleted_count = 0
            
            datasets = await self.list_datasets()
            
            for dataset_id in datasets:
                metadata = await self.get_metadata(dataset_id)
                if metadata and metadata.created_at < cutoff_date:
                    if await self.delete_data(dataset_id):
                        deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old datasets")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {str(e)}")
            return 0
    
    def _generate_file_path(self, dataset_id: str, format_type: StorageFormat) -> Path:
        """Generate file path for dataset."""
        format_dir = self.base_path / format_type.value
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_id}_{timestamp}.{format_type.value}"
        
        return format_dir / filename
    
    def _find_file_path(self, dataset_id: str, format_type: StorageFormat) -> Optional[Path]:
        """Find existing file path for dataset."""
        format_dir = self.base_path / format_type.value
        
        if not format_dir.exists():
            return None
        
        # Look for files starting with dataset_id
        for file_path in format_dir.glob(f"{dataset_id}*"):
            if file_path.is_file():
                return file_path
        
        return None
    
    def _detect_format(self, dataset_id: str) -> Optional[str]:
        """Detect storage format for dataset."""
        for format_type in StorageFormat:
            if self._find_file_path(dataset_id, format_type):
                return format_type.value
        return None
    
    async def _save_metadata(self, metadata: DataMetadata) -> None:
        """Save metadata to file."""
        metadata_file = self.metadata_path / f"{metadata.dataset_id}.json"
        
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata.to_dict(), indent=2))
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        
        async with aiofiles.open(file_path, 'rb') as f:
            async for chunk in self._read_chunks(f):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    async def _read_chunks(self, file_obj, chunk_size: int = 8192):
        """Read file in chunks asynchronously."""
        while True:
            chunk = await file_obj.read(chunk_size)
            if not chunk:
                break
            yield chunk
    
    async def _create_backup(self, file_path: Path, dataset_id: str) -> None:
        """Create backup of dataset."""
        try:
            backup_dir = self.base_path / "backups" / dataset_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{file_path.name}_{timestamp}"
            
            # Copy file
            import shutil
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.thread_pool,
                shutil.copy2,
                file_path, backup_path
            )
            
            # Cleanup old backups
            await self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {str(e)}")
    
    async def _cleanup_old_backups(self, backup_dir: Path) -> None:
        """Clean up old backup files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            for backup_file in backup_dir.glob("*"):
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        backup_file.unlink()
                        
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {str(e)}")
    
    def _convert_filters_to_hdf_where(self, filters: List[Tuple]) -> str:
        """Convert Parquet-style filters to HDF5 where clause."""
        # Simple conversion - would need more sophisticated logic for complex filters
        where_clauses = []
        
        for filter_tuple in filters:
            if len(filter_tuple) == 3:
                column, operator, value = filter_tuple
                if operator == '==':
                    where_clauses.append(f"{column} == {repr(value)}")
                elif operator == '!=':
                    where_clauses.append(f"{column} != {repr(value)}")
                elif operator == '>':
                    where_clauses.append(f"{column} > {value}")
                elif operator == '<':
                    where_clauses.append(f"{column} < {value}")
                elif operator == '>=':
                    where_clauses.append(f"{column} >= {value}")
                elif operator == '<=':
                    where_clauses.append(f"{column} <= {value}")
        
        return " & ".join(where_clauses) if where_clauses else None
    
    async def close(self) -> None:
        """Close storage system and cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            self.logger.info("Data storage system closed")
            
        except Exception as e:
            self.logger.error(f"Error closing storage system: {str(e)}")