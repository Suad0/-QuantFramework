"""
Data versioning and audit trail system.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import logging
import aiofiles
from uuid import uuid4


class ChangeType(Enum):
    """Types of data changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SCHEMA_CHANGE = "schema_change"
    MERGE = "merge"
    SPLIT = "split"


class DataSource(Enum):
    """Sources of data changes."""
    USER = "user"
    SYSTEM = "system"
    API = "api"
    BATCH_JOB = "batch_job"
    MIGRATION = "migration"


@dataclass
class DataVersion:
    """Represents a version of a dataset."""
    version_id: str
    dataset_id: str
    version_number: int
    parent_version_id: Optional[str]
    created_at: datetime
    created_by: str
    change_type: ChangeType
    change_description: str
    data_checksum: str
    schema_checksum: str
    row_count: int
    column_count: int
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['change_type'] = self.change_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['change_type'] = ChangeType(data['change_type'])
        return cls(**data)


@dataclass
class AuditLogEntry:
    """Represents an audit log entry."""
    entry_id: str
    timestamp: datetime
    dataset_id: str
    version_id: Optional[str]
    user_id: str
    action: str
    source: DataSource
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['source'] = self.source.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['source'] = DataSource(data['source'])
        return cls(**data)


@dataclass
class DataLineage:
    """Represents data lineage information."""
    dataset_id: str
    upstream_datasets: List[str] = field(default_factory=list)
    downstream_datasets: List[str] = field(default_factory=list)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class DataVersionManager:
    """Manages data versioning and audit trails."""
    
    def __init__(self, base_path: str = "data_versions"):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / "versions"
        self.audit_path = self.base_path / "audit"
        self.lineage_path = self.base_path / "lineage"
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._initialize_storage()
        
        # In-memory caches
        self._version_cache: Dict[str, DataVersion] = {}
        self._lineage_cache: Dict[str, DataLineage] = {}
    
    def _initialize_storage(self) -> None:
        """Initialize version storage directories."""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.versions_path.mkdir(exist_ok=True)
            self.audit_path.mkdir(exist_ok=True)
            self.lineage_path.mkdir(exist_ok=True)
            
            self.logger.info(f"Data version manager initialized at {self.base_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize version storage: {str(e)}")
            raise
    
    async def create_version(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        created_by: str,
        change_type: ChangeType,
        change_description: str,
        parent_version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> DataVersion:
        """Create a new version of a dataset."""
        
        try:
            # Generate version ID and get version number
            version_id = str(uuid4())
            version_number = await self._get_next_version_number(dataset_id)
            
            # Calculate checksums
            data_checksum = self._calculate_data_checksum(data)
            schema_checksum = self._calculate_schema_checksum(data)
            
            # Store data file
            file_path = await self._store_version_data(version_id, data)
            
            # Create version object
            version = DataVersion(
                version_id=version_id,
                dataset_id=dataset_id,
                version_number=version_number,
                parent_version_id=parent_version_id,
                created_at=datetime.now(),
                created_by=created_by,
                change_type=change_type,
                change_description=change_description,
                data_checksum=data_checksum,
                schema_checksum=schema_checksum,
                row_count=len(data),
                column_count=len(data.columns),
                file_path=str(file_path),
                metadata=metadata or {},
                tags=tags or []
            )
            
            # Save version metadata
            await self._save_version_metadata(version)
            
            # Update cache
            self._version_cache[version_id] = version
            
            # Log audit entry
            await self.log_audit_entry(
                dataset_id=dataset_id,
                version_id=version_id,
                user_id=created_by,
                action="create_version",
                source=DataSource.USER,
                details={
                    'change_type': change_type.value,
                    'change_description': change_description,
                    'row_count': len(data),
                    'column_count': len(data.columns)
                }
            )
            
            self.logger.info(f"Created version {version_number} for dataset '{dataset_id}'")
            
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version for dataset '{dataset_id}': {str(e)}")
            raise
    
    async def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get a specific version by ID."""
        
        # Check cache first
        if version_id in self._version_cache:
            return self._version_cache[version_id]
        
        try:
            version_file = self.versions_path / f"{version_id}.json"
            
            if not version_file.exists():
                return None
            
            async with aiofiles.open(version_file, 'r') as f:
                data = json.loads(await f.read())
                version = DataVersion.from_dict(data)
                
                # Update cache
                self._version_cache[version_id] = version
                
                return version
                
        except Exception as e:
            self.logger.error(f"Failed to get version '{version_id}': {str(e)}")
            return None
    
    async def get_latest_version(self, dataset_id: str) -> Optional[DataVersion]:
        """Get the latest version of a dataset."""
        
        try:
            versions = await self.list_versions(dataset_id)
            
            if not versions:
                return None
            
            # Sort by version number and return latest
            latest_version = max(versions, key=lambda v: v.version_number)
            return latest_version
            
        except Exception as e:
            self.logger.error(f"Failed to get latest version for dataset '{dataset_id}': {str(e)}")
            return None
    
    async def list_versions(
        self,
        dataset_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[DataVersion]:
        """List all versions of a dataset."""
        
        try:
            versions = []
            
            # Scan version files
            for version_file in self.versions_path.glob("*.json"):
                try:
                    async with aiofiles.open(version_file, 'r') as f:
                        data = json.loads(await f.read())
                        
                        if data.get('dataset_id') == dataset_id:
                            version = DataVersion.from_dict(data)
                            versions.append(version)
                            
                except Exception as e:
                    self.logger.warning(f"Failed to read version file {version_file}: {str(e)}")
                    continue
            
            # Sort by version number (descending)
            versions.sort(key=lambda v: v.version_number, reverse=True)
            
            # Apply pagination
            if offset > 0:
                versions = versions[offset:]
            
            if limit is not None:
                versions = versions[:limit]
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Failed to list versions for dataset '{dataset_id}': {str(e)}")
            return []
    
    async def load_version_data(self, version_id: str) -> Optional[pd.DataFrame]:
        """Load data for a specific version."""
        
        try:
            version = await self.get_version(version_id)
            
            if not version:
                return None
            
            file_path = Path(version.file_path)
            
            if not file_path.exists():
                self.logger.error(f"Version data file not found: {file_path}")
                return None
            
            # Load data based on file extension
            if file_path.suffix == '.parquet':
                data = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix == '.feather':
                data = pd.read_feather(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Verify data integrity
            current_checksum = self._calculate_data_checksum(data)
            if current_checksum != version.data_checksum:
                self.logger.warning(f"Data checksum mismatch for version {version_id}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load data for version '{version_id}': {str(e)}")
            return None
    
    async def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """Compare two versions of a dataset."""
        
        try:
            version1 = await self.get_version(version_id_1)
            version2 = await self.get_version(version_id_2)
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            if version1.dataset_id != version2.dataset_id:
                raise ValueError("Versions belong to different datasets")
            
            # Load data for both versions
            data1 = await self.load_version_data(version_id_1)
            data2 = await self.load_version_data(version_id_2)
            
            if data1 is None or data2 is None:
                raise ValueError("Failed to load data for comparison")
            
            # Compare basic metrics
            comparison = {
                'version_1': {
                    'version_id': version_id_1,
                    'version_number': version1.version_number,
                    'created_at': version1.created_at.isoformat(),
                    'row_count': version1.row_count,
                    'column_count': version1.column_count
                },
                'version_2': {
                    'version_id': version_id_2,
                    'version_number': version2.version_number,
                    'created_at': version2.created_at.isoformat(),
                    'row_count': version2.row_count,
                    'column_count': version2.column_count
                },
                'differences': {
                    'row_count_diff': version2.row_count - version1.row_count,
                    'column_count_diff': version2.column_count - version1.column_count,
                    'schema_changed': version1.schema_checksum != version2.schema_checksum,
                    'data_changed': version1.data_checksum != version2.data_checksum
                }
            }
            
            # Schema comparison
            schema1 = {col: str(dtype) for col, dtype in data1.dtypes.items()}
            schema2 = {col: str(dtype) for col, dtype in data2.dtypes.items()}
            
            added_columns = set(schema2.keys()) - set(schema1.keys())
            removed_columns = set(schema1.keys()) - set(schema2.keys())
            changed_columns = {
                col for col in set(schema1.keys()) & set(schema2.keys())
                if schema1[col] != schema2[col]
            }
            
            comparison['schema_changes'] = {
                'added_columns': list(added_columns),
                'removed_columns': list(removed_columns),
                'changed_columns': list(changed_columns)
            }
            
            # Data comparison (if schemas are compatible)
            if not added_columns and not removed_columns and not changed_columns:
                try:
                    # Compare data values
                    common_columns = list(set(data1.columns) & set(data2.columns))
                    
                    if common_columns:
                        data1_subset = data1[common_columns].sort_values(common_columns[0]).reset_index(drop=True)
                        data2_subset = data2[common_columns].sort_values(common_columns[0]).reset_index(drop=True)
                        
                        if len(data1_subset) == len(data2_subset):
                            differences = data1_subset.compare(data2_subset)
                            comparison['data_changes'] = {
                                'changed_cells': len(differences),
                                'changed_rows': len(differences.index.unique()) if not differences.empty else 0
                            }
                        else:
                            comparison['data_changes'] = {
                                'note': 'Row counts differ - detailed comparison not performed'
                            }
                
                except Exception as e:
                    comparison['data_changes'] = {
                        'error': f'Data comparison failed: {str(e)}'
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {str(e)}")
            raise
    
    async def delete_version(self, version_id: str, user_id: str) -> bool:
        """Delete a specific version."""
        
        try:
            version = await self.get_version(version_id)
            
            if not version:
                return False
            
            # Delete data file
            file_path = Path(version.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata file
            version_file = self.versions_path / f"{version_id}.json"
            if version_file.exists():
                version_file.unlink()
            
            # Remove from cache
            if version_id in self._version_cache:
                del self._version_cache[version_id]
            
            # Log audit entry
            await self.log_audit_entry(
                dataset_id=version.dataset_id,
                version_id=version_id,
                user_id=user_id,
                action="delete_version",
                source=DataSource.USER,
                details={'version_number': version.version_number}
            )
            
            self.logger.info(f"Deleted version {version_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete version '{version_id}': {str(e)}")
            return False
    
    async def log_audit_entry(
        self,
        dataset_id: str,
        user_id: str,
        action: str,
        source: DataSource,
        version_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Log an audit entry."""
        
        try:
            entry_id = str(uuid4())
            
            entry = AuditLogEntry(
                entry_id=entry_id,
                timestamp=datetime.now(),
                dataset_id=dataset_id,
                version_id=version_id,
                user_id=user_id,
                action=action,
                source=source,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id
            )
            
            # Save audit entry
            audit_file = self.audit_path / f"{entry_id}.json"
            
            async with aiofiles.open(audit_file, 'w') as f:
                await f.write(json.dumps(entry.to_dict(), indent=2))
            
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Failed to log audit entry: {str(e)}")
            raise
    
    async def get_audit_trail(
        self,
        dataset_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[AuditLogEntry]:
        """Get audit trail with optional filters."""
        
        try:
            entries = []
            
            # Scan audit files
            for audit_file in self.audit_path.glob("*.json"):
                try:
                    async with aiofiles.open(audit_file, 'r') as f:
                        data = json.loads(await f.read())
                        entry = AuditLogEntry.from_dict(data)
                        
                        # Apply filters
                        if dataset_id and entry.dataset_id != dataset_id:
                            continue
                        
                        if user_id and entry.user_id != user_id:
                            continue
                        
                        if action and entry.action != action:
                            continue
                        
                        if start_date and entry.timestamp < start_date:
                            continue
                        
                        if end_date and entry.timestamp > end_date:
                            continue
                        
                        entries.append(entry)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read audit file {audit_file}: {str(e)}")
                    continue
            
            # Sort by timestamp (descending)
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            
            # Apply pagination
            if offset > 0:
                entries = entries[offset:]
            
            if limit is not None:
                entries = entries[:limit]
            
            return entries
            
        except Exception as e:
            self.logger.error(f"Failed to get audit trail: {str(e)}")
            return []
    
    async def track_lineage(
        self,
        dataset_id: str,
        upstream_datasets: Optional[List[str]] = None,
        transformations: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Track data lineage for a dataset."""
        
        try:
            # Load existing lineage or create new
            lineage = await self._load_lineage(dataset_id)
            
            if lineage is None:
                lineage = DataLineage(dataset_id=dataset_id)
            
            # Update upstream datasets
            if upstream_datasets:
                lineage.upstream_datasets = list(set(lineage.upstream_datasets + upstream_datasets))
                
                # Update downstream lineage for upstream datasets
                for upstream_id in upstream_datasets:
                    upstream_lineage = await self._load_lineage(upstream_id)
                    if upstream_lineage is None:
                        upstream_lineage = DataLineage(dataset_id=upstream_id)
                    
                    if dataset_id not in upstream_lineage.downstream_datasets:
                        upstream_lineage.downstream_datasets.append(dataset_id)
                        upstream_lineage.updated_at = datetime.now()
                        await self._save_lineage(upstream_lineage)
            
            # Update transformations
            if transformations:
                lineage.transformations.extend(transformations)
            
            lineage.updated_at = datetime.now()
            
            # Save lineage
            await self._save_lineage(lineage)
            
            # Update cache
            self._lineage_cache[dataset_id] = lineage
            
        except Exception as e:
            self.logger.error(f"Failed to track lineage for dataset '{dataset_id}': {str(e)}")
            raise
    
    async def get_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """Get lineage information for a dataset."""
        
        # Check cache first
        if dataset_id in self._lineage_cache:
            return self._lineage_cache[dataset_id]
        
        return await self._load_lineage(dataset_id)
    
    async def _get_next_version_number(self, dataset_id: str) -> int:
        """Get the next version number for a dataset."""
        
        versions = await self.list_versions(dataset_id)
        
        if not versions:
            return 1
        
        max_version = max(v.version_number for v in versions)
        return max_version + 1
    
    def _calculate_data_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data content."""
        # Convert DataFrame to string representation and hash
        data_str = data.to_string(index=False)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _calculate_schema_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data schema."""
        schema_str = str(sorted([(col, str(dtype)) for col, dtype in data.dtypes.items()]))
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    async def _store_version_data(self, version_id: str, data: pd.DataFrame) -> Path:
        """Store version data to file."""
        
        # Use Parquet for efficient storage
        file_path = self.versions_path / f"{version_id}.parquet"
        
        # Store data
        data.to_parquet(file_path, compression='snappy', index=False)
        
        return file_path
    
    async def _save_version_metadata(self, version: DataVersion) -> None:
        """Save version metadata to file."""
        
        version_file = self.versions_path / f"{version.version_id}.json"
        
        async with aiofiles.open(version_file, 'w') as f:
            await f.write(json.dumps(version.to_dict(), indent=2))
    
    async def _load_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """Load lineage information from file."""
        
        try:
            lineage_file = self.lineage_path / f"{dataset_id}.json"
            
            if not lineage_file.exists():
                return None
            
            async with aiofiles.open(lineage_file, 'r') as f:
                data = json.loads(await f.read())
                
                lineage = DataLineage(
                    dataset_id=data['dataset_id'],
                    upstream_datasets=data.get('upstream_datasets', []),
                    downstream_datasets=data.get('downstream_datasets', []),
                    transformations=data.get('transformations', []),
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at'])
                )
                
                # Update cache
                self._lineage_cache[dataset_id] = lineage
                
                return lineage
                
        except Exception as e:
            self.logger.error(f"Failed to load lineage for dataset '{dataset_id}': {str(e)}")
            return None
    
    async def _save_lineage(self, lineage: DataLineage) -> None:
        """Save lineage information to file."""
        
        lineage_file = self.lineage_path / f"{lineage.dataset_id}.json"
        
        lineage_data = {
            'dataset_id': lineage.dataset_id,
            'upstream_datasets': lineage.upstream_datasets,
            'downstream_datasets': lineage.downstream_datasets,
            'transformations': lineage.transformations,
            'created_at': lineage.created_at.isoformat(),
            'updated_at': lineage.updated_at.isoformat()
        }
        
        async with aiofiles.open(lineage_file, 'w') as f:
            await f.write(json.dumps(lineage_data, indent=2))
    
    async def cleanup_old_versions(
        self,
        dataset_id: str,
        keep_versions: int = 10,
        keep_days: int = 30
    ) -> int:
        """Clean up old versions based on retention policy."""
        
        try:
            versions = await self.list_versions(dataset_id)
            
            if len(versions) <= keep_versions:
                return 0
            
            # Determine versions to delete
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            
            versions_to_delete = []
            
            # Keep the most recent versions
            versions_by_date = sorted(versions, key=lambda v: v.created_at, reverse=True)
            
            for i, version in enumerate(versions_by_date):
                if i >= keep_versions and version.created_at < cutoff_date:
                    versions_to_delete.append(version)
            
            # Delete old versions
            deleted_count = 0
            for version in versions_to_delete:
                if await self.delete_version(version.version_id, "system"):
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old versions for dataset '{dataset_id}'")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old versions: {str(e)}")
            return 0