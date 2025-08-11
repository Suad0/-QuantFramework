"""
Intelligent caching layer with TTL and invalidation strategies.
"""

import asyncio
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import logging
from pathlib import Path
import aiofiles
import redis.asyncio as redis


class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least recently used
    LFU = "lfu"  # Least frequently used
    MANUAL = "manual"  # Manual invalidation only


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> int:
        """Get age of cache entry in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for cache system."""
    strategy: CacheStrategy = CacheStrategy.TTL
    default_ttl_seconds: int = 3600  # 1 hour
    max_size: int = 1000  # Maximum number of entries
    max_memory_mb: int = 500  # Maximum memory usage
    cleanup_interval_seconds: int = 300  # 5 minutes
    persistence_enabled: bool = True
    persistence_path: str = "cache"
    redis_url: Optional[str] = None  # Redis connection string
    compression_enabled: bool = True
    compression_threshold_bytes: int = 1024  # Compress data larger than 1KB


class DataCache:
    """Intelligent data caching system with multiple backends."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()
        
        # Redis cache (optional)
        self._redis_client: Optional[redis.Redis] = None
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_bytes': 0
        }
        
        # Initialize flag
        self._initialized = False
    
    def initialize_sync(self) -> None:
        """Initialize cache system synchronously."""
        try:
            # For now, just initialize the basic components without async operations
            # Redis and persistent cache loading will be handled when first accessed
            self._initialized = True
            self.logger.info("Data cache system initialized (sync mode)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache system: {str(e)}")
    
    async def _initialize(self) -> None:
        """Initialize cache system asynchronously."""
        try:
            # Initialize Redis if configured
            if self.config.redis_url:
                self._redis_client = redis.from_url(self.config.redis_url)
                await self._redis_client.ping()
                self.logger.info("Redis cache backend initialized")
            
            # Load persistent cache if enabled
            if self.config.persistence_enabled:
                await self._load_persistent_cache()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._initialized = True
            self.logger.info("Data cache system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache system: {str(e)}")
    
    def get_sync(self, key: str) -> Optional[Any]:
        """Get data from cache synchronously (memory cache only)."""
        try:
            if not self._initialized:
                self.initialize_sync()
            
            # Only use memory cache in sync mode
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired:
                    del self._memory_cache[key]
                    self._stats['evictions'] += 1
                    self._stats['misses'] += 1
                    return None
                
                # Update access time
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._stats['hits'] += 1
                return entry.data
            
            self._stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            self._stats['misses'] += 1
            return None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        try:
            # Try Redis first if available
            if self._redis_client:
                data = await self._get_from_redis(key)
                if data is not None:
                    self._stats['hits'] += 1
                    return data
            
            # Try memory cache
            async with self._cache_lock:
                if key in self._memory_cache:
                    entry = self._memory_cache[key]
                    
                    # Check if expired
                    if entry.is_expired:
                        del self._memory_cache[key]
                        self._stats['evictions'] += 1
                        self._stats['misses'] += 1
                        return None
                    
                    # Update access info
                    entry.touch()
                    self._stats['hits'] += 1
                    
                    return entry.data
            
            self._stats['misses'] += 1
            return None
            
        except Exception as e:
            self.logger.error(f"Cache get error for key '{key}': {str(e)}")
            self._stats['misses'] += 1
            return None
    
    def set_sync(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set data in cache synchronously (memory cache only)."""
        try:
            if not self._initialized:
                self.initialize_sync()
            
            ttl = ttl_seconds or self.config.default_ttl_seconds
            metadata = metadata or {}
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                metadata=metadata
            )
            
            # Store in memory cache only (sync mode)
            # Check memory limits before adding
            if len(self._memory_cache) >= self.config.max_size:
                self._evict_entries_sync()
            
            self._memory_cache[key] = entry
            self._update_memory_stats_sync()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error for key '{key}': {str(e)}")
            return False
    
    async def set(
        self,
        key: str,
        data: Any,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set data in cache."""
        try:
            ttl = ttl_seconds or self.config.default_ttl_seconds
            metadata = metadata or {}
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl_seconds=ttl,
                metadata=metadata
            )
            
            # Store in Redis if available
            if self._redis_client:
                await self._set_in_redis(key, entry, ttl)
            
            # Store in memory cache
            async with self._cache_lock:
                # Check memory limits before adding
                if len(self._memory_cache) >= self.config.max_size:
                    await self._evict_entries()
                
                self._memory_cache[key] = entry
                await self._update_memory_stats()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cache set error for key '{key}': {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete data from cache."""
        try:
            deleted = False
            
            # Delete from Redis
            if self._redis_client:
                result = await self._redis_client.delete(key)
                deleted = deleted or bool(result)
            
            # Delete from memory cache
            async with self._cache_lock:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    deleted = True
                    await self._update_memory_stats()
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Cache delete error for key '{key}': {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache data."""
        try:
            # Clear Redis
            if self._redis_client:
                await self._redis_client.flushdb()
            
            # Clear memory cache
            async with self._cache_lock:
                self._memory_cache.clear()
                await self._update_memory_stats()
            
            self.logger.info("Cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {str(e)}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            deleted_count = 0
            
            # Redis pattern deletion
            if self._redis_client:
                keys = await self._redis_client.keys(pattern)
                if keys:
                    deleted_count += await self._redis_client.delete(*keys)
            
            # Memory cache pattern deletion
            async with self._cache_lock:
                keys_to_delete = [
                    key for key in self._memory_cache.keys()
                    if self._matches_pattern(key, pattern)
                ]
                
                for key in keys_to_delete:
                    del self._memory_cache[key]
                    deleted_count += 1
                
                if keys_to_delete:
                    await self._update_memory_stats()
            
            self.logger.info(f"Invalidated {deleted_count} cache entries matching pattern '{pattern}'")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Cache pattern invalidation error: {str(e)}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._cache_lock:
            memory_entries = len(self._memory_cache)
            
            # Calculate hit rate
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                **self._stats,
                'hit_rate_percent': round(hit_rate, 2),
                'memory_entries': memory_entries,
                'redis_available': self._redis_client is not None
            }
            
            # Add Redis stats if available
            if self._redis_client:
                try:
                    redis_info = await self._redis_client.info('memory')
                    stats['redis_memory_usage_bytes'] = redis_info.get('used_memory', 0)
                except:
                    pass
            
            return stats
    
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        try:
            data = await self._redis_client.get(key)
            if data is None:
                return None
            
            # Deserialize data
            if self.config.compression_enabled:
                import zlib
                data = zlib.decompress(data)
            
            entry_dict = pickle.loads(data)
            return entry_dict['data']
            
        except Exception as e:
            self.logger.warning(f"Redis get error: {str(e)}")
            return None
    
    async def _set_in_redis(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """Set data in Redis cache."""
        try:
            # Serialize entry
            entry_dict = {
                'data': entry.data,
                'created_at': entry.created_at.isoformat(),
                'metadata': entry.metadata
            }
            
            data = pickle.dumps(entry_dict)
            
            # Compress if enabled and data is large enough
            if self.config.compression_enabled and len(data) > self.config.compression_threshold_bytes:
                import zlib
                data = zlib.compress(data)
            
            # Set with TTL
            await self._redis_client.setex(key, ttl, data)
            
        except Exception as e:
            self.logger.warning(f"Redis set error: {str(e)}")
    
    async def _evict_entries(self) -> None:
        """Evict entries based on cache strategy."""
        if not self._memory_cache:
            return
        
        entries_to_remove = max(1, len(self._memory_cache) // 10)  # Remove 10%
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # TTL or default
            # Remove oldest entries
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].created_at
            )
        
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._memory_cache[key]
            self._stats['evictions'] += 1
        
        await self._update_memory_stats()
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_expired_entries()
                
                # Save persistent cache periodically
                if self.config.persistence_enabled:
                    await self._save_persistent_cache()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from memory cache."""
        async with self._cache_lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
                self._stats['evictions'] += 1
            
            if expired_keys:
                await self._update_memory_stats()
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _update_memory_stats(self) -> None:
        """Update memory usage statistics."""
        try:
            total_size = 0
            for entry in self._memory_cache.values():
                # Rough estimate of memory usage
                total_size += len(pickle.dumps(entry.data))
            
            self._stats['memory_usage_bytes'] = total_size
            
        except Exception as e:
            self.logger.warning(f"Failed to update memory stats: {str(e)}")
    
    async def _save_persistent_cache(self) -> None:
        """Save cache to persistent storage."""
        try:
            cache_dir = Path(self.config.persistence_path)
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / "cache_data.pkl"
            
            async with self._cache_lock:
                # Only save non-expired entries
                valid_entries = {
                    key: entry for key, entry in self._memory_cache.items()
                    if not entry.is_expired
                }
            
            if valid_entries:
                async with aiofiles.open(cache_file, 'wb') as f:
                    await f.write(pickle.dumps(valid_entries))
                
                self.logger.debug(f"Saved {len(valid_entries)} cache entries to persistent storage")
            
        except Exception as e:
            self.logger.warning(f"Failed to save persistent cache: {str(e)}")
    
    async def _load_persistent_cache(self) -> None:
        """Load cache from persistent storage."""
        try:
            cache_file = Path(self.config.persistence_path) / "cache_data.pkl"
            
            if not cache_file.exists():
                return
            
            async with aiofiles.open(cache_file, 'rb') as f:
                data = await f.read()
                cached_entries = pickle.loads(data)
            
            async with self._cache_lock:
                # Load non-expired entries
                loaded_count = 0
                for key, entry in cached_entries.items():
                    if not entry.is_expired:
                        self._memory_cache[key] = entry
                        loaded_count += 1
                
                await self._update_memory_stats()
            
            self.logger.info(f"Loaded {loaded_count} cache entries from persistent storage")
            
        except Exception as e:
            self.logger.warning(f"Failed to load persistent cache: {str(e)}")
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a deterministic key from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def close(self) -> None:
        """Close cache system and cleanup resources."""
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Save persistent cache
            if self.config.persistence_enabled:
                await self._save_persistent_cache()
            
            # Close Redis connection
            if self._redis_client:
                await self._redis_client.close()
            
            self.logger.info("Cache system closed")
            
        except Exception as e:
            self.logger.error(f"Error closing cache system: {str(e)}")


    def _evict_entries_sync(self) -> None:
        """Evict entries synchronously."""
        if not self._memory_cache:
            return
        
        entries_to_remove = max(1, len(self._memory_cache) // 10)  # Remove 10%
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # TTL or default
            # Remove oldest entries
            sorted_entries = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1].created_at
            )
        
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._memory_cache[key]
            self._stats['evictions'] += 1
    
    def _update_memory_stats_sync(self) -> None:
        """Update memory statistics synchronously."""
        total_size = 0
        for entry in self._memory_cache.values():
            try:
                total_size += len(pickle.dumps(entry.data))
            except:
                total_size += 1024  # Estimate if pickle fails
        
        self._stats['memory_usage_bytes'] = total_size
    
    def invalidate_pattern_sync(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern synchronously."""
        try:
            if not self._initialized:
                self.initialize_sync()
            
            removed_count = 0
            keys_to_remove = []
            
            for key in self._memory_cache.keys():
                if self._matches_pattern(key, pattern):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._memory_cache[key]
                removed_count += 1
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Cache pattern invalidation error: {str(e)}")
            return 0
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """Get cache statistics synchronously."""
        try:
            if not self._initialized:
                self.initialize_sync()
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'memory_usage_bytes': self._stats['memory_usage_bytes'],
                'memory_usage_mb': self._stats['memory_usage_bytes'] / (1024 * 1024),
                'cache_size': len(self._memory_cache),
                'hit_rate': self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) if (self._stats['hits'] + self._stats['misses']) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {str(e)}")
            return {}
    
    def cleanup_expired_entries_sync(self) -> int:
        """Clean up expired entries synchronously."""
        try:
            if not self._initialized:
                self.initialize_sync()
            
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                self._stats['evictions'] += 1
            
            return len(expired_keys)
            
        except Exception as e:
            self.logger.error(f"Cache cleanup error: {str(e)}")
            return 0
    
    def close_sync(self) -> None:
        """Close cache synchronously."""
        try:
            # Cancel cleanup task if running
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            # Clear memory cache
            self._memory_cache.clear()
            
            # Close Redis connection if exists
            if self._redis_client:
                # Note: Redis client close is async, but we'll skip it in sync mode
                self._redis_client = None
            
            self.logger.info("Cache closed successfully (sync)")
            
        except Exception as e:
            self.logger.error(f"Error closing cache: {str(e)}")


class CacheDecorator:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache: DataCache,
        ttl_seconds: Optional[int] = None,
        key_prefix: str = "",
        invalidate_on_error: bool = False
    ):
        self.cache = cache
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.invalidate_on_error = invalidate_on_error
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{self.key_prefix}:{func.__name__}:{self.cache.generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.cache.set(cache_key, result, self.ttl_seconds)
                
                return result
                
            except Exception as e:
                # Optionally invalidate cache on error
                if self.invalidate_on_error:
                    await self.cache.delete(cache_key)
                raise e
        
        return wrapper