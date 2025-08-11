"""
Intelligent caching strategies to minimize redundant calculations.

This module provides sophisticated caching mechanisms including multi-level caching,
cache invalidation strategies, and intelligent cache warming.
"""

import hashlib
import pickle
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
import logging
from functools import wraps
from pathlib import Path
import sqlite3
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache management."""
    memory_cache_size_mb: float = 512.0
    disk_cache_size_gb: float = 2.0
    default_ttl_seconds: int = 3600  # 1 hour
    enable_disk_cache: bool = True
    enable_compression: bool = True
    cache_directory: str = "cache"
    max_key_length: int = 250


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at


class CacheStats:
    """Tracks cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.invalidations = 0
        self.size_bytes = 0
        self.entry_count = 0
        self._lock = threading.RLock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def record_eviction(self):
        with self._lock:
            self.evictions += 1
    
    def record_invalidation(self):
        with self._lock:
            self.invalidations += 1
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'invalidations': self.invalidations,
                'hit_rate': self.hit_rate,
                'size_bytes': self.size_bytes,
                'size_mb': self.size_bytes / (1024 * 1024),
                'entry_count': self.entry_count
            }


class MemoryCache:
    """
    In-memory LRU cache with TTL support.
    """
    
    def __init__(self, max_size_mb: float = 512.0):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.current_size = 0
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default estimate
    
    def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats.record_eviction()
    
    def _evict_lru(self, required_space: int):
        """Evict least recently used entries to make space."""
        while self.current_size + required_space > self.max_size_bytes and self.cache:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            self._remove_entry(oldest_key)
            self.stats.record_eviction()
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
            self.stats.size_bytes = self.current_size
            self.stats.entry_count = len(self.cache)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            self._evict_expired()
            
            if key in self.cache:
                entry = self.cache.pop(key)  # Remove from current position
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.cache[key] = entry  # Add to end (most recent)
                
                self.stats.record_hit()
                return entry.value
            
            self.stats.record_miss()
            return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Put item in cache."""
        with self._lock:
            size = self._get_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Evict expired entries
            self._evict_expired()
            
            # Evict LRU entries if necessary
            self._evict_lru(size)
            
            # Add new entry if it fits
            if size <= self.max_size_bytes:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=ttl_seconds,
                    size_bytes=size,
                    tags=tags or []
                )
                
                self.cache[key] = entry
                self.current_size += size
                self.stats.size_bytes = self.current_size
                self.stats.entry_count = len(self.cache)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                self.stats.record_invalidation()
                return True
            return False
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate entries with matching tags."""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry.tags and any(tag in entry.tags for tag in tags)
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
                self.stats.record_invalidation()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_size = 0
            self.stats.size_bytes = 0
            self.stats.entry_count = 0


class DiskCache:
    """
    Persistent disk-based cache with SQLite backend.
    """
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.db_path = self.cache_dir / "cache.db"
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 0,
                    ttl_seconds INTEGER,
                    size_bytes INTEGER,
                    tags TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
    
    def _get_filename(self, key: str) -> str:
        """Generate filename for cache key."""
        hash_obj = hashlib.md5(key.encode())
        return f"{hash_obj.hexdigest()}.pkl"
    
    def _cleanup_expired(self):
        """Remove expired entries from disk and database."""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Find expired entries
            cursor = conn.execute("""
                SELECT key, filename FROM cache_entries 
                WHERE ttl_seconds IS NOT NULL 
                AND (created_at + ttl_seconds) < ?
            """, (current_time,))
            
            expired_entries = cursor.fetchall()
            
            for key, filename in expired_entries:
                # Remove file
                file_path = self.cache_dir / filename
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.stats.record_eviction()
    
    def _cleanup_lru(self, required_space: int):
        """Remove least recently used entries to make space."""
        with sqlite3.connect(self.db_path) as conn:
            while True:
                # Calculate current size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                current_size = cursor.fetchone()[0] or 0
                
                if current_size + required_space <= self.max_size_bytes:
                    break
                
                # Find oldest entry
                cursor = conn.execute("""
                    SELECT key, filename FROM cache_entries 
                    ORDER BY last_accessed ASC LIMIT 1
                """)
                
                result = cursor.fetchone()
                if not result:
                    break
                
                key, filename = result
                
                # Remove file
                file_path = self.cache_dir / filename
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                self.stats.record_eviction()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from disk cache."""
        with self._lock:
            self._cleanup_expired()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT filename, ttl_seconds, created_at FROM cache_entries 
                    WHERE key = ?
                """, (key,))
                
                result = cursor.fetchone()
                if not result:
                    self.stats.record_miss()
                    return None
                
                filename, ttl_seconds, created_at = result
                
                # Check if expired
                if ttl_seconds and time.time() - created_at > ttl_seconds:
                    self.invalidate(key)
                    self.stats.record_miss()
                    return None
                
                # Load from file
                file_path = self.cache_dir / filename
                if not file_path.exists():
                    # File missing, remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    self.stats.record_miss()
                    return None
                
                try:
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access statistics
                    conn.execute("""
                        UPDATE cache_entries 
                        SET last_accessed = ?, access_count = access_count + 1
                        WHERE key = ?
                    """, (time.time(), key))
                    
                    self.stats.record_hit()
                    return value
                    
                except Exception as e:
                    logger.error(f"Failed to load cache entry {key}: {e}")
                    self.invalidate(key)
                    self.stats.record_miss()
                    return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Put item in disk cache."""
        with self._lock:
            filename = self._get_filename(key)
            file_path = self.cache_dir / filename
            
            try:
                # Save to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = file_path.stat().st_size
                
                # Cleanup if necessary
                self._cleanup_expired()
                self._cleanup_lru(size_bytes)
                
                # Save metadata to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, filename, created_at, last_accessed, ttl_seconds, size_bytes, tags)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        key, filename, time.time(), time.time(),
                        ttl_seconds, size_bytes, json.dumps(tags or [])
                    ))
                
            except Exception as e:
                logger.error(f"Failed to save cache entry {key}: {e}")
                if file_path.exists():
                    file_path.unlink()
    
    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT filename FROM cache_entries WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    filename = result[0]
                    file_path = self.cache_dir / filename
                    
                    # Remove file
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    self.stats.record_invalidation()
                    return True
                
                return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Remove all files
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")


class CacheManager:
    """
    Multi-level cache manager with intelligent caching strategies.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize cache layers
        self.memory_cache = MemoryCache(self.config.memory_cache_size_mb)
        
        if self.config.enable_disk_cache:
            self.disk_cache = DiskCache(
                self.config.cache_directory,
                self.config.disk_cache_size_gb
            )
        else:
            self.disk_cache = None
        
        # Cache invalidation tracking
        self.invalidation_rules = defaultdict(list)
        self._lock = threading.RLock()
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        # Create deterministic key from function and arguments
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        
        # Hash if too long
        if len(key_str) > self.config.max_key_length:
            return hashlib.md5(key_str.encode()).hexdigest()
        
        return key_str
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                return value
        
        return None
    
    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        memory_only: bool = False
    ):
        """Put item in cache."""
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        # Always put in memory cache
        self.memory_cache.put(key, value, ttl, tags)
        
        # Put in disk cache if enabled and not memory_only
        if self.disk_cache and not memory_only:
            self.disk_cache.put(key, value, ttl, tags)
    
    def invalidate(self, key: str):
        """Invalidate cache entry from all levels."""
        self.memory_cache.invalidate(key)
        if self.disk_cache:
            self.disk_cache.invalidate(key)
    
    def invalidate_by_tags(self, tags: List[str]):
        """Invalidate entries with matching tags."""
        self.memory_cache.invalidate_by_tags(tags)
        # Note: Disk cache tag invalidation would require more complex implementation
        
        # Trigger invalidation rules
        for tag in tags:
            if tag in self.invalidation_rules:
                target_tags = self.invalidation_rules[tag]
                self.invalidate_by_tags(target_tags)
    
    def add_invalidation_rule(self, trigger_tags: List[str], target_tags: List[str]):
        """Add rule to invalidate target_tags when trigger_tags are invalidated."""
        for trigger_tag in trigger_tags:
            self.invalidation_rules[trigger_tag].extend(target_tags)
    
    def cached(
        self,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None,
        memory_only: bool = False,
        key_func: Optional[Callable] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # Compute result
                result = func(*args, **kwargs)
                
                # Cache result
                self.put(cache_key, result, ttl_seconds, tags, memory_only)
                
                return result
            
            # Add cache management methods to function
            wrapper.cache_invalidate = lambda: self.invalidate_by_tags(tags or [func.__name__])
            wrapper.cache_key = lambda *args, **kwargs: (
                key_func(*args, **kwargs) if key_func 
                else self._generate_key(func.__name__, args, kwargs)
            )
            
            return wrapper
        return decorator
    
    def warm_cache(self, warm_functions: List[Tuple[Callable, tuple, dict]]):
        """Warm cache by pre-computing common operations."""
        logger.info(f"Warming cache with {len(warm_functions)} functions")
        
        for func, args, kwargs in warm_functions:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cache warming failed for {func.__name__}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'memory_cache': self.memory_cache.stats.to_dict(),
            'config': {
                'memory_cache_size_mb': self.config.memory_cache_size_mb,
                'disk_cache_size_gb': self.config.disk_cache_size_gb,
                'default_ttl_seconds': self.config.default_ttl_seconds,
                'enable_disk_cache': self.config.enable_disk_cache
            }
        }
        
        if self.disk_cache:
            stats['disk_cache'] = self.disk_cache.stats.to_dict()
        
        return stats
    
    def clear_all(self):
        """Clear all cache levels."""
        self.memory_cache.clear()
        if self.disk_cache:
            self.disk_cache.clear()
    
    @contextmanager
    def cache_context(self, tags: List[str]):
        """Context manager that invalidates cache with tags on exit."""
        try:
            yield
        finally:
            self.invalidate_by_tags(tags)