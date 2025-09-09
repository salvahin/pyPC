"""
Cache Manager with LRU, LFU, and adaptive replacement policies
Target: 30-45% coverage, ~60 cyclomatic complexity
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict
import threading
import hashlib

class CacheEntry:
    def __init__(self, key: str, value: Any, size: int):
        self.key = key
        self.value = value
        self.size = size
        self.access_count = 0
        self.last_access = time.time()
        self.creation_time = time.time()
        self.dirty = False
        
class CacheManager:
    def __init__(self, max_size: int = 1000, policy: str = "adaptive"):
        self.max_size = max_size
        self.current_size = 0
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = {}
        self.lru_order = OrderedDict()
        self.access_frequencies = {}
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self.ghost_cache = {}  # For adaptive replacement
        self.partition_point = max_size // 2
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
                
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access = time.time()
            self.hit_count += 1
            
            if self.policy == "lru":
                self._update_lru_access(key)
            elif self.policy == "lfu":
                self._update_lfu_access(key)
            elif self.policy == "adaptive":
                self._update_adaptive_access(key)
            
            if entry.dirty and self._should_write_through():
                self._write_through(key, entry.value)
                
            return entry.value
    
    def put(self, key: str, value: Any, size: int = 1) -> bool:
        with self.lock:
            if size > self.max_size:
                return False
                
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                size_diff = size - old_entry.size
                
                if self.current_size + size_diff <= self.max_size:
                    old_entry.value = value
                    old_entry.size = size
                    old_entry.dirty = True
                    self.current_size += size_diff
                    return True
                else:
                    # Need to evict to make room for larger value
                    self._evict_entry(key)
            
            # Make room for new entry
            while self.current_size + size > self.max_size:
                if not self._evict_one():
                    return False
            
            # Create new entry
            entry = CacheEntry(key, value, size)
            entry.dirty = True
            self.cache[key] = entry
            self.current_size += size
            
            if self.policy == "lru":
                self.lru_order[key] = True
            elif self.policy == "lfu":
                self.access_frequencies[key] = 1
            elif self.policy == "adaptive":
                self._adaptive_insert(key)
                
            return True
    
    def _evict_one(self) -> bool:
        if not self.cache:
            return False
            
        if self.policy == "lru":
            victim_key = self._find_lru_victim()
        elif self.policy == "lfu":
            victim_key = self._find_lfu_victim()
        elif self.policy == "adaptive":
            victim_key = self._find_adaptive_victim()
        else:
            victim_key = next(iter(self.cache))
            
        if victim_key:
            return self._evict_entry(victim_key)
        return False
    
    def _evict_entry(self, key: str) -> bool:
        if key not in self.cache:
            return False
            
        entry = self.cache[key]
        
        # Write back if dirty and write-back policy
        if entry.dirty and self._should_write_back():
            if not self._write_back(key, entry.value):
                return False
        
        # Update ghost cache for adaptive policy
        if self.policy == "adaptive":
            self.ghost_cache[key] = {
                'eviction_time': time.time(),
                'access_count': entry.access_count,
                'size': entry.size
            }
            
            # Limit ghost cache size
            if len(self.ghost_cache) > self.max_size:
                oldest_ghost = min(self.ghost_cache.keys(), 
                                 key=lambda k: self.ghost_cache[k]['eviction_time'])
                del self.ghost_cache[oldest_ghost]
        
        # Remove from data structures
        self.current_size -= entry.size
        del self.cache[key]
        
        if key in self.lru_order:
            del self.lru_order[key]
        if key in self.access_frequencies:
            del self.access_frequencies[key]
            
        self.eviction_count += 1
        return True
    
    def _find_lru_victim(self) -> Optional[str]:
        if not self.lru_order:
            return None
        return next(iter(self.lru_order))
    
    def _find_lfu_victim(self) -> Optional[str]:
        if not self.access_frequencies:
            return None
        return min(self.access_frequencies.keys(), 
                  key=lambda k: (self.access_frequencies[k], 
                               self.cache[k].creation_time))
    
    def _find_adaptive_victim(self) -> Optional[str]:
        if not self.cache:
            return None
            
        # ARC-like adaptive replacement
        lru_candidates = []
        lfu_candidates = []
        
        for key, entry in self.cache.items():
            if len(lru_candidates) < self.partition_point:
                lru_candidates.append((key, entry.last_access))
            else:
                lfu_candidates.append((key, entry.access_count))
        
        # Choose victim based on hit rates in each partition
        if self._should_evict_from_lru_partition():
            if lru_candidates:
                return min(lru_candidates, key=lambda x: x[1])[0]
        
        if lfu_candidates:
            return min(lfu_candidates, key=lambda x: x[1])[0]
        
        if lru_candidates:
            return min(lru_candidates, key=lambda x: x[1])[0]
            
        return None
    
    def _should_evict_from_lru_partition(self) -> bool:
        # Complex heuristic based on hit rates and ghost cache
        lru_hits = sum(1 for k in list(self.cache.keys())[:self.partition_point] 
                      if self.cache[k].access_count > 0)
        total_lru = min(len(self.cache), self.partition_point)
        
        if total_lru == 0:
            return True
            
        lru_hit_rate = lru_hits / total_lru
        
        # Check ghost cache influence
        ghost_influence = len([k for k in self.ghost_cache 
                             if time.time() - self.ghost_cache[k]['eviction_time'] < 3600])
        
        return lru_hit_rate < 0.5 or ghost_influence > self.partition_point * 0.3
    
    def _update_lru_access(self, key: str):
        if key in self.lru_order:
            del self.lru_order[key]
        self.lru_order[key] = True
    
    def _update_lfu_access(self, key: str):
        self.access_frequencies[key] = self.access_frequencies.get(key, 0) + 1
    
    def _update_adaptive_access(self, key: str):
        self._update_lru_access(key)
        self._update_lfu_access(key)
        
        # Adjust partition point based on performance
        if self.hit_count % 100 == 0 and self.hit_count > 0:
            self._adjust_partition_point()
    
    def _adaptive_insert(self, key: str):
        self.lru_order[key] = True
        self.access_frequencies[key] = 1
        
        # Check if this was recently evicted
        if key in self.ghost_cache:
            ghost_entry = self.ghost_cache[key]
            time_since_eviction = time.time() - ghost_entry['eviction_time']
            
            # If recently evicted, adjust strategy
            if time_since_eviction < 1800:  # 30 minutes
                self.partition_point = min(self.max_size - 1, 
                                         self.partition_point + 1)
            
            del self.ghost_cache[key]
    
    def _adjust_partition_point(self):
        # Adaptive partition point adjustment based on hit patterns
        recent_keys = list(self.cache.keys())[-100:] if len(self.cache) >= 100 else list(self.cache.keys())
        
        lru_performance = 0
        lfu_performance = 0
        
        for key in recent_keys:
            entry = self.cache[key]
            age = time.time() - entry.creation_time
            
            if age > 3600 and entry.access_count > 5:  # Old but frequently accessed
                lfu_performance += 1
            elif age < 300 and entry.access_count <= 2:  # New and infrequently accessed
                lru_performance += 1
        
        if lfu_performance > lru_performance * 1.5:
            self.partition_point = max(10, self.partition_point - 5)
        elif lru_performance > lfu_performance * 1.5:
            self.partition_point = min(self.max_size - 10, self.partition_point + 5)
    
    def _should_write_through(self) -> bool:
        # Complex decision based on access patterns and system load
        return (self.hit_count % 10 == 0 and 
                self.current_size > self.max_size * 0.8)
    
    def _should_write_back(self) -> bool:
        # Different conditions for write-back
        return self.current_size > self.max_size * 0.9
    
    def _write_through(self, key: str, value: Any) -> bool:
        # Simulate write-through to persistent storage
        checksum = hashlib.md5(str(value).encode()).hexdigest()
        return len(checksum) == 32  # Simple success condition
    
    def _write_back(self, key: str, value: Any) -> bool:
        # Simulate write-back with potential failure
        checksum = hashlib.md5(f"{key}:{value}".encode()).hexdigest()
        # Simulate occasional write failures
        return int(checksum[-1], 16) > 2
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'eviction_count': self.eviction_count,
                'current_size': self.current_size,
                'cache_entries': len(self.cache),
                'partition_point': self.partition_point,
                'ghost_entries': len(self.ghost_cache)
            }
    
    def invalidate(self, pattern: str = None) -> int:
        with self.lock:
            if pattern is None:
                count = len(self.cache)
                self.cache.clear()
                self.lru_order.clear()
                self.access_frequencies.clear()
                self.current_size = 0
                return count
            
            # Pattern-based invalidation
            keys_to_remove = []
            for key in self.cache:
                if pattern in key or self._matches_complex_pattern(key, pattern):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._evict_entry(key)
                
            return len(keys_to_remove)
    
    def _matches_complex_pattern(self, key: str, pattern: str) -> bool:
        # Complex pattern matching with multiple conditions
        if '*' in pattern:
            pattern_parts = pattern.split('*')
            if len(pattern_parts) == 2:
                prefix, suffix = pattern_parts
                return key.startswith(prefix) and key.endswith(suffix)
        
        if ':' in pattern:
            # Namespace-based matching
            namespace = pattern.split(':')[0]
            return key.startswith(f"{namespace}:")
        
        return False

def target_function(cache_size: int, operations: List[Tuple[str, str, Any]]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    if cache_size <= 0 or cache_size > 10000:
        return {'error': 'invalid_cache_size'}
    
    if not operations or len(operations) > 1000:
        return {'error': 'invalid_operations_count'}
    
    # Determine cache policy based on operation patterns
    policy = "lru"
    if len(operations) > 100:
        get_ops = sum(1 for op in operations if op[0] == 'get')
        put_ops = sum(1 for op in operations if op[0] == 'put')
        
        if put_ops > get_ops * 2:
            policy = "lfu"
        elif get_ops > put_ops * 3:
            policy = "adaptive"
    
    cache = CacheManager(cache_size, policy)
    results = []
    
    for i, (operation, key, value) in enumerate(operations):
        if not isinstance(key, str) or len(key) > 100:
            results.append({'error': f'invalid_key_at_{i}'})
            continue
            
        if operation == 'get':
            result = cache.get(key)
            results.append({'operation': 'get', 'key': key, 'result': result})
        elif operation == 'put':
            if value is None:
                results.append({'error': f'null_value_at_{i}'})
                continue
            
            size = len(str(value)) if isinstance(value, (str, int, float)) else 1
            success = cache.put(key, value, size)
            results.append({'operation': 'put', 'key': key, 'success': success})
        elif operation == 'invalidate':
            count = cache.invalidate(key if key != '*' else None)
            results.append({'operation': 'invalidate', 'count': count})
        else:
            results.append({'error': f'unknown_operation_at_{i}'})
    
    stats = cache.get_stats()
    
    # Complex result analysis
    if stats['hit_rate'] > 0.9 and len(results) > 50:
        return {'status': 'high_performance', 'stats': stats, 'results': results}
    elif stats['eviction_count'] > cache_size * 2:
        return {'status': 'high_churn', 'stats': stats, 'results': results}
    elif stats['hit_rate'] < 0.1 and stats['hit_count'] > 10:
        return {'status': 'poor_locality', 'stats': stats, 'results': results}
    else:
        return {'status': 'normal', 'stats': stats, 'results': results}