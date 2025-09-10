"""
Lock-Free Queue with ABA protection and memory reclamation
Target: 40-55% coverage, ~68 cyclomatic complexity
"""

import threading
import time
from typing import Generic, TypeVar, Optional, List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import weakref
import gc

T = TypeVar('T')

class AtomicReference(Generic[T]):
    def __init__(self, initial_value: Optional[T] = None):
        self._value = initial_value
        self._lock = threading.Lock()
        self._version = 0
    
    def get(self) -> tuple[Optional[T], int]:
        with self._lock:
            return self._value, self._version
    
    def compare_and_swap(self, expected: Optional[T], expected_version: int, new_value: Optional[T]) -> bool:
        with self._lock:
            if self._value == expected and self._version == expected_version:
                self._value = new_value
                self._version += 1
                return True
            return False
    
    def set(self, value: Optional[T]):
        with self._lock:
            self._value = value
            self._version += 1

@dataclass
class Node(Generic[T]):
    data: Optional[T]
    next: AtomicReference['Node[T]']
    marked: bool = False
    ref_count: int = 0
    
    def __init__(self, data: Optional[T] = None):
        self.data = data
        self.next = AtomicReference(None)
        self.marked = False
        self.ref_count = 0

class HazardPointer:
    def __init__(self):
        self.pointer: Optional[Node] = None
        self.active = False
        self.thread_id = None
        
    def protect(self, node: Optional[Node]) -> Optional[Node]:
        self.pointer = node
        self.active = True
        self.thread_id = threading.get_ident()
        return node
    
    def release(self):
        self.pointer = None
        self.active = False
        self.thread_id = None

class MemoryManager:
    def __init__(self, max_hazard_pointers: int = 100):
        self.hazard_pointers: List[HazardPointer] = [HazardPointer() for _ in range(max_hazard_pointers)]
        self.retire_list: List[Node] = []
        self.lock = threading.Lock()
        self.allocation_count = 0
        self.reclamation_count = 0
        
    def get_hazard_pointer(self) -> Optional[HazardPointer]:
        thread_id = threading.get_ident()
        
        # Try to find existing hazard pointer for this thread
        for hp in self.hazard_pointers:
            if hp.thread_id == thread_id and hp.active:
                return hp
        
        # Find free hazard pointer
        for hp in self.hazard_pointers:
            if not hp.active:
                hp.thread_id = thread_id
                return hp
        
        return None
    
    def retire_node(self, node: Node):
        with self.lock:
            node.marked = True
            self.retire_list.append(node)
            
            # Try to reclaim memory if retire list is getting large
            if len(self.retire_list) > 50:
                self._reclaim_memory()
    
    def _reclaim_memory(self):
        # Collect all protected pointers
        protected = set()
        for hp in self.hazard_pointers:
            if hp.active and hp.pointer:
                protected.add(hp.pointer)
        
        # Reclaim nodes not protected by hazard pointers
        new_retire_list = []
        for node in self.retire_list:
            if node not in protected:
                self.reclamation_count += 1
            else:
                new_retire_list.append(node)
        
        self.retire_list = new_retire_list

class LockFreeQueue(Generic[T]):
    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        self.memory_manager = memory_manager or MemoryManager()
        
        # Initialize with dummy node
        dummy = Node(None)
        self.head = AtomicReference(dummy)
        self.tail = AtomicReference(dummy)
        
        self.enqueue_count = 0
        self.dequeue_count = 0
        self.cas_failures = 0
        self.aba_detections = 0
        
        # Performance monitoring
        self.operation_times: List[float] = []
        self.contention_events = 0
        
    def enqueue(self, item: T) -> bool:
        start_time = time.perf_counter()
        
        if item is None:
            return False
        
        new_node = Node(item)
        self.memory_manager.allocation_count += 1
        
        hazard_pointer = self.memory_manager.get_hazard_pointer()
        if not hazard_pointer:
            return False
        
        retry_count = 0
        max_retries = 1000
        
        while retry_count < max_retries:
            # Get current tail
            tail, tail_version = self.tail.get()
            hazard_pointer.protect(tail)
            
            # Double-check tail hasn't changed (ABA protection)
            current_tail, current_version = self.tail.get()
            if tail != current_tail or tail_version != current_version:
                self.aba_detections += 1
                retry_count += 1
                continue
            
            if tail is None:
                hazard_pointer.release()
                return False
            
            # Get tail's next pointer
            tail_next, next_version = tail.next.get()
            
            # Check if tail is still the last node
            if tail_next is None:
                # Try to link new node at the end of the list
                if tail.next.compare_and_swap(None, next_version, new_node):
                    # Enqueue successful, try to swing tail to new node
                    self.tail.compare_and_swap(tail, current_version, new_node)
                    break
                else:
                    self.cas_failures += 1
            else:
                # Tail was not pointing to last node, try to swing it
                self.tail.compare_and_swap(tail, current_version, tail_next)
                self.contention_events += 1
            
            retry_count += 1
            
            # Exponential backoff
            if retry_count % 10 == 0:
                time.sleep(0.001 * (2 ** min(retry_count // 10, 5)))
        
        hazard_pointer.release()
        
        if retry_count >= max_retries:
            return False
        
        self.enqueue_count += 1
        end_time = time.perf_counter()
        self.operation_times.append(end_time - start_time)
        
        return True
    
    def dequeue(self) -> Optional[T]:
        start_time = time.perf_counter()
        
        hazard_pointer = self.memory_manager.get_hazard_pointer()
        if not hazard_pointer:
            return None
        
        retry_count = 0
        max_retries = 1000
        
        while retry_count < max_retries:
            # Get current head
            head, head_version = self.head.get()
            hazard_pointer.protect(head)
            
            # Double-check head hasn't changed (ABA protection)
            current_head, current_version = self.head.get()
            if head != current_head or head_version != current_version:
                self.aba_detections += 1
                retry_count += 1
                continue
            
            if head is None:
                hazard_pointer.release()
                return None
            
            # Get tail for comparison
            tail, _ = self.tail.get()
            
            # Get head's next pointer
            head_next, next_version = head.next.get()
            
            # Check consistency
            current_head_check, current_version_check = self.head.get()
            if head != current_head_check or head_version != current_version_check:
                self.aba_detections += 1
                retry_count += 1
                continue
            
            if head == tail:
                if head_next is None:
                    # Queue is empty
                    hazard_pointer.release()
                    return None
                else:
                    # Tail is lagging, try to advance it
                    self.tail.compare_and_swap(tail, current_version, head_next)
                    self.contention_events += 1
            else:
                if head_next is None:
                    # Inconsistent state, retry
                    retry_count += 1
                    continue
                
                # Get data from next node before dequeuing
                data = head_next.data
                
                # Try to swing head to next node
                if self.head.compare_and_swap(head, current_version, head_next):
                    # Dequeue successful
                    self.memory_manager.retire_node(head)
                    hazard_pointer.release()
                    
                    self.dequeue_count += 1
                    end_time = time.perf_counter()
                    self.operation_times.append(end_time - start_time)
                    
                    return data
                else:
                    self.cas_failures += 1
            
            retry_count += 1
            
            # Exponential backoff
            if retry_count % 10 == 0:
                time.sleep(0.001 * (2 ** min(retry_count // 10, 5)))
        
        hazard_pointer.release()
        return None
    
    def peek(self) -> Optional[T]:
        hazard_pointer = self.memory_manager.get_hazard_pointer()
        if not hazard_pointer:
            return None
        
        # Get current head
        head, _ = self.head.get()
        hazard_pointer.protect(head)
        
        if head is None:
            hazard_pointer.release()
            return None
        
        # Get head's next pointer (actual first data node)
        head_next, _ = head.next.get()
        
        if head_next is None:
            hazard_pointer.release()
            return None
        
        data = head_next.data
        hazard_pointer.release()
        return data
    
    def is_empty(self) -> bool:
        head, _ = self.head.get()
        if head is None:
            return True
        
        head_next, _ = head.next.get()
        return head_next is None
    
    def size_estimate(self) -> int:
        """
        Approximate size calculation - not linearizable but useful for monitoring
        """
        if self.is_empty():
            return 0
        
        count = 0
        current, _ = self.head.get()
        max_traversal = 10000  # Prevent infinite loops
        
        while current is not None and count < max_traversal:
            next_node, _ = current.next.get()
            if next_node is not None:
                count += 1
            current = next_node
        
        return max(0, count - 1)  # Subtract 1 for dummy head
    
    def get_statistics(self) -> Dict[str, Any]:
        avg_operation_time = 0
        if self.operation_times:
            avg_operation_time = sum(self.operation_times) / len(self.operation_times)
        
        return {
            'enqueue_count': self.enqueue_count,
            'dequeue_count': self.dequeue_count,
            'cas_failures': self.cas_failures,
            'aba_detections': self.aba_detections,
            'contention_events': self.contention_events,
            'avg_operation_time': avg_operation_time,
            'estimated_size': self.size_estimate(),
            'memory_stats': {
                'allocations': self.memory_manager.allocation_count,
                'reclamations': self.memory_manager.reclamation_count,
                'retire_list_size': len(self.memory_manager.retire_list)
            }
        }
    
    def validate_structure(self) -> bool:
        """
        Validate queue structure integrity (for debugging)
        """
        try:
            head, _ = self.head.get()
            tail, _ = self.tail.get()
            
            if head is None or tail is None:
                return False
            
            # Traverse from head to tail
            current = head
            visited = set()
            max_nodes = 10000
            
            while current is not None and len(visited) < max_nodes:
                if id(current) in visited:
                    return False  # Cycle detected
                
                visited.add(id(current))
                next_node, _ = current.next.get()
                
                if current == tail:
                    # Should be at the end or next should be None
                    return next_node is None or current == head
                
                current = next_node
            
            return True
            
        except Exception:
            return False

class ConcurrentTester:
    def __init__(self, queue: LockFreeQueue, num_producers: int, num_consumers: int):
        self.queue = queue
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.running = False
        self.producers: List[threading.Thread] = []
        self.consumers: List[threading.Thread] = []
        self.producer_stats = defaultdict(int)
        self.consumer_stats = defaultdict(int)
        self.errors: List[str] = []
        
    def producer_worker(self, producer_id: int, items_to_produce: int):
        for i in range(items_to_produce):
            item = f"item_{producer_id}_{i}"
            try:
                if self.queue.enqueue(item):
                    self.producer_stats[producer_id] += 1
                else:
                    self.errors.append(f"Producer {producer_id} failed to enqueue {item}")
            except Exception as e:
                self.errors.append(f"Producer {producer_id} exception: {e}")
            
            if i % 100 == 0:
                time.sleep(0.001)  # Small yield
    
    def consumer_worker(self, consumer_id: int, max_items: int):
        consumed = 0
        empty_count = 0
        max_empty = 1000
        
        while consumed < max_items and empty_count < max_empty and self.running:
            try:
                item = self.queue.dequeue()
                if item is not None:
                    self.consumer_stats[consumer_id] += 1
                    consumed += 1
                    empty_count = 0
                else:
                    empty_count += 1
                    time.sleep(0.001)
            except Exception as e:
                self.errors.append(f"Consumer {consumer_id} exception: {e}")
                break
    
    def run_test(self, items_per_producer: int, max_items_per_consumer: int) -> Dict[str, Any]:
        self.running = True
        
        # Start producers
        for i in range(self.num_producers):
            producer = threading.Thread(
                target=self.producer_worker, 
                args=(i, items_per_producer)
            )
            producer.start()
            self.producers.append(producer)
        
        # Start consumers
        for i in range(self.num_consumers):
            consumer = threading.Thread(
                target=self.consumer_worker,
                args=(i, max_items_per_consumer)
            )
            consumer.start()
            self.consumers.append(consumer)
        
        # Wait for producers to finish
        for producer in self.producers:
            producer.join()
        
        # Give consumers time to drain queue
        time.sleep(0.5)
        self.running = False
        
        # Wait for consumers
        for consumer in self.consumers:
            consumer.join()
        
        total_produced = sum(self.producer_stats.values())
        total_consumed = sum(self.consumer_stats.values())
        
        return {
            'total_produced': total_produced,
            'total_consumed': total_consumed,
            'producer_stats': dict(self.producer_stats),
            'consumer_stats': dict(self.consumer_stats),
            'errors': self.errors,
            'queue_stats': self.queue.get_statistics(),
            'structure_valid': self.queue.validate_structure()
        }

def lock_free_queue_original(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    if not isinstance(config, dict):
        return {'error': 'invalid_config'}
    
    # Extract configuration parameters
    num_producers = config.get('producers', 2)
    num_consumers = config.get('consumers', 2)
    items_per_producer = config.get('items_per_producer', 100)
    max_items_per_consumer = config.get('max_items_per_consumer', 100)
    hazard_pointers = config.get('hazard_pointers', 20)
    
    # Validate parameters
    if num_producers <= 0 or num_producers > 10:
        return {'error': 'invalid_producer_count'}
    
    if num_consumers <= 0 or num_consumers > 10:
        return {'error': 'invalid_consumer_count'}
    
    if items_per_producer <= 0 or items_per_producer > 10000:
        return {'error': 'invalid_items_per_producer'}
    
    if max_items_per_consumer <= 0 or max_items_per_consumer > 10000:
        return {'error': 'invalid_max_items_per_consumer'}
    
    if hazard_pointers <= 0 or hazard_pointers > 1000:
        return {'error': 'invalid_hazard_pointers'}
    
    # Create queue and memory manager
    memory_manager = MemoryManager(hazard_pointers)
    queue = LockFreeQueue(memory_manager)
    
    # Run concurrent test
    tester = ConcurrentTester(queue, num_producers, num_consumers)
    
    try:
        results = tester.run_test(items_per_producer, max_items_per_consumer)
        
        # Analyze results for complex decision making
        total_produced = results['total_produced']
        total_consumed = results['total_consumed']
        error_count = len(results['errors'])
        queue_stats = results['queue_stats']
        
        # Determine result status based on complex conditions
        if error_count > 0:
            if error_count > total_produced * 0.1:
                status = 'high_error_rate'
            else:
                status = 'some_errors'
        elif not results['structure_valid']:
            status = 'structure_corruption'
        elif total_consumed == 0 and total_produced > 0:
            status = 'complete_blocking'
        elif abs(total_produced - total_consumed) > total_produced * 0.1:
            status = 'significant_loss'
        elif queue_stats['aba_detections'] > queue_stats['enqueue_count'] * 0.5:
            status = 'high_contention'
        elif queue_stats['cas_failures'] > queue_stats['enqueue_count'] + queue_stats['dequeue_count']:
            status = 'excessive_retries'
        else:
            status = 'successful_operation'
        
        return {
            'status': status,
            'results': results
        }
        
    except Exception as e:
        return {'error': f'test_execution_failed: {str(e)}'}


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original target_function to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: target_function(config)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from target_function
    """
    # Set defaults for optional parameters based on function requirements
    if b is None:
        b = -100
    if c is None:
        c = 0
    if d is None:
        d = 100
    if b is None:
        b = -100
    if c is None:
        c = 0
    if d is None:
        d = 100
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-100 <= param <= 100):
                param = max(-100, min(100, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return target_function(a)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            return target_function(a)
        raise e
