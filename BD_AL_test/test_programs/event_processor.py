"""
Event Processing System with complex routing and filtering
Target: 35-50% coverage, ~65 cyclomatic complexity
"""

import time
import threading
from typing import Dict, List, Any, Callable, Optional, Set
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

class EventType(Enum):
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"
    METRIC_EVENT = "metric_event"
    SECURITY_EVENT = "security_event"

class EventPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    event_id: str
    event_type: EventType
    priority: EventPriority
    payload: Dict[str, Any]
    timestamp: float
    source: str
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class EventFilter:
    def __init__(self, filter_id: str, conditions: Dict[str, Any]):
        self.filter_id = filter_id
        self.conditions = conditions
        self.active = True
        self.match_count = 0
        
    def matches(self, event: Event) -> bool:
        if not self.active:
            return False
            
        for key, condition in self.conditions.items():
            if not self._evaluate_condition(event, key, condition):
                return False
        
        self.match_count += 1
        return True
    
    def _evaluate_condition(self, event: Event, key: str, condition: Any) -> bool:
        if key == "event_type":
            if isinstance(condition, list):
                return event.event_type in condition
            return event.event_type == condition
        elif key == "priority":
            if isinstance(condition, dict):
                if "min" in condition and event.priority.value < condition["min"]:
                    return False
                if "max" in condition and event.priority.value > condition["max"]:
                    return False
                return True
            return event.priority == condition
        elif key == "source":
            if isinstance(condition, str) and condition.endswith("*"):
                return event.source.startswith(condition[:-1])
            return event.source == condition
        elif key == "payload":
            return self._matches_payload_condition(event.payload, condition)
        elif key == "age":
            event_age = time.time() - event.timestamp
            if isinstance(condition, dict):
                if "min" in condition and event_age < condition["min"]:
                    return False
                if "max" in condition and event_age > condition["max"]:
                    return False
                return True
            return event_age <= condition
        
        return True
    
    def _matches_payload_condition(self, payload: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        for key, expected in condition.items():
            if key not in payload:
                return False
            
            if isinstance(expected, dict) and "contains" in expected:
                if expected["contains"] not in str(payload[key]):
                    return False
            elif isinstance(expected, dict) and "range" in expected:
                try:
                    value = float(payload[key])
                    if value < expected["range"][0] or value > expected["range"][1]:
                        return False
                except (ValueError, TypeError):
                    return False
            else:
                if payload[key] != expected:
                    return False
        
        return True

class EventHandler:
    def __init__(self, handler_id: str, callback: Callable[[Event], bool], 
                 batch_size: int = 1, timeout: float = 0):
        self.handler_id = handler_id
        self.callback = callback
        self.batch_size = batch_size
        self.timeout = timeout
        self.processed_count = 0
        self.error_count = 0
        self.last_error = None
        self.active = True
        
    def process(self, events: List[Event]) -> bool:
        if not self.active:
            return False
            
        try:
            if self.batch_size == 1:
                success = self.callback(events[0])
            else:
                success = self.callback(events)
                
            if success:
                self.processed_count += len(events)
            else:
                self.error_count += 1
                
            return success
        except Exception as e:
            self.error_count += 1
            self.last_error = str(e)
            return False

class EventProcessor:
    def __init__(self, max_queue_size: int = 10000, worker_threads: int = 4):
        self.max_queue_size = max_queue_size
        self.worker_threads = worker_threads
        self.event_queue = deque()
        self.priority_queues = {priority: deque() for priority in EventPriority}
        self.dead_letter_queue = deque()
        
        self.filters: Dict[str, EventFilter] = {}
        self.handlers: Dict[str, EventHandler] = {}
        self.routes: Dict[str, Set[str]] = defaultdict(set)  # filter_id -> handler_ids
        
        self.running = False
        self.workers = []
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        
        self.stats = {
            'events_processed': 0,
            'events_dropped': 0,
            'events_retried': 0,
            'processing_errors': 0,
            'queue_overflows': 0
        }
        
        self.circuit_breakers = {}
        self.rate_limiters = {}
        
    def add_filter(self, filter_id: str, conditions: Dict[str, Any]) -> bool:
        if filter_id in self.filters:
            return False
            
        self.filters[filter_id] = EventFilter(filter_id, conditions)
        return True
    
    def add_handler(self, handler_id: str, callback: Callable[[Event], bool],
                   batch_size: int = 1, timeout: float = 0) -> bool:
        if handler_id in self.handlers:
            return False
            
        self.handlers[handler_id] = EventHandler(handler_id, callback, batch_size, timeout)
        return True
    
    def add_route(self, filter_id: str, handler_id: str) -> bool:
        if filter_id not in self.filters or handler_id not in self.handlers:
            return False
            
        self.routes[filter_id].add(handler_id)
        return True
    
    def submit_event(self, event: Event) -> bool:
        with self.lock:
            if len(self.event_queue) >= self.max_queue_size:
                self.stats['queue_overflows'] += 1
                
                # Try to make room by dropping low priority events
                if not self._make_queue_space():
                    self.stats['events_dropped'] += 1
                    return False
            
            # Apply rate limiting
            if not self._check_rate_limit(event.source):
                self.stats['events_dropped'] += 1
                return False
            
            # Add to appropriate priority queue
            self.priority_queues[event.priority].append(event)
            self.condition.notify()
            
            return True
    
    def _make_queue_space(self) -> bool:
        # Remove low priority events first
        for priority in [EventPriority.LOW, EventPriority.MEDIUM]:
            if self.priority_queues[priority]:
                dropped = self.priority_queues[priority].popleft()
                self.stats['events_dropped'] += 1
                return True
        return False
    
    def _check_rate_limit(self, source: str) -> bool:
        current_time = time.time()
        
        if source not in self.rate_limiters:
            self.rate_limiters[source] = {
                'count': 1,
                'window_start': current_time
            }
            return True
        
        limiter = self.rate_limiters[source]
        
        # Reset window if needed (1 second window)
        if current_time - limiter['window_start'] > 1.0:
            limiter['count'] = 1
            limiter['window_start'] = current_time
            return True
        
        # Check if under limit (100 events per second per source)
        if limiter['count'] < 100:
            limiter['count'] += 1
            return True
        
        return False
    
    def start(self):
        if self.running:
            return
            
        self.running = True
        for i in range(self.worker_threads):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        self.running = False
        with self.condition:
            self.condition.notify_all()
        
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def _worker_loop(self, worker_id: int):
        batch_buffer = defaultdict(list)
        last_batch_time = defaultdict(float)
        
        while self.running:
            event = self._get_next_event()
            
            if not event:
                self._process_pending_batches(batch_buffer, last_batch_time, force=True)
                continue
            
            # Apply filters and route to handlers
            matched_handlers = self._find_matching_handlers(event)
            
            if not matched_handlers:
                self.dead_letter_queue.append(event)
                continue
            
            # Group events by handler for batching
            for handler_id in matched_handlers:
                handler = self.handlers[handler_id]
                
                if handler.batch_size == 1:
                    self._process_single_event(event, handler)
                else:
                    batch_buffer[handler_id].append(event)
                    
                    if len(batch_buffer[handler_id]) >= handler.batch_size:
                        self._process_batch(batch_buffer[handler_id], handler)
                        batch_buffer[handler_id].clear()
                        last_batch_time[handler_id] = time.time()
            
            # Process timeout-based batches
            self._process_pending_batches(batch_buffer, last_batch_time)
            
            self.stats['events_processed'] += 1
    
    def _get_next_event(self) -> Optional[Event]:
        with self.condition:
            # Check priority queues in order
            for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                           EventPriority.MEDIUM, EventPriority.LOW]:
                if self.priority_queues[priority]:
                    return self.priority_queues[priority].popleft()
            
            # Wait for new events
            if self.running:
                self.condition.wait(timeout=1.0)
            
            return None
    
    def _find_matching_handlers(self, event: Event) -> Set[str]:
        matched_handlers = set()
        
        for filter_id, event_filter in self.filters.items():
            if event_filter.matches(event):
                matched_handlers.update(self.routes[filter_id])
        
        return matched_handlers
    
    def _process_single_event(self, event: Event, handler: EventHandler) -> bool:
        # Check circuit breaker
        if not self._check_circuit_breaker(handler.handler_id):
            return False
        
        success = handler.process([event])
        
        if not success:
            self._handle_processing_failure(event, handler)
        else:
            self._reset_circuit_breaker(handler.handler_id)
        
        return success
    
    def _process_batch(self, events: List[Event], handler: EventHandler) -> bool:
        if not self._check_circuit_breaker(handler.handler_id):
            return False
        
        success = handler.process(events)
        
        if not success:
            for event in events:
                self._handle_processing_failure(event, handler)
        else:
            self._reset_circuit_breaker(handler.handler_id)
        
        return success
    
    def _process_pending_batches(self, batch_buffer: Dict[str, List[Event]], 
                               last_batch_time: Dict[str, float], force: bool = False):
        current_time = time.time()
        
        for handler_id, events in batch_buffer.items():
            if not events:
                continue
                
            handler = self.handlers[handler_id]
            time_since_last = current_time - last_batch_time.get(handler_id, current_time)
            
            if force or (handler.timeout > 0 and time_since_last >= handler.timeout):
                self._process_batch(events, handler)
                events.clear()
                last_batch_time[handler_id] = current_time
    
    def _handle_processing_failure(self, event: Event, handler: EventHandler):
        self.stats['processing_errors'] += 1
        self._record_circuit_breaker_failure(handler.handler_id)
        
        if event.retry_count < event.max_retries:
            event.retry_count += 1
            self.stats['events_retried'] += 1
            
            # Add back to queue with delay
            delayed_event = Event(
                event_id=event.event_id,
                event_type=event.event_type,
                priority=EventPriority.LOW,  # Lower priority for retries
                payload=event.payload,
                timestamp=time.time() + (2 ** event.retry_count),  # Exponential backoff
                source=event.source,
                correlation_id=event.correlation_id,
                retry_count=event.retry_count,
                max_retries=event.max_retries
            )
            
            self.priority_queues[EventPriority.LOW].append(delayed_event)
        else:
            self.dead_letter_queue.append(event)
    
    def _check_circuit_breaker(self, handler_id: str) -> bool:
        if handler_id not in self.circuit_breakers:
            return True
            
        breaker = self.circuit_breakers[handler_id]
        current_time = time.time()
        
        if breaker['state'] == 'open':
            if current_time - breaker['last_failure'] > breaker['timeout']:
                breaker['state'] = 'half_open'
                breaker['consecutive_failures'] = 0
                return True
            return False
        
        return True
    
    def _record_circuit_breaker_failure(self, handler_id: str):
        if handler_id not in self.circuit_breakers:
            self.circuit_breakers[handler_id] = {
                'consecutive_failures': 0,
                'last_failure': time.time(),
                'state': 'closed',
                'timeout': 60.0  # 1 minute
            }
        
        breaker = self.circuit_breakers[handler_id]
        breaker['consecutive_failures'] += 1
        breaker['last_failure'] = time.time()
        
        if breaker['consecutive_failures'] >= 5:
            breaker['state'] = 'open'
    
    def _reset_circuit_breaker(self, handler_id: str):
        if handler_id in self.circuit_breakers:
            self.circuit_breakers[handler_id]['consecutive_failures'] = 0
            self.circuit_breakers[handler_id]['state'] = 'closed'
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            queue_sizes = {
                priority.name: len(queue) 
                for priority, queue in self.priority_queues.items()
            }
            
            handler_stats = {
                handler_id: {
                    'processed': handler.processed_count,
                    'errors': handler.error_count,
                    'active': handler.active
                }
                for handler_id, handler in self.handlers.items()
            }
            
            return {
                'queue_sizes': queue_sizes,
                'dead_letter_queue_size': len(self.dead_letter_queue),
                'handler_stats': handler_stats,
                'processing_stats': self.stats.copy(),
                'circuit_breaker_states': {
                    handler_id: breaker['state']
                    for handler_id, breaker in self.circuit_breakers.items()
                }
            }

def event_processor_original(config: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    if not isinstance(config, dict):
        return {'error': 'invalid_config'}
    
    if not isinstance(events, list) or len(events) > 1000:
        return {'error': 'invalid_events'}
    
    # Validate configuration
    max_queue_size = config.get('max_queue_size', 100)
    worker_threads = config.get('worker_threads', 2)
    
    if max_queue_size <= 0 or max_queue_size > 50000:
        return {'error': 'invalid_queue_size'}
    
    if worker_threads <= 0 or worker_threads > 10:
        return {'error': 'invalid_worker_count'}
    
    processor = EventProcessor(max_queue_size, worker_threads)
    
    # Set up filters and handlers based on config
    filters = config.get('filters', [])
    handlers = config.get('handlers', [])
    routes = config.get('routes', [])
    
    for filter_config in filters:
        if not isinstance(filter_config, dict) or 'id' not in filter_config:
            return {'error': 'invalid_filter_config'}
        processor.add_filter(filter_config['id'], filter_config.get('conditions', {}))
    
    # Simple handler that just returns success/failure
    def test_handler(events_batch):
        return len(events_batch) > 0
    
    for handler_config in handlers:
        if not isinstance(handler_config, dict) or 'id' not in handler_config:
            return {'error': 'invalid_handler_config'}
        
        batch_size = handler_config.get('batch_size', 1)
        timeout = handler_config.get('timeout', 0)
        
        processor.add_handler(handler_config['id'], test_handler, batch_size, timeout)
    
    for route in routes:
        if not isinstance(route, dict) or 'filter' not in route or 'handler' not in route:
            return {'error': 'invalid_route_config'}
        processor.add_route(route['filter'], route['handler'])
    
    processor.start()
    
    try:
        # Process events
        submitted_count = 0
        for event_data in events:
            if not isinstance(event_data, dict):
                continue
            
            try:
                event_type = EventType(event_data.get('type', 'user_action'))
                priority = EventPriority(event_data.get('priority', 1))
            except ValueError:
                continue
            
            event = Event(
                event_id=event_data.get('id', f'event_{submitted_count}'),
                event_type=event_type,
                priority=priority,
                payload=event_data.get('payload', {}),
                timestamp=time.time(),
                source=event_data.get('source', 'test')
            )
            
            if processor.submit_event(event):
                submitted_count += 1
        
        # Wait for processing
        time.sleep(0.1)
        
        stats = processor.get_stats()
        
        # Complex result analysis
        total_processed = stats['processing_stats']['events_processed']
        error_rate = stats['processing_stats']['processing_errors'] / max(1, total_processed)
        
        if error_rate > 0.5:
            result_status = 'high_error_rate'
        elif stats['processing_stats']['queue_overflows'] > 0:
            result_status = 'capacity_exceeded'
        elif total_processed == submitted_count and submitted_count > 0:
            result_status = 'full_processing'
        elif total_processed == 0 and submitted_count > 0:
            result_status = 'no_processing'
        else:
            result_status = 'partial_processing'
        
        return {
            'status': result_status,
            'submitted_events': submitted_count,
            'stats': stats
        }
    
    finally:
        processor.stop()


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original target_function to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: target_function(config, events)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from target_function
    """
    # Set defaults for optional parameters based on function requirements
    if c is None:
        c = 0
    if d is None:
        d = 20
    if b is None:
        b = -20
    if c is None:
        c = 0
    if d is None:
        d = 20
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-20 <= param <= 20):
                param = max(-20, min(20, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return target_function(a, b)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return target_function(a)
            except:
                pass
            return target_function(a)
        raise e
