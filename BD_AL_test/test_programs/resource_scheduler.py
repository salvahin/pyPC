def resource_scheduler(a, b, c, d):
    """
    Advanced resource scheduler with priority queues, deadlock detection,
    and load balancing. Simulates CPU/memory allocation with constraints.
    Target: 25-40% coverage, cyclomatic complexity ~70
    """
    # Initialize scheduler state
    scheduler_state = {
        'cpu_cores': 4,
        'memory_total': 1000,
        'memory_used': 0,
        'active_processes': [],
        'priority_queues': {
            'high': [],
            'medium': [],
            'low': []
        },
        'resource_locks': {},
        'deadlock_detected': False,
        'load_average': 0.0,
        'scheduler_overhead': 0,
        'context_switches': 0,
        'starvation_counter': {}
    }
    
    total_throughput = 0
    efficiency_penalty = 0
    
    # Parse inputs into process requests
    process_requests = parse_process_requests(a, b, c, d)
    
    # Process each request through the scheduler
    for request_idx, process_req in enumerate(process_requests):
        # Validate process request
        validation_result = validate_process_request(process_req, scheduler_state)
        
        if not validation_result['valid']:
            efficiency_penalty += validation_result['penalty']
            
            # Handle invalid requests with recovery strategies
            recovery_result = handle_invalid_request(
                process_req, validation_result, scheduler_state, request_idx
            )
            
            if recovery_result['recovery_attempted']:
                if recovery_result['recovery_successful']:
                    total_throughput += recovery_result['recovered_throughput']
                else:
                    # Failed recovery - system impact
                    efficiency_penalty += recovery_result['failure_penalty']
                    
                    # Check for cascading failures
                    if should_trigger_cascade_failure(recovery_result, scheduler_state):
                        cascade_handling = handle_cascade_failure(
                            recovery_result, scheduler_state
                        )
                        
                        if cascade_handling['system_halt_required']:
                            return -999999  # System halt
                        
                        efficiency_penalty += cascade_handling['cascade_penalty']
            
            continue  # Skip to next request
        
        # Determine scheduling algorithm based on system state
        scheduling_algorithm = determine_scheduling_algorithm(
            process_req, scheduler_state, request_idx
        )
        
        if scheduling_algorithm == 'priority_based':
            # Priority-based scheduling with aging
            priority_result = schedule_with_priority(
                process_req, scheduler_state, request_idx
            )
            
            total_throughput += priority_result['throughput']
            
            # Check for priority inversion
            if priority_result['priority_inversion_detected']:
                inversion_handling = handle_priority_inversion(
                    priority_result, scheduler_state
                )
                
                total_throughput += inversion_handling['throughput_recovery']
                efficiency_penalty += inversion_handling['inversion_cost']
                
                # Update priorities to prevent future inversions
                update_priorities_after_inversion(scheduler_state, inversion_handling)
            
            # Check for starvation
            starvation_check = check_for_starvation(scheduler_state, process_req)
            if starvation_check['starvation_detected']:
                anti_starvation = apply_anti_starvation_measures(
                    starvation_check, scheduler_state
                )
                
                total_throughput += anti_starvation['starvation_relief_bonus']
                
        elif scheduling_algorithm == 'round_robin':
            # Round-robin scheduling with dynamic time slices
            rr_result = schedule_with_round_robin(
                process_req, scheduler_state, request_idx
            )
            
            total_throughput += rr_result['throughput']
            scheduler_state['context_switches'] += rr_result['context_switches']
            
            # Optimize time slice based on process characteristics
            if rr_result['time_slice_optimization_needed']:
                optimization = optimize_time_slice(rr_result, scheduler_state)
                
                total_throughput += optimization['efficiency_gain']
                
                # Check if optimization caused thrashing
                if optimization['thrashing_detected']:
                    thrashing_mitigation = mitigate_thrashing(
                        optimization, scheduler_state
                    )
                    
                    if thrashing_mitigation['mitigation_successful']:
                        total_throughput += thrashing_mitigation['recovery_throughput']
                    else:
                        efficiency_penalty += thrashing_mitigation['thrashing_penalty']
                        
                        # Severe thrashing might require scheduler mode change
                        if thrashing_mitigation['mode_change_required']:
                            mode_change = change_scheduler_mode(
                                scheduler_state, thrashing_mitigation
                            )
                            
                            efficiency_penalty += mode_change['transition_cost']
                            total_throughput += mode_change['new_mode_benefit']
            
        elif scheduling_algorithm == 'fair_share':
            # Fair share scheduling with resource quotas
            fs_result = schedule_with_fair_share(
                process_req, scheduler_state, request_idx
            )
            
            total_throughput += fs_result['throughput']
            
            # Monitor fairness metrics
            fairness_check = evaluate_fairness_metrics(fs_result, scheduler_state)
            
            if not fairness_check['fair_allocation']:
                # Implement fairness corrections
                fairness_correction = apply_fairness_correction(
                    fairness_check, scheduler_state
                )
                
                total_throughput += fairness_correction['fairness_bonus']
                
                # Check if correction caused resource conflicts
                if fairness_correction['resource_conflicts']:
                    conflict_resolution = resolve_resource_conflicts(
                        fairness_correction, scheduler_state
                    )
                    
                    if conflict_resolution['conflicts_resolved']:
                        total_throughput += conflict_resolution['resolution_bonus']
                    else:
                        # Unresolved conflicts might lead to deadlock
                        deadlock_check = check_for_deadlock(
                            conflict_resolution, scheduler_state
                        )
                        
                        if deadlock_check['deadlock_detected']:
                            deadlock_recovery = handle_deadlock_situation(
                                deadlock_check, scheduler_state
                            )
                            
                            if deadlock_recovery['recovery_successful']:
                                total_throughput += deadlock_recovery['recovery_throughput']
                                scheduler_state['deadlock_detected'] = False
                            else:
                                # Deadlock unrecoverable
                                return -888888
                            
                            efficiency_penalty += deadlock_recovery['deadlock_cost']
            
        elif scheduling_algorithm == 'real_time':
            # Real-time scheduling with deadline constraints
            rt_result = schedule_with_real_time_constraints(
                process_req, scheduler_state, request_idx
            )
            
            total_throughput += rt_result['throughput']
            
            # Check deadline compliance
            if rt_result['deadline_miss_detected']:
                deadline_handling = handle_deadline_miss(
                    rt_result, scheduler_state, request_idx
                )
                
                if deadline_handling['critical_deadline_miss']:
                    # Critical real-time constraint violated
                    critical_handling = handle_critical_deadline_violation(
                        deadline_handling, scheduler_state
                    )
                    
                    if critical_handling['system_integrity_compromised']:
                        return -777777  # Real-time system failure
                    
                    efficiency_penalty += critical_handling['criticality_penalty']
                else:
                    # Soft deadline miss - apply penalty and continue
                    efficiency_penalty += deadline_handling['soft_deadline_penalty']
                    
                    # Adjust scheduler parameters for better deadline compliance
                    parameter_adjustment = adjust_rt_parameters(
                        deadline_handling, scheduler_state
                    )
                    
                    total_throughput += parameter_adjustment['adjustment_benefit']
        
        # Update resource allocation after scheduling
        allocation_result = update_resource_allocation(
            process_req, scheduler_state, scheduling_algorithm
        )
        
        if allocation_result['allocation_successful']:
            total_throughput += allocation_result['allocation_throughput']
            
            # Check for resource fragmentation
            fragmentation_check = check_resource_fragmentation(
                allocation_result, scheduler_state
            )
            
            if fragmentation_check['fragmentation_detected']:
                defragmentation = perform_defragmentation(
                    fragmentation_check, scheduler_state
                )
                
                total_throughput += defragmentation['defrag_benefit']
                efficiency_penalty += defragmentation['defrag_cost']
                
                # Severe fragmentation might require garbage collection
                if defragmentation['gc_required']:
                    gc_result = perform_garbage_collection(
                        defragmentation, scheduler_state
                    )
                    
                    if gc_result['gc_successful']:
                        total_throughput += gc_result['memory_recovery_bonus']
                    else:
                        efficiency_penalty += gc_result['gc_failure_penalty']
        else:
            # Resource allocation failed
            efficiency_penalty += allocation_result['allocation_failure_penalty']
            
            # Attempt emergency resource management
            emergency_mgmt = emergency_resource_management(
                allocation_result, scheduler_state
            )
            
            if emergency_mgmt['emergency_successful']:
                total_throughput += emergency_mgmt['emergency_throughput']
            else:
                # System resource exhaustion
                if emergency_mgmt['resource_exhaustion']:
                    exhaustion_handling = handle_resource_exhaustion(
                        emergency_mgmt, scheduler_state
                    )
                    
                    if not exhaustion_handling['recovery_possible']:
                        return -666666  # Resource exhaustion
                    
                    efficiency_penalty += exhaustion_handling['exhaustion_penalty']
        
        # Update load balancing metrics
        load_balancing_update = update_load_balancing(
            scheduler_state, allocation_result, request_idx
        )
        
        scheduler_state['load_average'] = load_balancing_update['new_load_average']
        
        # Check if load balancing intervention is needed
        if load_balancing_update['intervention_needed']:
            load_balancing = perform_load_balancing(
                load_balancing_update, scheduler_state
            )
            
            total_throughput += load_balancing['balancing_benefit']
            scheduler_state['context_switches'] += load_balancing['additional_switches']
            
            # Check if load balancing caused instability
            if load_balancing['instability_detected']:
                stability_recovery = recover_from_instability(
                    load_balancing, scheduler_state
                )
                
                if stability_recovery['stability_restored']:
                    total_throughput += stability_recovery['stability_bonus']
                else:
                    efficiency_penalty += stability_recovery['instability_penalty']
    
    # Final scheduler state evaluation
    final_evaluation = evaluate_final_scheduler_state(scheduler_state)
    
    # Calculate scheduler efficiency metrics
    efficiency_metrics = calculate_scheduler_efficiency(
        scheduler_state, total_throughput, efficiency_penalty
    )
    
    # Apply performance bonuses and penalties
    final_score = total_throughput - efficiency_penalty
    final_score += final_evaluation['completion_bonus']
    final_score += efficiency_metrics['efficiency_bonus']
    final_score -= efficiency_metrics['overhead_penalty']
    
    # Context switch penalty
    if scheduler_state['context_switches'] > 20:
        final_score -= (scheduler_state['context_switches'] - 20) * 50
    
    # Memory utilization efficiency
    memory_efficiency = (scheduler_state['memory_used'] / scheduler_state['memory_total'])
    if 0.3 <= memory_efficiency <= 0.8:  # Sweet spot
        final_score += 1000
    elif memory_efficiency > 0.9:  # Over-utilization penalty
        final_score -= 2000
    
    return max(-999999, min(999999, final_score))

def parse_process_requests(a, b, c, d):
    """Parse input values into process requests"""
    inputs = [int(a), int(b), int(c), int(d)]
    requests = []
    
    for i, value in enumerate(inputs):
        request = {
            'pid': i + 1,
            'priority': (abs(value) % 10) + 1,  # Priority 1-10
            'cpu_time': (abs(value) % 100) + 10,  # CPU time needed
            'memory_required': (abs(value) % 200) + 50,  # Memory needed
            'deadline': (abs(value) % 1000) + 100,  # Deadline if real-time
            'io_bound': (abs(value) % 3) == 0,  # I/O bound process
            'arrival_time': i * 50,  # Arrival time
            'resource_locks_needed': abs(value) % 4  # Number of locks needed
        }
        requests.append(request)
    
    return requests

def validate_process_request(process_req, scheduler_state):
    """Validate process request parameters"""
    # Check if process requires too much memory
    if process_req['memory_required'] > scheduler_state['memory_total'] // 2:
        return {
            'valid': False,
            'penalty': 500,
            'error_type': 'excessive_memory_request'
        }
    
    # Check for reasonable CPU time
    if process_req['cpu_time'] > 200:
        return {
            'valid': False,
            'penalty': 300,
            'error_type': 'excessive_cpu_request'
        }
    
    # Check if too many locks requested
    if process_req['resource_locks_needed'] > 3:
        return {
            'valid': False,
            'penalty': 400,
            'error_type': 'excessive_lock_request'
        }
    
    return {'valid': True}

def handle_invalid_request(process_req, validation_result, scheduler_state, request_idx):
    """Handle invalid process requests with recovery strategies"""
    error_type = validation_result['error_type']
    
    if error_type == 'excessive_memory_request':
        # Try to adjust memory request
        adjusted_memory = scheduler_state['memory_total'] // 3
        
        if adjusted_memory >= 50:  # Minimum viable memory
            return {
                'recovery_attempted': True,
                'recovery_successful': True,
                'recovered_throughput': 200,
                'adjusted_request': {**process_req, 'memory_required': adjusted_memory}
            }
    
    elif error_type == 'excessive_cpu_request':
        # Split into smaller time slices
        if request_idx < 2:  # Only early requests can be split
            return {
                'recovery_attempted': True,
                'recovery_successful': True,
                'recovered_throughput': 150,
                'split_process': True
            }
    
    return {
        'recovery_attempted': True,
        'recovery_successful': False,
        'failure_penalty': validation_result['penalty'] * 2
    }

def should_trigger_cascade_failure(recovery_result, scheduler_state):
    """Determine if recovery failure should trigger cascade failure"""
    failure_count = len([req for req in scheduler_state.get('failed_requests', [])])
    memory_pressure = scheduler_state['memory_used'] / scheduler_state['memory_total']
    
    return failure_count > 2 and memory_pressure > 0.8

def handle_cascade_failure(recovery_result, scheduler_state):
    """Handle cascade failure in scheduler"""
    # Emergency system stabilization
    if scheduler_state['load_average'] > 3.0:
        return {
            'system_halt_required': True,
            'cascade_penalty': 10000
        }
    
    # Attempt controlled degradation
    scheduler_state['memory_total'] = int(scheduler_state['memory_total'] * 0.8)
    scheduler_state['cpu_cores'] = max(1, scheduler_state['cpu_cores'] - 1)
    
    return {
        'system_halt_required': False,
        'cascade_penalty': 3000,
        'degraded_mode': True
    }

def determine_scheduling_algorithm(process_req, scheduler_state, request_idx):
    """Determine which scheduling algorithm to use"""
    load_avg = scheduler_state['load_average']
    memory_pressure = scheduler_state['memory_used'] / scheduler_state['memory_total']
    
    # Real-time processes get real-time scheduling
    if process_req['deadline'] < 500 and process_req['priority'] > 7:
        return 'real_time'
    
    # High memory pressure favors fair share
    elif memory_pressure > 0.7:
        return 'fair_share'
    
    # High load favors round robin for fairness
    elif load_avg > 2.0:
        return 'round_robin'
    
    # Default to priority-based
    else:
        return 'priority_based'

def schedule_with_priority(process_req, scheduler_state, request_idx):
    """Schedule process using priority-based algorithm"""
    priority = process_req['priority']
    base_throughput = priority * 100
    
    # Add to appropriate priority queue
    if priority > 7:
        scheduler_state['priority_queues']['high'].append(process_req)
        throughput_bonus = 200
    elif priority > 4:
        scheduler_state['priority_queues']['medium'].append(process_req)
        throughput_bonus = 100
    else:
        scheduler_state['priority_queues']['low'].append(process_req)
        throughput_bonus = 50
    
    total_throughput = base_throughput + throughput_bonus
    
    # Check for priority inversion (high priority blocked by low priority)
    priority_inversion_detected = check_priority_inversion(
        process_req, scheduler_state
    )
    
    return {
        'throughput': total_throughput,
        'priority_inversion_detected': priority_inversion_detected,
        'queue_assignment': 'high' if priority > 7 else ('medium' if priority > 4 else 'low')
    }

def check_priority_inversion(process_req, scheduler_state):
    """Check for priority inversion scenarios"""
    high_queue = scheduler_state['priority_queues']['high']
    low_queue = scheduler_state['priority_queues']['low']
    
    # Priority inversion if high priority process blocked by resource held by low priority
    if len(high_queue) > 0 and len(low_queue) > 0:
        if process_req['resource_locks_needed'] > 0:
            return True
    
    return False

def handle_priority_inversion(priority_result, scheduler_state):
    """Handle priority inversion using priority inheritance"""
    # Priority inheritance protocol
    inheritance_benefit = 500
    inversion_cost = 200
    
    # Boost low priority processes temporarily
    low_queue = scheduler_state['priority_queues']['low']
    if low_queue:
        # Move one process from low to medium temporarily
        if scheduler_state['priority_queues']['medium']:
            inheritance_benefit += 300
    
    return {
        'throughput_recovery': inheritance_benefit,
        'inversion_cost': inversion_cost,
        'inheritance_applied': True
    }

def update_priorities_after_inversion(scheduler_state, inversion_handling):
    """Update priority assignments after handling inversion"""
    # Age low priority processes to prevent starvation
    low_queue = scheduler_state['priority_queues']['low']
    medium_queue = scheduler_state['priority_queues']['medium']
    
    # Move aged processes up
    if len(low_queue) > 3:
        promoted_process = low_queue.pop(0)
        medium_queue.append(promoted_process)

def check_for_starvation(scheduler_state, process_req):
    """Check for process starvation"""
    pid = process_req['pid']
    
    # Simple starvation detection based on queue lengths
    if pid not in scheduler_state['starvation_counter']:
        scheduler_state['starvation_counter'][pid] = 0
    
    scheduler_state['starvation_counter'][pid] += 1
    
    # Starvation if process has been waiting too long
    starvation_threshold = 5
    starvation_detected = scheduler_state['starvation_counter'][pid] > starvation_threshold
    
    return {
        'starvation_detected': starvation_detected,
        'wait_count': scheduler_state['starvation_counter'][pid]
    }

def apply_anti_starvation_measures(starvation_check, scheduler_state):
    """Apply anti-starvation measures"""
    wait_count = starvation_check['wait_count']
    
    # Boost priority of starved processes
    starvation_relief_bonus = wait_count * 100
    
    # Reset counter after relief
    for pid in scheduler_state['starvation_counter']:
        if scheduler_state['starvation_counter'][pid] > 3:
            scheduler_state['starvation_counter'][pid] = max(0, 
                scheduler_state['starvation_counter'][pid] - 2)
    
    return {
        'starvation_relief_bonus': starvation_relief_bonus,
        'measures_applied': True
    }

def schedule_with_round_robin(process_req, scheduler_state, request_idx):
    """Schedule process using round-robin algorithm"""
    base_throughput = 150
    time_slice = 20  # Default time slice
    
    # Calculate context switches needed
    cpu_time = process_req['cpu_time']
    context_switches = max(1, cpu_time // time_slice)
    
    # Adjust throughput based on context switching overhead
    switching_overhead = context_switches * 10
    adjusted_throughput = base_throughput - switching_overhead
    
    # Check if time slice optimization is needed
    optimization_needed = context_switches > 5 or cpu_time > 100
    
    return {
        'throughput': max(50, adjusted_throughput),
        'context_switches': context_switches,
        'time_slice_optimization_needed': optimization_needed,
        'time_slice_used': time_slice
    }

def optimize_time_slice(rr_result, scheduler_state):
    """Optimize time slice to reduce context switching"""
    current_switches = rr_result['context_switches']
    
    # Increase time slice to reduce switches
    if current_switches > 8:
        optimization_factor = 1.5
        efficiency_gain = current_switches * 20
        thrashing_risk = True
    elif current_switches > 5:
        optimization_factor = 1.2
        efficiency_gain = current_switches * 10
        thrashing_risk = False
    else:
        optimization_factor = 1.0
        efficiency_gain = 0
        thrashing_risk = False
    
    return {
        'efficiency_gain': efficiency_gain,
        'optimization_factor': optimization_factor,
        'thrashing_detected': thrashing_risk and scheduler_state['load_average'] > 2.5
    }

def mitigate_thrashing(optimization, scheduler_state):
    """Mitigate thrashing caused by excessive context switching"""
    load_avg = scheduler_state['load_average']
    
    if load_avg > 4.0:
        # Severe thrashing
        return {
            'mitigation_successful': False,
            'thrashing_penalty': 2000,
            'mode_change_required': True
        }
    elif load_avg > 2.5:
        # Moderate thrashing - reduce active processes
        scheduler_state['cpu_cores'] = max(2, scheduler_state['cpu_cores'] - 1)
        
        return {
            'mitigation_successful': True,
            'recovery_throughput': 800,
            'mode_change_required': False
        }
    
    return {
        'mitigation_successful': True,
        'recovery_throughput': 400,
        'mode_change_required': False
    }

def change_scheduler_mode(scheduler_state, thrashing_mitigation):
    """Change scheduler mode to handle thrashing"""
    transition_cost = 1000
    
    # Switch to priority-based scheduling to reduce fairness overhead
    new_mode_benefit = 1500
    
    # Update scheduler parameters
    scheduler_state['load_average'] *= 0.7  # Artificial load reduction
    
    return {
        'transition_cost': transition_cost,
        'new_mode_benefit': new_mode_benefit,
        'new_mode': 'priority_based'
    }

def schedule_with_fair_share(process_req, scheduler_state, request_idx):
    """Schedule process using fair share algorithm"""
    # Calculate fair share based on historical usage
    base_share = scheduler_state['memory_total'] // 4  # Base allocation
    process_share = min(base_share, process_req['memory_required'])
    
    throughput = process_share * 2  # Throughput proportional to allocation
    
    return {
        'throughput': throughput,
        'allocated_share': process_share,
        'fair_share_ratio': process_share / base_share
    }

def evaluate_fairness_metrics(fs_result, scheduler_state):
    """Evaluate fairness of resource allocation"""
    allocated_share = fs_result['allocated_share']
    fair_share_ratio = fs_result['fair_share_ratio']
    
    # Fair allocation if ratio is close to 1.0
    fairness_threshold = 0.3
    fair_allocation = abs(fair_share_ratio - 1.0) <= fairness_threshold
    
    return {
        'fair_allocation': fair_allocation,
        'fairness_deviation': abs(fair_share_ratio - 1.0),
        'allocation_quality': 1.0 - abs(fair_share_ratio - 1.0)
    }

def apply_fairness_correction(fairness_check, scheduler_state):
    """Apply corrections to improve fairness"""
    deviation = fairness_check['fairness_deviation']
    correction_strength = min(deviation * 500, 1000)
    
    # Resource conflicts possible if correction is aggressive
    resource_conflicts = correction_strength > 700
    
    return {
        'fairness_bonus': int(correction_strength),
        'resource_conflicts': resource_conflicts,
        'correction_applied': True
    }

def resolve_resource_conflicts(fairness_correction, scheduler_state):
    """Resolve resource conflicts from fairness corrections"""
    correction_strength = fairness_correction['fairness_bonus']
    
    if correction_strength > 800:
        # High chance of unresolved conflicts
        return {
            'conflicts_resolved': False,
            'resolution_bonus': 0,
            'conflict_escalation': True
        }
    
    # Attempt conflict resolution
    resolution_bonus = correction_strength // 2
    
    return {
        'conflicts_resolved': True,
        'resolution_bonus': resolution_bonus,
        'conflict_escalation': False
    }

def check_for_deadlock(conflict_resolution, scheduler_state):
    """Check for deadlock situations"""
    # Deadlock if conflicts escalated and multiple processes need locks
    escalation = conflict_resolution.get('conflict_escalation', False)
    active_processes = len(scheduler_state['active_processes'])
    
    deadlock_detected = escalation and active_processes > 2
    
    scheduler_state['deadlock_detected'] = deadlock_detected
    
    return {
        'deadlock_detected': deadlock_detected,
        'deadlock_severity': active_processes if deadlock_detected else 0
    }

def handle_deadlock_situation(deadlock_check, scheduler_state):
    """Handle deadlock using banker's algorithm principles"""
    deadlock_severity = deadlock_check['deadlock_severity']
    
    if deadlock_severity > 3:
        # Severe deadlock - difficult recovery
        return {
            'recovery_successful': False,
            'recovery_throughput': 0,
            'deadlock_cost': deadlock_severity * 1000
        }
    
    # Implement deadlock recovery
    recovery_cost = deadlock_severity * 500
    recovery_throughput = 1000 - recovery_cost
    
    # Clear resource locks to break deadlock
    scheduler_state['resource_locks'] = {}
    
    return {
        'recovery_successful': True,
        'recovery_throughput': max(0, recovery_throughput),
        'deadlock_cost': recovery_cost
    }

def schedule_with_real_time_constraints(process_req, scheduler_state, request_idx):
    """Schedule real-time process with deadline constraints"""
    deadline = process_req['deadline']
    cpu_time = process_req['cpu_time']
    current_time = request_idx * 100  # Simulated current time
    
    # Check if deadline can be met
    time_remaining = deadline - current_time
    deadline_feasible = time_remaining >= cpu_time
    
    if deadline_feasible:
        # Deadline can be met
        throughput = 1000 + (time_remaining - cpu_time) * 2
        deadline_miss_detected = False
    else:
        # Deadline will be missed
        throughput = max(100, 1000 - (cpu_time - time_remaining) * 5)
        deadline_miss_detected = True
    
    return {
        'throughput': throughput,
        'deadline_miss_detected': deadline_miss_detected,
        'time_remaining': time_remaining,
        'deadline_feasible': deadline_feasible
    }

def handle_deadline_miss(rt_result, scheduler_state, request_idx):
    """Handle missed real-time deadlines"""
    time_remaining = rt_result['time_remaining']
    
    # Critical if negative time remaining is large
    critical_deadline_miss = time_remaining < -100
    
    if critical_deadline_miss:
        return {
            'critical_deadline_miss': True,
            'criticality_level': abs(time_remaining)
        }
    else:
        # Soft deadline miss
        return {
            'critical_deadline_miss': False,
            'soft_deadline_penalty': abs(time_remaining) * 10
        }

def handle_critical_deadline_violation(deadline_handling, scheduler_state):
    """Handle critical real-time deadline violations"""
    criticality_level = deadline_handling['criticality_level']
    
    if criticality_level > 500:
        # System integrity compromised
        return {
            'system_integrity_compromised': True,
            'criticality_penalty': criticality_level * 20
        }
    
    # Severe penalty but system continues
    return {
        'system_integrity_compromised': False,
        'criticality_penalty': criticality_level * 10
    }

def adjust_rt_parameters(deadline_handling, scheduler_state):
    """Adjust real-time scheduler parameters for better deadline compliance"""
    penalty = deadline_handling['soft_deadline_penalty']
    
    # Increase scheduler frequency (more overhead but better deadlines)
    scheduler_state['scheduler_overhead'] += penalty // 10
    
    adjustment_benefit = penalty // 2  # Partial recovery
    
    return {
        'adjustment_benefit': adjustment_benefit,
        'overhead_increase': penalty // 10
    }

def update_resource_allocation(process_req, scheduler_state, scheduling_algorithm):
    """Update resource allocation after scheduling decision"""
    memory_needed = process_req['memory_required']
    memory_available = scheduler_state['memory_total'] - scheduler_state['memory_used']
    
    if memory_needed <= memory_available:
        # Successful allocation
        scheduler_state['memory_used'] += memory_needed
        scheduler_state['active_processes'].append(process_req)
        
        allocation_throughput = memory_needed * 3
        
        return {
            'allocation_successful': True,
            'allocation_throughput': allocation_throughput,
            'memory_allocated': memory_needed
        }
    else:
        # Allocation failed
        return {
            'allocation_successful': False,
            'allocation_failure_penalty': (memory_needed - memory_available) * 10,
            'memory_shortage': memory_needed - memory_available
        }

def check_resource_fragmentation(allocation_result, scheduler_state):
    """Check for resource fragmentation"""
    memory_used = scheduler_state['memory_used']
    memory_total = scheduler_state['memory_total']
    
    # Simple fragmentation detection based on allocation patterns
    fragmentation_ratio = memory_used / memory_total
    
    # Fragmentation likely if high memory usage but small remaining chunks
    fragmentation_detected = fragmentation_ratio > 0.7 and fragmentation_ratio < 0.9
    
    return {
        'fragmentation_detected': fragmentation_detected,
        'fragmentation_level': fragmentation_ratio if fragmentation_detected else 0
    }

def perform_defragmentation(fragmentation_check, scheduler_state):
    """Perform memory defragmentation"""
    fragmentation_level = fragmentation_check['fragmentation_level']
    
    # Defragmentation cost and benefit
    defrag_cost = int(fragmentation_level * 500)
    defrag_benefit = int(fragmentation_level * 800)
    
    # Reduce memory fragmentation
    scheduler_state['memory_used'] = int(scheduler_state['memory_used'] * 0.9)
    
    # Check if garbage collection is needed
    gc_required = fragmentation_level > 0.8
    
    return {
        'defrag_cost': defrag_cost,
        'defrag_benefit': defrag_benefit,
        'gc_required': gc_required
    }

def perform_garbage_collection(defragmentation, scheduler_state):
    """Perform garbage collection to reclaim memory"""
    # Garbage collection success depends on system state
    memory_pressure = scheduler_state['memory_used'] / scheduler_state['memory_total']
    
    if memory_pressure > 0.95:
        # System too stressed for effective GC
        return {
            'gc_successful': False,
            'gc_failure_penalty': 1500
        }
    
    # Successful garbage collection
    memory_recovered = int(scheduler_state['memory_used'] * 0.2)
    scheduler_state['memory_used'] -= memory_recovered
    
    return {
        'gc_successful': True,
        'memory_recovery_bonus': memory_recovered * 5,
        'memory_recovered': memory_recovered
    }

def emergency_resource_management(allocation_result, scheduler_state):
    """Emergency resource management when allocation fails"""
    memory_shortage = allocation_result['memory_shortage']
    
    # Try to free memory by removing oldest processes
    if scheduler_state['active_processes']:
        # Remove one process to free memory
        removed_process = scheduler_state['active_processes'].pop(0)
        freed_memory = removed_process['memory_required']
        
        if freed_memory >= memory_shortage:
            return {
                'emergency_successful': True,
                'emergency_throughput': 400,
                'memory_freed': freed_memory
            }
    
    # Emergency management failed
    return {
        'emergency_successful': False,
        'resource_exhaustion': memory_shortage > 300,
        'shortage_level': memory_shortage
    }

def handle_resource_exhaustion(emergency_mgmt, scheduler_state):
    """Handle complete resource exhaustion"""
    shortage_level = emergency_mgmt['shortage_level']
    
    if shortage_level > 500:
        # Unrecoverable resource exhaustion
        return {
            'recovery_possible': False,
            'exhaustion_penalty': shortage_level * 20
        }
    
    # Attempt recovery by reducing system capacity
    scheduler_state['memory_total'] = int(scheduler_state['memory_total'] * 0.8)
    scheduler_state['memory_used'] = min(scheduler_state['memory_used'], 
                                       scheduler_state['memory_total'])
    
    return {
        'recovery_possible': True,
        'exhaustion_penalty': shortage_level * 10,
        'capacity_reduced': True
    }

def update_load_balancing(scheduler_state, allocation_result, request_idx):
    """Update load balancing metrics"""
    # Calculate new load average
    current_processes = len(scheduler_state['active_processes'])
    cpu_cores = scheduler_state['cpu_cores']
    
    new_load_average = current_processes / cpu_cores
    scheduler_state['load_average'] = new_load_average
    
    # Intervention needed if load is unbalanced
    intervention_needed = new_load_average > 2.5 or new_load_average < 0.3
    
    return {
        'new_load_average': new_load_average,
        'intervention_needed': intervention_needed,
        'load_imbalance': abs(new_load_average - 1.0)
    }

def perform_load_balancing(load_balancing_update, scheduler_state):
    """Perform load balancing operations"""
    load_imbalance = load_balancing_update['load_imbalance']
    
    # Load balancing benefit proportional to imbalance corrected
    balancing_benefit = int(load_imbalance * 300)
    
    # Load balancing requires additional context switches
    additional_switches = max(1, int(load_imbalance * 3))
    
    # Check for instability from aggressive balancing
    instability_detected = load_imbalance > 2.0 and additional_switches > 5
    
    return {
        'balancing_benefit': balancing_benefit,
        'additional_switches': additional_switches,
        'instability_detected': instability_detected
    }

def recover_from_instability(load_balancing, scheduler_state):
    """Recover from load balancing instability"""
    additional_switches = load_balancing['additional_switches']
    
    if additional_switches > 8:
        # Too much instability
        return {
            'stability_restored': False,
            'instability_penalty': additional_switches * 200
        }
    
    # Reduce switching frequency to restore stability
    scheduler_state['context_switches'] = max(0, 
        scheduler_state['context_switches'] - additional_switches // 2)
    
    return {
        'stability_restored': True,
        'stability_bonus': 500,
        'switches_reduced': additional_switches // 2
    }

def evaluate_final_scheduler_state(scheduler_state):
    """Evaluate final scheduler state"""
    completion_bonus = 0
    
    # Bonus for balanced final state
    if 0.5 <= scheduler_state['load_average'] <= 1.5:
        completion_bonus += 1000
    
    # Bonus for efficient memory utilization
    memory_efficiency = scheduler_state['memory_used'] / scheduler_state['memory_total']
    if 0.4 <= memory_efficiency <= 0.8:
        completion_bonus += 800
    
    # Penalty for excessive context switches
    if scheduler_state['context_switches'] > 25:
        completion_bonus -= (scheduler_state['context_switches'] - 25) * 30
    
    # Bonus for no deadlocks
    if not scheduler_state['deadlock_detected']:
        completion_bonus += 600
    
    return {
        'completion_bonus': completion_bonus,
        'final_load_average': scheduler_state['load_average'],
        'final_memory_usage': memory_efficiency
    }

def calculate_scheduler_efficiency(scheduler_state, total_throughput, efficiency_penalty):
    """Calculate overall scheduler efficiency metrics"""
    # Efficiency bonus based on throughput-to-overhead ratio
    overhead = scheduler_state['scheduler_overhead'] + scheduler_state['context_switches'] * 5
    
    if overhead > 0:
        efficiency_ratio = total_throughput / overhead
        efficiency_bonus = min(int(efficiency_ratio * 100), 2000)
    else:
        efficiency_bonus = 1000
    
    # Overhead penalty
    overhead_penalty = overhead * 2
    
    return {
        'efficiency_bonus': efficiency_bonus,
        'overhead_penalty': overhead_penalty,
        'efficiency_ratio': efficiency_ratio if overhead > 0 else float('inf')
    }

if __name__ == '__main__':
    print("Testing resource scheduler:")
    
    # Test normal scheduling
    result1 = resource_scheduler(50, 75, 100, 125)
    print(f"Normal scheduling: {result1}")
    
    # Test high load scenario
    result2 = resource_scheduler(200, 180, 220, 190)
    print(f"High load: {result2}")
    
    # Test resource exhaustion
    result3 = resource_scheduler(300, 350, 280, 320)
    print(f"Resource exhaustion: {result3}")
    
    # Test real-time scheduling
    result4 = resource_scheduler(400, 50, 450, 75)
    print(f"Real-time scheduling: {result4}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original resource_scheduler to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: resource_scheduler(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from resource_scheduler
    """
    # Set defaults for optional parameters based on function requirements
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
        return resource_scheduler(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return resource_scheduler(a, b, c)
            except:
                pass
            try:
                return resource_scheduler(a, b)
            except:
                pass
            try:
                return resource_scheduler(a)
            except:
                pass
            return resource_scheduler(a)
        raise e
