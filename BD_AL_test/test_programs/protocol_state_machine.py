def protocol_state_machine(a, b, c, d):
    """
    Complex protocol state machine validator with security checks.
    Simulates network protocol handling with state transitions,
    authentication, and attack detection.
    Target: 20-35% coverage, cyclomatic complexity ~65
    """
    # Initialize protocol state machine
    protocol_state = {
        'current_state': 'INIT',
        'auth_level': 0,
        'session_id': None,
        'packet_count': 0,
        'error_count': 0,
        'security_alerts': [],
        'connection_quality': 1.0,
        'timeout_counter': 0,
        'last_command': None,
        'protocol_version': '1.0'
    }
    
    # Convert inputs to protocol commands
    commands = parse_protocol_commands(a, b, c, d)
    protocol_score = 0
    security_penalty = 0
    
    # Process each command in sequence
    for cmd_index, command in enumerate(commands):
        # Validate command format and permissions
        validation_result = validate_command(command, protocol_state, cmd_index)
        
        if not validation_result['valid']:
            # Handle invalid command
            protocol_state['error_count'] += 1
            security_penalty += validation_result['penalty']
            
            # Check for potential attack patterns
            attack_analysis = analyze_attack_pattern(
                command, validation_result, protocol_state, cmd_index
            )
            
            if attack_analysis['attack_detected']:
                # Handle different attack types
                if attack_analysis['attack_type'] == 'replay':
                    replay_response = handle_replay_attack(
                        attack_analysis, protocol_state, cmd_index
                    )
                    
                    if replay_response['connection_terminated']:
                        return -900000  # Connection terminated due to replay attack
                    
                    security_penalty += replay_response['security_cost']
                    
                elif attack_analysis['attack_type'] == 'injection':
                    injection_response = handle_injection_attack(
                        attack_analysis, protocol_state, command
                    )
                    
                    if injection_response['system_compromised']:
                        return -800000  # System compromised
                    
                    security_penalty += injection_response['mitigation_cost']
                    
                    # Update security posture
                    if injection_response['escalate_security']:
                        escalation_result = escalate_security_level(
                            protocol_state, injection_response
                        )
                        protocol_score += escalation_result['hardening_bonus']
                        
                elif attack_analysis['attack_type'] == 'timing':
                    timing_response = handle_timing_attack(
                        attack_analysis, protocol_state, cmd_index
                    )
                    
                    if timing_response['timing_leak_detected']:
                        leak_mitigation = mitigate_timing_leak(
                            timing_response, protocol_state
                        )
                        security_penalty += leak_mitigation['mitigation_cost']
                        
                        if not leak_mitigation['successfully_mitigated']:
                            return -700000  # Timing leak exploitation
                
                elif attack_analysis['attack_type'] == 'buffer_overflow':
                    overflow_response = handle_buffer_overflow_attack(
                        attack_analysis, protocol_state, command
                    )
                    
                    if overflow_response['memory_corruption']:
                        corruption_handling = handle_memory_corruption(
                            overflow_response, protocol_state
                        )
                        
                        if corruption_handling['system_unstable']:
                            return -600000  # System instability
                        
                        security_penalty += corruption_handling['recovery_cost']
                
                # Log security incident
                protocol_state['security_alerts'].append({
                    'type': attack_analysis['attack_type'],
                    'command_index': cmd_index,
                    'severity': attack_analysis['severity']
                })
            
            # Attempt error recovery
            error_recovery = attempt_error_recovery(
                validation_result, protocol_state, cmd_index
            )
            
            if error_recovery['recovery_possible']:
                protocol_score += error_recovery['recovery_bonus']
                
                # Update state after recovery
                protocol_state['current_state'] = error_recovery['new_state']
            else:
                # Escalating error condition
                if should_terminate_connection(error_recovery, protocol_state):
                    return -500000  # Connection terminated due to errors
                    
            continue  # Skip to next command
        
        # Process valid command
        command_result = process_valid_command(command, protocol_state, cmd_index)
        
        if command_result['state_changed']:
            # Handle state transition
            transition_result = handle_state_transition(
                protocol_state['current_state'],
                command_result['new_state'],
                protocol_state,
                command
            )
            
            if transition_result['transition_valid']:
                protocol_state['current_state'] = command_result['new_state']
                protocol_score += transition_result['transition_bonus']
                
                # Check for special state handling
                if command_result['new_state'] == 'AUTHENTICATED':
                    auth_bonus = handle_authentication_success(
                        protocol_state, command, cmd_index
                    )
                    protocol_score += auth_bonus['auth_bonus']
                    protocol_state['auth_level'] = auth_bonus['new_auth_level']
                    
                    # Generate session
                    session_result = generate_session(protocol_state, command)
                    protocol_state['session_id'] = session_result['session_id']
                    protocol_score += session_result['session_bonus']
                    
                elif command_result['new_state'] == 'SECURE_CHANNEL':
                    # Establish secure channel
                    secure_channel = establish_secure_channel(
                        protocol_state, command, cmd_index
                    )
                    
                    if secure_channel['channel_established']:
                        protocol_score += secure_channel['security_bonus']
                        
                        # Validate channel integrity
                        integrity_check = validate_channel_integrity(
                            secure_channel, protocol_state
                        )
                        
                        if not integrity_check['integrity_maintained']:
                            # Channel compromise detected
                            compromise_response = handle_channel_compromise(
                                integrity_check, protocol_state
                            )
                            
                            if compromise_response['channel_terminated']:
                                return -400000  # Secure channel terminated
                            
                            security_penalty += compromise_response['compromise_penalty']
                    else:
                        # Failed to establish secure channel
                        security_penalty += secure_channel['failure_penalty']
                
                elif command_result['new_state'] == 'DATA_TRANSFER':
                    # Handle data transfer state
                    transfer_handling = handle_data_transfer_state(
                        protocol_state, command, cmd_index
                    )
                    
                    protocol_score += transfer_handling['transfer_bonus']
                    
                    # Check for data integrity
                    if transfer_handling['integrity_check_required']:
                        data_integrity = check_data_integrity(
                            transfer_handling, protocol_state, command
                        )
                        
                        if not data_integrity['data_valid']:
                            # Data corruption detected
                            corruption_response = handle_data_corruption(
                                data_integrity, protocol_state
                            )
                            
                            security_penalty += corruption_response['corruption_penalty']
                            
                            if corruption_response['malicious_corruption']:
                                # Potential data tampering attack
                                tampering_response = investigate_tampering(
                                    corruption_response, protocol_state
                                )
                                
                                if tampering_response['attack_confirmed']:
                                    return -300000  # Data tampering attack detected
                
                elif command_result['new_state'] == 'TERMINATING':
                    # Handle connection termination
                    termination_handling = handle_connection_termination(
                        protocol_state, command, cmd_index
                    )
                    
                    protocol_score += termination_handling['clean_termination_bonus']
                    
                    # Final security audit
                    final_audit = perform_final_security_audit(
                        protocol_state, termination_handling
                    )
                    
                    protocol_score += final_audit['audit_bonus']
                    security_penalty += final_audit['audit_penalty']
                    
                    # Clean termination achieved
                    if final_audit['clean_termination']:
                        protocol_score += 5000  # Clean protocol completion
                        break  # End processing
            else:
                # Invalid state transition
                security_penalty += transition_result['invalid_transition_penalty']
                
                # Check for state confusion attacks
                confusion_analysis = analyze_state_confusion(
                    transition_result, protocol_state, command
                )
                
                if confusion_analysis['confusion_attack']:
                    return -200000  # State confusion attack
        else:
            # Command processed without state change
            protocol_score += command_result['processing_bonus']
        
        # Update protocol metrics
        protocol_state['packet_count'] += 1
        update_connection_quality(protocol_state, command_result)
        
        # Check for timeout conditions
        timeout_check = check_protocol_timeout(protocol_state, cmd_index)
        if timeout_check['timeout_occurred']:
            timeout_handling = handle_protocol_timeout(
                timeout_check, protocol_state
            )
            
            if timeout_handling['connection_dropped']:
                return -100000  # Connection dropped due to timeout
            
            security_penalty += timeout_handling['timeout_penalty']
    
    # Final protocol state evaluation
    final_evaluation = evaluate_final_protocol_state(protocol_state)
    
    # Calculate final score with security adjustments
    final_score = protocol_score - security_penalty
    final_score += final_evaluation['completion_bonus']
    final_score -= final_evaluation['incomplete_penalty']
    
    # Apply protocol efficiency multiplier
    efficiency_multiplier = calculate_efficiency_multiplier(protocol_state)
    final_score = int(final_score * efficiency_multiplier)
    
    return max(-999999, min(999999, final_score))

def parse_protocol_commands(a, b, c, d):
    """Parse input values into protocol commands"""
    inputs = [int(a), int(b), int(c), int(d)]
    commands = []
    
    for i, value in enumerate(inputs):
        command = {
            'type': determine_command_type(value),
            'payload': abs(value) % 1000,
            'sequence': i,
            'timestamp': i * 100,  # Simulated timestamp
            'checksum': (abs(value) * 17) % 256,
            'flags': value % 16
        }
        commands.append(command)
    
    return commands

def determine_command_type(value):
    """Determine command type from input value"""
    cmd_code = abs(value) % 20
    
    if cmd_code < 3:
        return 'CONNECT'
    elif cmd_code < 6:
        return 'AUTHENTICATE'
    elif cmd_code < 9:
        return 'SECURE_HANDSHAKE'
    elif cmd_code < 12:
        return 'DATA'
    elif cmd_code < 15:
        return 'HEARTBEAT'
    elif cmd_code < 17:
        return 'CONTROL'
    elif cmd_code < 19:
        return 'DISCONNECT'
    else:
        return 'UNKNOWN'

def validate_command(command, protocol_state, cmd_index):
    """Validate command format and permissions"""
    # Basic format validation
    if command['type'] == 'UNKNOWN':
        return {
            'valid': False,
            'penalty': 200,
            'error_type': 'unknown_command'
        }
    
    # State-based validation
    current_state = protocol_state['current_state']
    
    # Check if command is allowed in current state
    if not is_command_allowed(command['type'], current_state):
        return {
            'valid': False,
            'penalty': 300,
            'error_type': 'invalid_state_command'
        }
    
    # Authentication requirement check
    if requires_authentication(command['type']) and protocol_state['auth_level'] == 0:
        return {
            'valid': False,
            'penalty': 400,
            'error_type': 'authentication_required'
        }
    
    # Payload validation
    payload_check = validate_payload(command, protocol_state)
    if not payload_check['valid']:
        return {
            'valid': False,
            'penalty': payload_check['penalty'],
            'error_type': 'invalid_payload'
        }
    
    # Checksum validation
    expected_checksum = calculate_expected_checksum(command)
    if command['checksum'] != expected_checksum:
        return {
            'valid': False,
            'penalty': 150,
            'error_type': 'checksum_mismatch'
        }
    
    return {'valid': True}

def is_command_allowed(cmd_type, current_state):
    """Check if command is allowed in current state"""
    allowed_commands = {
        'INIT': ['CONNECT'],
        'CONNECTED': ['AUTHENTICATE', 'DISCONNECT'],
        'AUTHENTICATED': ['SECURE_HANDSHAKE', 'DATA', 'HEARTBEAT', 'CONTROL', 'DISCONNECT'],
        'SECURE_CHANNEL': ['DATA', 'HEARTBEAT', 'CONTROL', 'DISCONNECT'],
        'DATA_TRANSFER': ['DATA', 'HEARTBEAT', 'DISCONNECT'],
        'TERMINATING': []
    }
    
    return cmd_type in allowed_commands.get(current_state, [])

def requires_authentication(cmd_type):
    """Check if command requires authentication"""
    auth_required = ['SECURE_HANDSHAKE', 'DATA', 'CONTROL']
    return cmd_type in auth_required

def validate_payload(command, protocol_state):
    """Validate command payload"""
    payload = command['payload']
    
    # Size validation
    if payload > 800:
        return {
            'valid': False,
            'penalty': 250,
            'error_reason': 'payload_too_large'
        }
    
    # Content validation based on command type
    if command['type'] == 'DATA':
        if payload == 0:
            return {
                'valid': False,
                'penalty': 100,
                'error_reason': 'empty_data_payload'
            }
    
    elif command['type'] == 'AUTHENTICATE':
        if payload < 100:  # Minimum credential length
            return {
                'valid': False,
                'penalty': 300,
                'error_reason': 'insufficient_credentials'
            }
    
    return {'valid': True}

def calculate_expected_checksum(command):
    """Calculate expected checksum for command"""
    # Simple checksum calculation
    base = command['payload'] + command['sequence'] * 2 + command['flags']
    return (base * 17) % 256

def analyze_attack_pattern(command, validation_result, protocol_state, cmd_index):
    """Analyze potential attack patterns"""
    attack_indicators = 0
    attack_type = 'none'
    
    # Check for replay attack indicators
    if command['sequence'] <= cmd_index - 2:  # Out of sequence
        attack_indicators += 2
        attack_type = 'replay'
    
    # Check for injection attack indicators
    if validation_result['error_type'] == 'invalid_payload' and command['payload'] > 500:
        attack_indicators += 3
        attack_type = 'injection'
    
    # Check for timing attack indicators
    if cmd_index > 0 and abs(command['timestamp'] - (cmd_index * 100)) > 50:
        attack_indicators += 1
        if attack_type == 'none':
            attack_type = 'timing'
    
    # Check for buffer overflow indicators
    if command['payload'] > 700 and validation_result['error_type'] == 'invalid_payload':
        attack_indicators += 4
        attack_type = 'buffer_overflow'
    
    return {
        'attack_detected': attack_indicators >= 2,
        'attack_type': attack_type,
        'severity': min(attack_indicators, 5),
        'confidence': attack_indicators / 5.0
    }

def handle_replay_attack(attack_analysis, protocol_state, cmd_index):
    """Handle detected replay attack"""
    severity = attack_analysis['severity']
    
    if severity >= 4:
        # High confidence replay attack
        return {
            'connection_terminated': True,
            'security_cost': 1000,
            'countermeasures': ['connection_drop', 'ip_block']
        }
    elif severity >= 2:
        # Possible replay attack
        # Increase security measures
        protocol_state['timeout_counter'] += 2
        
        return {
            'connection_terminated': False,
            'security_cost': severity * 200,
            'countermeasures': ['sequence_validation', 'timestamp_check']
        }
    
    return {
        'connection_terminated': False,
        'security_cost': 100,
        'countermeasures': ['logging']
    }

def handle_injection_attack(attack_analysis, protocol_state, command):
    """Handle detected injection attack"""
    severity = attack_analysis['severity']
    payload_size = command['payload']
    
    if severity >= 4 and payload_size > 600:
        # Critical injection attempt
        return {
            'system_compromised': True,
            'mitigation_cost': 2000,
            'escalate_security': True
        }
    elif severity >= 3:
        # Serious injection attempt
        return {
            'system_compromised': False,
            'mitigation_cost': severity * 300,
            'escalate_security': True
        }
    else:
        # Minor injection attempt
        return {
            'system_compromised': False,
            'mitigation_cost': severity * 150,
            'escalate_security': False
        }

def escalate_security_level(protocol_state, injection_response):
    """Escalate security level in response to threats"""
    # Implement additional security measures
    protocol_state['auth_level'] = min(protocol_state['auth_level'] + 1, 5)
    protocol_state['connection_quality'] *= 0.8  # Reduce quality due to threat
    
    hardening_bonus = 500 + protocol_state['auth_level'] * 100
    
    return {
        'hardening_bonus': hardening_bonus,
        'new_security_level': protocol_state['auth_level']
    }

def handle_timing_attack(attack_analysis, protocol_state, cmd_index):
    """Handle detected timing attack"""
    confidence = attack_analysis['confidence']
    
    # Timing attacks are subtle
    if confidence > 0.7:
        return {
            'timing_leak_detected': True,
            'leak_severity': int(confidence * 100),
            'mitigation_required': True
        }
    
    return {
        'timing_leak_detected': False,
        'leak_severity': 0,
        'mitigation_required': False
    }

def mitigate_timing_leak(timing_response, protocol_state):
    """Mitigate timing-based information leakage"""
    leak_severity = timing_response['leak_severity']
    
    # Add artificial delays to mask timing
    if leak_severity > 80:
        # High severity - difficult to mitigate
        return {
            'successfully_mitigated': False,
            'mitigation_cost': leak_severity * 10
        }
    
    # Add timing randomization
    protocol_state['timeout_counter'] += 1
    
    return {
        'successfully_mitigated': True,
        'mitigation_cost': leak_severity * 5
    }

def handle_buffer_overflow_attack(attack_analysis, protocol_state, command):
    """Handle buffer overflow attack attempt"""
    payload_size = command['payload']
    severity = attack_analysis['severity']
    
    if payload_size > 750 and severity >= 4:
        # Critical buffer overflow attempt
        return {
            'memory_corruption': True,
            'corruption_level': payload_size - 750,
            'immediate_threat': True
        }
    
    return {
        'memory_corruption': False,
        'corruption_level': 0,
        'immediate_threat': False
    }

def handle_memory_corruption(overflow_response, protocol_state):
    """Handle memory corruption from buffer overflow"""
    corruption_level = overflow_response['corruption_level']
    
    if corruption_level > 200:
        # Severe corruption
        return {
            'system_unstable': True,
            'recovery_cost': corruption_level * 20,
            'recovery_possible': False
        }
    
    # Attempt recovery
    recovery_cost = corruption_level * 10
    protocol_state['error_count'] += corruption_level // 50
    
    return {
        'system_unstable': False,
        'recovery_cost': recovery_cost,
        'recovery_possible': True
    }

def attempt_error_recovery(validation_result, protocol_state, cmd_index):
    """Attempt to recover from command validation errors"""
    error_type = validation_result['error_type']
    penalty = validation_result['penalty']
    
    # Recovery difficulty increases with error count
    recovery_difficulty = penalty + protocol_state['error_count'] * 50
    
    if recovery_difficulty > 800:
        # Too many errors or too severe
        return {
            'recovery_possible': False,
            'recovery_bonus': 0,
            'new_state': protocol_state['current_state']
        }
    
    # Attempt recovery based on error type
    if error_type == 'checksum_mismatch':
        # Can often recover from checksum errors
        return {
            'recovery_possible': True,
            'recovery_bonus': 200,
            'new_state': protocol_state['current_state']
        }
    elif error_type == 'authentication_required':
        # Move to authentication required state
        return {
            'recovery_possible': True,
            'recovery_bonus': 150,
            'new_state': 'CONNECTED'  # Back to authentication step
        }
    elif error_type == 'invalid_state_command':
        # Reset to appropriate state
        return {
            'recovery_possible': cmd_index < 2,  # Only early in protocol
            'recovery_bonus': 100,
            'new_state': 'CONNECTED'
        }
    
    return {
        'recovery_possible': False,
        'recovery_bonus': 0,
        'new_state': protocol_state['current_state']
    }

def should_terminate_connection(error_recovery, protocol_state):
    """Determine if connection should be terminated due to errors"""
    error_count = protocol_state['error_count']
    security_alerts = len(protocol_state['security_alerts'])
    
    # Too many errors or security incidents
    if error_count > 5 or security_alerts > 2:
        return True
    
    # Failed recovery indicates serious problems
    if not error_recovery['recovery_possible'] and error_count > 2:
        return True
    
    return False

def process_valid_command(command, protocol_state, cmd_index):
    """Process a valid command"""
    cmd_type = command['type']
    processing_bonus = 100
    state_changed = False
    new_state = protocol_state['current_state']
    
    if cmd_type == 'CONNECT':
        if protocol_state['current_state'] == 'INIT':
            state_changed = True
            new_state = 'CONNECTED'
            processing_bonus += 200
    
    elif cmd_type == 'AUTHENTICATE':
        if protocol_state['current_state'] == 'CONNECTED':
            # Authentication success depends on payload strength
            if command['payload'] > 300:
                state_changed = True
                new_state = 'AUTHENTICATED'
                processing_bonus += 300
            else:
                # Weak authentication
                processing_bonus += 100
    
    elif cmd_type == 'SECURE_HANDSHAKE':
        if protocol_state['current_state'] == 'AUTHENTICATED':
            state_changed = True
            new_state = 'SECURE_CHANNEL'
            processing_bonus += 400
    
    elif cmd_type == 'DATA':
        if protocol_state['current_state'] in ['AUTHENTICATED', 'SECURE_CHANNEL', 'DATA_TRANSFER']:
            if protocol_state['current_state'] != 'DATA_TRANSFER':
                state_changed = True
                new_state = 'DATA_TRANSFER'
            processing_bonus += command['payload'] // 10
    
    elif cmd_type == 'HEARTBEAT':
        # Heartbeat maintains connection quality
        protocol_state['connection_quality'] = min(1.0, protocol_state['connection_quality'] + 0.1)
        processing_bonus += 50
    
    elif cmd_type == 'CONTROL':
        processing_bonus += 150
    
    elif cmd_type == 'DISCONNECT':
        state_changed = True
        new_state = 'TERMINATING'
        processing_bonus += 250
    
    return {
        'state_changed': state_changed,
        'new_state': new_state,
        'processing_bonus': processing_bonus
    }

def handle_state_transition(old_state, new_state, protocol_state, command):
    """Handle state transitions with validation"""
    # Define valid state transitions
    valid_transitions = {
        'INIT': ['CONNECTED'],
        'CONNECTED': ['AUTHENTICATED', 'TERMINATING'],
        'AUTHENTICATED': ['SECURE_CHANNEL', 'DATA_TRANSFER', 'TERMINATING'],
        'SECURE_CHANNEL': ['DATA_TRANSFER', 'TERMINATING'],
        'DATA_TRANSFER': ['TERMINATING'],
        'TERMINATING': []
    }
    
    if new_state in valid_transitions.get(old_state, []):
        transition_bonus = calculate_transition_bonus(old_state, new_state, command)
        
        return {
            'transition_valid': True,
            'transition_bonus': transition_bonus
        }
    else:
        return {
            'transition_valid': False,
            'invalid_transition_penalty': 500
        }

def calculate_transition_bonus(old_state, new_state, command):
    """Calculate bonus for valid state transitions"""
    base_bonus = 200
    
    # More complex transitions get higher bonuses
    transition_complexity = {
        ('INIT', 'CONNECTED'): 1,
        ('CONNECTED', 'AUTHENTICATED'): 2,
        ('AUTHENTICATED', 'SECURE_CHANNEL'): 3,
        ('AUTHENTICATED', 'DATA_TRANSFER'): 2,
        ('SECURE_CHANNEL', 'DATA_TRANSFER'): 2,
        ('DATA_TRANSFER', 'TERMINATING'): 1,
        ('AUTHENTICATED', 'TERMINATING'): 1,
        ('SECURE_CHANNEL', 'TERMINATING'): 1,
        ('CONNECTED', 'TERMINATING'): 1
    }
    
    complexity = transition_complexity.get((old_state, new_state), 1)
    return base_bonus * complexity

def handle_authentication_success(protocol_state, command, cmd_index):
    """Handle successful authentication"""
    auth_strength = command['payload']
    
    # Determine authentication level based on credential strength
    if auth_strength > 700:
        auth_level = 3  # High security
        auth_bonus = 1000
    elif auth_strength > 400:
        auth_level = 2  # Medium security
        auth_bonus = 600
    else:
        auth_level = 1  # Basic security
        auth_bonus = 300
    
    # Early authentication bonus
    if cmd_index == 1:
        auth_bonus += 200
    
    return {
        'auth_bonus': auth_bonus,
        'new_auth_level': auth_level
    }

def generate_session(protocol_state, command):
    """Generate session after successful authentication"""
    # Simple session ID generation based on command
    session_id = (command['payload'] * 31 + command['checksum']) % 100000
    
    session_bonus = 400
    
    # Strong session generation bonus
    if command['payload'] > 500:
        session_bonus += 200
    
    return {
        'session_id': session_id,
        'session_bonus': session_bonus
    }

def establish_secure_channel(protocol_state, command, cmd_index):
    """Establish secure communication channel"""
    channel_strength = command['payload'] + protocol_state['auth_level'] * 100
    
    if channel_strength > 800:
        # Strong secure channel
        return {
            'channel_established': True,
            'security_bonus': 1500,
            'channel_strength': channel_strength
        }
    elif channel_strength > 400:
        # Moderate secure channel
        return {
            'channel_established': True,
            'security_bonus': 800,
            'channel_strength': channel_strength
        }
    else:
        # Failed to establish secure channel
        return {
            'channel_established': False,
            'failure_penalty': 600,
            'channel_strength': channel_strength
        }

def validate_channel_integrity(secure_channel, protocol_state):
    """Validate secure channel integrity"""
    channel_strength = secure_channel['channel_strength']
    
    # Check for integrity based on various factors
    integrity_threshold = 500 + protocol_state['error_count'] * 100
    
    integrity_maintained = channel_strength > integrity_threshold
    
    return {
        'integrity_maintained': integrity_maintained,
        'integrity_level': channel_strength / (integrity_threshold + 1)
    }

def handle_channel_compromise(integrity_check, protocol_state):
    """Handle secure channel compromise"""
    integrity_level = integrity_check['integrity_level']
    
    if integrity_level < 0.3:
        # Severe compromise
        return {
            'channel_terminated': True,
            'compromise_penalty': 2000
        }
    
    # Partial compromise - attempt recovery
    return {
        'channel_terminated': False,
        'compromise_penalty': int((1.0 - integrity_level) * 1000)
    }

def handle_data_transfer_state(protocol_state, command, cmd_index):
    """Handle data transfer state processing"""
    data_size = command['payload']
    transfer_bonus = data_size // 5
    
    # Large data transfers require integrity checks
    integrity_check_required = data_size > 300
    
    return {
        'transfer_bonus': transfer_bonus,
        'integrity_check_required': integrity_check_required,
        'data_size': data_size
    }

def check_data_integrity(transfer_handling, protocol_state, command):
    """Check data integrity during transfer"""
    data_size = transfer_handling['data_size']
    expected_checksum = calculate_data_checksum(data_size, command)
    actual_checksum = command['checksum']
    
    checksum_match = abs(expected_checksum - actual_checksum) <= 5
    
    return {
        'data_valid': checksum_match,
        'checksum_error': abs(expected_checksum - actual_checksum),
        'expected_checksum': expected_checksum,
        'actual_checksum': actual_checksum
    }

def calculate_data_checksum(data_size, command):
    """Calculate expected data checksum"""
    return (data_size + command['sequence'] * 3 + command['flags'] * 2) % 256

def handle_data_corruption(data_integrity, protocol_state):
    """Handle detected data corruption"""
    checksum_error = data_integrity['checksum_error']
    
    # Determine if corruption appears malicious
    malicious_threshold = 50
    malicious_corruption = checksum_error > malicious_threshold
    
    corruption_penalty = checksum_error * 20
    
    return {
        'corruption_penalty': corruption_penalty,
        'malicious_corruption': malicious_corruption,
        'error_magnitude': checksum_error
    }

def investigate_tampering(corruption_response, protocol_state):
    """Investigate potential data tampering"""
    error_magnitude = corruption_response['error_magnitude']
    security_alerts = len(protocol_state['security_alerts'])
    
    # High error with previous security issues suggests attack
    attack_probability = (error_magnitude / 100.0) + (security_alerts * 0.2)
    
    return {
        'attack_confirmed': attack_probability > 0.8,
        'attack_probability': min(attack_probability, 1.0)
    }

def handle_connection_termination(protocol_state, command, cmd_index):
    """Handle connection termination"""
    # Bonus for clean termination
    clean_termination_bonus = 500
    
    # Early termination penalty
    if cmd_index < 2:
        clean_termination_bonus -= 200
    
    # Bonus for successful data transfer before termination
    if protocol_state['packet_count'] > 2:
        clean_termination_bonus += 300
    
    return {
        'clean_termination_bonus': clean_termination_bonus,
        'termination_reason': 'normal'
    }

def perform_final_security_audit(protocol_state, termination_handling):
    """Perform final security audit"""
    audit_bonus = 0
    audit_penalty = 0
    
    # Audit connection quality
    if protocol_state['connection_quality'] > 0.8:
        audit_bonus += 400
    
    # Audit error rate
    error_rate = protocol_state['error_count'] / max(protocol_state['packet_count'], 1)
    if error_rate < 0.2:
        audit_bonus += 300
    elif error_rate > 0.5:
        audit_penalty += 400
    
    # Audit security incidents
    security_incidents = len(protocol_state['security_alerts'])
    if security_incidents == 0:
        audit_bonus += 500
    else:
        audit_penalty += security_incidents * 200
    
    # Determine if termination was clean
    clean_termination = (security_incidents == 0 and 
                        error_rate < 0.3 and 
                        protocol_state['connection_quality'] > 0.6)
    
    return {
        'audit_bonus': audit_bonus,
        'audit_penalty': audit_penalty,
        'clean_termination': clean_termination
    }

def analyze_state_confusion(transition_result, protocol_state, command):
    """Analyze potential state confusion attacks"""
    # State confusion attacks try to cause invalid state transitions
    penalty = transition_result['invalid_transition_penalty']
    
    # Multiple invalid transitions suggest attack
    if protocol_state['error_count'] > 2 and penalty > 400:
        return {'confusion_attack': True}
    
    return {'confusion_attack': False}

def update_connection_quality(protocol_state, command_result):
    """Update connection quality based on command processing"""
    if 'processing_bonus' in command_result:
        bonus = command_result['processing_bonus']
        
        if bonus > 300:
            # Good command processing
            protocol_state['connection_quality'] = min(1.0, protocol_state['connection_quality'] + 0.05)
        elif bonus < 100:
            # Poor command processing
            protocol_state['connection_quality'] = max(0.0, protocol_state['connection_quality'] - 0.1)

def check_protocol_timeout(protocol_state, cmd_index):
    """Check for protocol timeout conditions"""
    # Simulate timeout based on processing delays
    timeout_threshold = 300 + cmd_index * 50
    current_processing_time = protocol_state['error_count'] * 100 + protocol_state['timeout_counter'] * 150
    
    timeout_occurred = current_processing_time > timeout_threshold
    
    return {
        'timeout_occurred': timeout_occurred,
        'processing_time': current_processing_time,
        'threshold': timeout_threshold
    }

def handle_protocol_timeout(timeout_check, protocol_state):
    """Handle protocol timeout"""
    processing_time = timeout_check['processing_time']
    threshold = timeout_check['threshold']
    
    timeout_severity = processing_time - threshold
    
    if timeout_severity > 500:
        # Severe timeout - drop connection
        return {
            'connection_dropped': True,
            'timeout_penalty': timeout_severity * 2
        }
    
    # Mild timeout - add penalty and continue
    protocol_state['timeout_counter'] += 1
    
    return {
        'connection_dropped': False,
        'timeout_penalty': timeout_severity
    }

def evaluate_final_protocol_state(protocol_state):
    """Evaluate final protocol state"""
    completion_bonus = 0
    incomplete_penalty = 0
    
    # Check if protocol reached appropriate final state
    final_state = protocol_state['current_state']
    
    if final_state == 'TERMINATING':
        completion_bonus += 1000  # Proper protocol completion
    elif final_state in ['AUTHENTICATED', 'SECURE_CHANNEL', 'DATA_TRANSFER']:
        completion_bonus += 500   # Partial completion
    else:
        incomplete_penalty += 800  # Incomplete protocol
    
    # Factor in connection quality
    quality_factor = protocol_state['connection_quality']
    completion_bonus = int(completion_bonus * quality_factor)
    
    return {
        'completion_bonus': completion_bonus,
        'incomplete_penalty': incomplete_penalty,
        'final_state': final_state
    }

def calculate_efficiency_multiplier(protocol_state):
    """Calculate efficiency multiplier based on protocol performance"""
    base_multiplier = 1.0
    
    # Error efficiency
    error_rate = protocol_state['error_count'] / max(protocol_state['packet_count'], 1)
    error_multiplier = max(0.5, 1.0 - error_rate)
    
    # Security efficiency
    security_incidents = len(protocol_state['security_alerts'])
    security_multiplier = max(0.6, 1.0 - security_incidents * 0.1)
    
    # Connection quality
    quality_multiplier = protocol_state['connection_quality']
    
    return base_multiplier * error_multiplier * security_multiplier * quality_multiplier

if __name__ == '__main__':
    print("Testing protocol state machine:")
    
    # Test normal protocol flow
    result1 = protocol_state_machine(10, 350, 500, 80)
    print(f"Normal flow: {result1}")
    
    # Test with security threats
    result2 = protocol_state_machine(90, 800, 750, 950)
    print(f"Security threats: {result2}")
    
    # Test authentication failure
    result3 = protocol_state_machine(5, 50, 200, 15)
    print(f"Auth failure: {result3}")
    
    # Test early termination
    result4 = protocol_state_machine(10, 20, 30, 85)
    print(f"Early termination: {result4}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original protocol_state_machine to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: protocol_state_machine(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from protocol_state_machine
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
        return protocol_state_machine(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return protocol_state_machine(a, b, c)
            except:
                pass
            try:
                return protocol_state_machine(a, b)
            except:
                pass
            try:
                return protocol_state_machine(a)
            except:
                pass
            return protocol_state_machine(a)
        raise e
