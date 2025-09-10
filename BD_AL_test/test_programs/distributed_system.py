"""
Distributed System with Consensus and Fault Tolerance
Target: 35-50% coverage, ~85 cyclomatic complexity
"""

import time
import random
import threading
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import json

class NodeStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPECTED = "suspected"
    FAILED = "failed"
    RECOVERING = "recovering"

class MessageType(Enum):
    HEARTBEAT = "heartbeat"
    ELECTION = "election"
    COMMIT = "commit"
    PREPARE = "prepare"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    DATA_SYNC = "data_sync"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"

class ConsensusAlgorithm(Enum):
    RAFT = "raft"
    PBFT = "pbft"
    PAXOS = "paxos"

@dataclass
class NetworkMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: float
    term: int = 0
    sequence_number: int = 0
    signature: Optional[str] = None

@dataclass
class ConsensusProposal:
    proposal_id: str
    proposer_id: str
    value: Any
    timestamp: float
    term: int
    votes_received: Set[str] = field(default_factory=set)
    votes_needed: int = 0
    status: str = "pending"  # pending, committed, rejected

class DistributedNode:
    def __init__(self, node_id: str, consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT):
        self.node_id = node_id
        self.consensus_algorithm = consensus_algorithm
        self.status = NodeStatus.ACTIVE
        
        # Cluster membership
        self.cluster_nodes: Set[str] = {node_id}
        self.leader_id: Optional[str] = None
        self.current_term = 0
        self.voted_for: Optional[str] = None
        
        # State machine
        self.log: List[Dict[str, Any]] = []
        self.commit_index = 0
        self.last_applied = 0
        self.state: Dict[str, Any] = {}
        
        # Consensus state
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.active_proposals: Set[str] = set()
        
        # Network simulation
        self.message_queue: deque = deque()
        self.sent_messages: List[NetworkMessage] = []
        self.received_messages: List[NetworkMessage] = []
        
        # Failure detection
        self.heartbeat_timeout = 5.0
        self.election_timeout = random.uniform(10.0, 15.0)
        self.last_heartbeat: Dict[str, float] = {}
        self.failure_detector_active = True
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'proposals_made': 0,
            'proposals_committed': 0,
            'leader_changes': 0,
            'consensus_rounds': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.running = False
        self.background_thread: Optional[threading.Thread] = None
        
    def join_cluster(self, existing_nodes: Set[str]) -> bool:
        """Join an existing cluster"""
        with self.lock:
            if not existing_nodes:
                # Bootstrap new cluster
                self.cluster_nodes = {self.node_id}
                self.leader_id = self.node_id
                self.current_term = 1
                return True
            
            # Request to join existing cluster
            self.cluster_nodes.update(existing_nodes)
            self.cluster_nodes.add(self.node_id)
            
            # Send join request to suspected leader
            if existing_nodes:
                suspected_leader = next(iter(existing_nodes))
                join_message = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=suspected_leader,
                    message_type=MessageType.DATA_SYNC,
                    payload={'action': 'join_request', 'node_id': self.node_id},
                    timestamp=time.time()
                )
                
                self._send_message(join_message)
            
            return True
    
    def propose_value(self, value: Any, proposal_id: Optional[str] = None) -> str:
        """Propose a value for consensus"""
        with self.lock:
            if not proposal_id:
                proposal_id = f"proposal_{self.node_id}_{int(time.time() * 1000)}"
            
            # Check if we can propose (typically only leader can)
            if self.consensus_algorithm == ConsensusAlgorithm.RAFT:
                if self.leader_id != self.node_id:
                    return ""  # Only leader can propose in Raft
            
            proposal = ConsensusProposal(
                proposal_id=proposal_id,
                proposer_id=self.node_id,
                value=value,
                timestamp=time.time(),
                term=self.current_term,
                votes_needed=self._calculate_majority()
            )
            
            self.proposals[proposal_id] = proposal
            self.active_proposals.add(proposal_id)
            self.metrics['proposals_made'] += 1
            
            # Initiate consensus based on algorithm
            if self.consensus_algorithm == ConsensusAlgorithm.RAFT:
                self._start_raft_consensus(proposal)
            elif self.consensus_algorithm == ConsensusAlgorithm.PBFT:
                self._start_pbft_consensus(proposal)
            elif self.consensus_algorithm == ConsensusAlgorithm.PAXOS:
                self._start_paxos_consensus(proposal)
            
            return proposal_id
    
    def handle_message(self, message: NetworkMessage) -> bool:
        """Handle incoming network message"""
        with self.lock:
            self.received_messages.append(message)
            self.metrics['messages_received'] += 1
            
            # Update heartbeat tracking
            if message.message_type == MessageType.HEARTBEAT:
                self.last_heartbeat[message.sender_id] = message.timestamp
            
            # Handle based on message type and consensus algorithm
            try:
                if self.consensus_algorithm == ConsensusAlgorithm.RAFT:
                    return self._handle_raft_message(message)
                elif self.consensus_algorithm == ConsensusAlgorithm.PBFT:
                    return self._handle_pbft_message(message)
                elif self.consensus_algorithm == ConsensusAlgorithm.PAXOS:
                    return self._handle_paxos_message(message)
                else:
                    return self._handle_generic_message(message)
                    
            except Exception:
                return False
    
    def start_background_tasks(self):
        """Start background tasks (heartbeat, failure detection, etc.)"""
        if self.running:
            return
        
        self.running = True
        self.background_thread = threading.Thread(target=self._background_loop)
        self.background_thread.daemon = True
        self.background_thread.start()
    
    def stop(self):
        """Stop the node"""
        with self.lock:
            self.running = False
            self.status = NodeStatus.INACTIVE
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=2.0)
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state"""
        with self.lock:
            active_nodes = [node for node in self.cluster_nodes 
                          if node == self.node_id or 
                          (node in self.last_heartbeat and 
                           time.time() - self.last_heartbeat[node] < self.heartbeat_timeout * 2)]
            
            return {
                'node_id': self.node_id,
                'status': self.status.value,
                'cluster_size': len(self.cluster_nodes),
                'active_nodes': len(active_nodes),
                'leader_id': self.leader_id,
                'current_term': self.current_term,
                'log_size': len(self.log),
                'commit_index': self.commit_index,
                'active_proposals': len(self.active_proposals),
                'state_size': len(self.state),
                'metrics': self.metrics.copy()
            }
    
    def _start_raft_consensus(self, proposal: ConsensusProposal):
        """Start Raft consensus for proposal"""
        if self.leader_id != self.node_id:
            return  # Only leader can start consensus
        
        # Create log entry
        log_entry = {
            'term': self.current_term,
            'proposal_id': proposal.proposal_id,
            'value': proposal.value,
            'timestamp': proposal.timestamp
        }
        
        self.log.append(log_entry)
        log_index = len(self.log) - 1
        
        # Send AppendEntries to all followers
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                append_message = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.COMMIT,
                    payload={
                        'log_entry': log_entry,
                        'log_index': log_index,
                        'leader_commit': self.commit_index,
                        'prev_log_index': log_index - 1 if log_index > 0 else -1,
                        'prev_log_term': self.log[log_index - 1]['term'] if log_index > 0 else 0
                    },
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(append_message)
    
    def _start_pbft_consensus(self, proposal: ConsensusProposal):
        """Start PBFT consensus (3-phase protocol)"""
        # Phase 1: Pre-prepare
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                preprepare_message = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.PREPARE,
                    payload={
                        'proposal_id': proposal.proposal_id,
                        'value': proposal.value,
                        'view': self.current_term,
                        'sequence': len(self.log)
                    },
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(preprepare_message)
    
    def _start_paxos_consensus(self, proposal: ConsensusProposal):
        """Start Paxos consensus (Prepare phase)"""
        proposal_number = (self.current_term, int(self.node_id.split('_')[-1]) if '_' in self.node_id else 0)
        
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                prepare_message = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.PREPARE,
                    payload={
                        'proposal_id': proposal.proposal_id,
                        'proposal_number': proposal_number,
                        'phase': 'prepare'
                    },
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(prepare_message)
    
    def _handle_raft_message(self, message: NetworkMessage) -> bool:
        """Handle Raft-specific messages"""
        if message.message_type == MessageType.VOTE_REQUEST:
            return self._handle_raft_vote_request(message)
        elif message.message_type == MessageType.VOTE_RESPONSE:
            return self._handle_raft_vote_response(message)
        elif message.message_type == MessageType.COMMIT:
            return self._handle_raft_append_entries(message)
        elif message.message_type == MessageType.HEARTBEAT:
            return self._handle_raft_heartbeat(message)
        
        return True
    
    def _handle_raft_vote_request(self, message: NetworkMessage) -> bool:
        """Handle Raft vote request"""
        candidate_id = message.sender_id
        candidate_term = message.term
        
        grant_vote = False
        
        if candidate_term > self.current_term:
            # Higher term, update and consider voting
            self.current_term = candidate_term
            self.voted_for = None
            self.leader_id = None
        
        if (candidate_term == self.current_term and 
            (self.voted_for is None or self.voted_for == candidate_id)):
            
            # Additional checks for log consistency would go here
            grant_vote = True
            self.voted_for = candidate_id
        
        # Send vote response
        response = NetworkMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            recipient_id=candidate_id,
            message_type=MessageType.VOTE_RESPONSE,
            payload={
                'vote_granted': grant_vote,
                'term': self.current_term
            },
            timestamp=time.time(),
            term=self.current_term
        )
        
        self._send_message(response)
        return True
    
    def _handle_raft_vote_response(self, message: NetworkMessage) -> bool:
        """Handle Raft vote response"""
        if self.status != NodeStatus.ACTIVE or message.term < self.current_term:
            return True
        
        if message.term > self.current_term:
            self.current_term = message.term
            self.voted_for = None
            self.leader_id = None
            return True
        
        # If we're still a candidate and received a vote
        vote_granted = message.payload.get('vote_granted', False)
        if vote_granted and self.leader_id is None:
            # Count votes (simplified - would need proper vote tracking)
            majority = self._calculate_majority()
            
            # Simulate receiving majority (would need proper vote counting)
            if random.random() < 0.7:  # Simulate winning election
                self._become_leader()
        
        return True
    
    def _handle_raft_append_entries(self, message: NetworkMessage) -> bool:
        """Handle Raft AppendEntries"""
        leader_id = message.sender_id
        leader_term = message.term
        
        success = False
        
        if leader_term >= self.current_term:
            self.current_term = leader_term
            self.leader_id = leader_id
            self.last_heartbeat[leader_id] = message.timestamp
            
            # Process log entry if present
            log_entry = message.payload.get('log_entry')
            if log_entry:
                log_index = message.payload.get('log_index', 0)
                
                # Simplified consistency check
                if log_index == len(self.log):
                    self.log.append(log_entry)
                    success = True
                    
                    # Update commit index
                    leader_commit = message.payload.get('leader_commit', 0)
                    if leader_commit > self.commit_index:
                        self.commit_index = min(leader_commit, len(self.log) - 1)
                        self._apply_committed_entries()
        
        # Send response
        response = NetworkMessage(
            message_id=self._generate_message_id(),
            sender_id=self.node_id,
            recipient_id=leader_id,
            message_type=MessageType.COMMIT,
            payload={
                'success': success,
                'term': self.current_term,
                'match_index': len(self.log) - 1 if success else -1
            },
            timestamp=time.time(),
            term=self.current_term
        )
        
        self._send_message(response)
        return True
    
    def _handle_pbft_message(self, message: NetworkMessage) -> bool:
        """Handle PBFT-specific messages"""
        if message.message_type == MessageType.PREPARE:
            return self._handle_pbft_prepare(message)
        elif message.message_type == MessageType.COMMIT:
            return self._handle_pbft_commit(message)
        
        return True
    
    def _handle_pbft_prepare(self, message: NetworkMessage) -> bool:
        """Handle PBFT prepare message"""
        proposal_id = message.payload.get('proposal_id')
        
        if proposal_id not in self.proposals:
            # Create proposal from message
            self.proposals[proposal_id] = ConsensusProposal(
                proposal_id=proposal_id,
                proposer_id=message.sender_id,
                value=message.payload.get('value'),
                timestamp=message.timestamp,
                term=message.term
            )
        
        # Send prepare response to all nodes
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                prepare_response = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.VOTE_RESPONSE,
                    payload={
                        'proposal_id': proposal_id,
                        'phase': 'prepare',
                        'view': self.current_term
                    },
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(prepare_response)
        
        return True
    
    def _background_loop(self):
        """Background thread for periodic tasks"""
        while self.running:
            try:
                with self.lock:
                    # Send heartbeats if leader
                    if self.leader_id == self.node_id:
                        self._send_heartbeats()
                    
                    # Check for failed nodes
                    if self.failure_detector_active:
                        self._detect_failures()
                    
                    # Check for election timeout
                    if (self.leader_id is None or 
                        (self.leader_id in self.last_heartbeat and
                         time.time() - self.last_heartbeat[self.leader_id] > self.election_timeout)):
                        self._start_election()
                
                time.sleep(1.0)  # Background task interval
                
            except Exception:
                # Continue running even if background task fails
                continue
    
    def _send_heartbeats(self):
        """Send heartbeat messages to all cluster nodes"""
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                heartbeat = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.HEARTBEAT,
                    payload={'leader_commit': self.commit_index},
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(heartbeat)
    
    def _detect_failures(self):
        """Detect failed nodes based on heartbeat timeouts"""
        current_time = time.time()
        
        for node_id in list(self.cluster_nodes):
            if node_id != self.node_id:
                last_seen = self.last_heartbeat.get(node_id, 0)
                
                if current_time - last_seen > self.heartbeat_timeout * 3:
                    # Node appears to have failed
                    if node_id == self.leader_id:
                        self.leader_id = None  # Trigger election
    
    def _start_election(self):
        """Start leader election"""
        if self.status != NodeStatus.ACTIVE:
            return
        
        self.current_term += 1
        self.voted_for = self.node_id
        self.leader_id = None
        self.metrics['leader_changes'] += 1
        
        # Send vote requests
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                vote_request = NetworkMessage(
                    message_id=self._generate_message_id(),
                    sender_id=self.node_id,
                    recipient_id=node_id,
                    message_type=MessageType.VOTE_REQUEST,
                    payload={
                        'candidate_id': self.node_id,
                        'last_log_index': len(self.log) - 1,
                        'last_log_term': self.log[-1]['term'] if self.log else 0
                    },
                    timestamp=time.time(),
                    term=self.current_term
                )
                
                self._send_message(vote_request)
    
    def _become_leader(self):
        """Become cluster leader"""
        self.leader_id = self.node_id
        self.status = NodeStatus.ACTIVE
        
        # Send initial heartbeats
        self._send_heartbeats()
    
    def _apply_committed_entries(self):
        """Apply committed log entries to state machine"""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            
            if self.last_applied < len(self.log):
                entry = self.log[self.last_applied]
                proposal_id = entry.get('proposal_id')
                
                if proposal_id in self.proposals:
                    proposal = self.proposals[proposal_id]
                    proposal.status = "committed"
                    self.metrics['proposals_committed'] += 1
                    
                    # Apply to state machine
                    self.state[proposal_id] = proposal.value
    
    def _send_message(self, message: NetworkMessage):
        """Send message (simulated network)"""
        message.signature = self._sign_message(message)
        self.sent_messages.append(message)
        self.message_queue.append(message)
        self.metrics['messages_sent'] += 1
    
    def _sign_message(self, message: NetworkMessage) -> str:
        """Sign message for integrity"""
        content = f"{message.sender_id}{message.recipient_id}{message.message_type.value}{message.timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{self.node_id}_{int(time.time() * 1000000)}"
    
    def _calculate_majority(self) -> int:
        """Calculate majority threshold"""
        return (len(self.cluster_nodes) // 2) + 1
    
    def _handle_generic_message(self, message: NetworkMessage) -> bool:
        """Handle generic messages"""
        return True
    
    def _handle_raft_heartbeat(self, message: NetworkMessage) -> bool:
        """Handle Raft heartbeat"""
        self.last_heartbeat[message.sender_id] = message.timestamp
        
        if message.term >= self.current_term:
            self.current_term = message.term
            self.leader_id = message.sender_id
        
        return True
    
    def _handle_pbft_commit(self, message: NetworkMessage) -> bool:
        """Handle PBFT commit message"""
        return True
    
    def _handle_paxos_message(self, message: NetworkMessage) -> bool:
        """Handle Paxos-specific messages"""
        return True

class DistributedSystem:
    def __init__(self, num_nodes: int, consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.RAFT):
        self.nodes: Dict[str, DistributedNode] = {}
        self.consensus_algorithm = consensus_algorithm
        self.network_partition: Set[Tuple[str, str]] = set()
        self.message_delay_range = (0.1, 0.5)
        self.running = False
        
        # Create nodes
        for i in range(num_nodes):
            node_id = f"node_{i}"
            node = DistributedNode(node_id, consensus_algorithm)
            self.nodes[node_id] = node
        
        # Connect nodes to cluster
        node_ids = set(self.nodes.keys())
        for node in self.nodes.values():
            node.join_cluster(node_ids - {node.node_id})
    
    def start_system(self):
        """Start all nodes"""
        self.running = True
        for node in self.nodes.values():
            node.start_background_tasks()
    
    def stop_system(self):
        """Stop all nodes"""
        self.running = False
        for node in self.nodes.values():
            node.stop()
    
    def simulate_network(self, duration: float):
        """Simulate network message delivery"""
        end_time = time.time() + duration
        
        while time.time() < end_time and self.running:
            # Collect all pending messages
            all_messages = []
            for node in self.nodes.values():
                while node.message_queue:
                    all_messages.append(node.message_queue.popleft())
            
            # Deliver messages with simulated delay and potential partition
            for message in all_messages:
                if (message.sender_id, message.recipient_id) not in self.network_partition:
                    # Simulate network delay
                    delay = random.uniform(*self.message_delay_range)
                    time.sleep(delay)
                    
                    # Deliver message
                    if message.recipient_id in self.nodes:
                        self.nodes[message.recipient_id].handle_message(message)
            
            time.sleep(0.1)  # Network simulation interval
    
    def create_network_partition(self, partition_groups: List[Set[str]]):
        """Create network partition between groups"""
        self.network_partition.clear()
        
        for i, group1 in enumerate(partition_groups):
            for j, group2 in enumerate(partition_groups):
                if i != j:
                    for node1 in group1:
                        for node2 in group2:
                            self.network_partition.add((node1, node2))
                            self.network_partition.add((node2, node1))
    
    def heal_network_partition(self):
        """Remove network partition"""
        self.network_partition.clear()
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get overall system state"""
        node_states = {node_id: node.get_cluster_state() for node_id, node in self.nodes.items()}
        
        # Analyze consensus
        leaders = set()
        terms = set()
        
        for state in node_states.values():
            if state['leader_id']:
                leaders.add(state['leader_id'])
            terms.add(state['current_term'])
        
        return {
            'num_nodes': len(self.nodes),
            'active_nodes': sum(1 for state in node_states.values() if state['status'] == 'active'),
            'leaders': list(leaders),
            'leader_consensus': len(leaders) <= 1,
            'max_term': max(terms) if terms else 0,
            'network_partitions': len(self.network_partition),
            'node_states': node_states
        }

def distributed_system_original(system_config: Dict[str, Any], simulation_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    # Validate system configuration
    if not isinstance(system_config, dict):
        return {'error': 'invalid_system_config'}
    
    if not isinstance(simulation_params, dict):
        return {'error': 'invalid_simulation_params'}
    
    # Extract system parameters
    num_nodes = system_config.get('num_nodes', 3)
    consensus_algorithm = system_config.get('consensus_algorithm', 'raft')
    
    if num_nodes < 1 or num_nodes > 20:
        return {'error': 'invalid_node_count'}
    
    if consensus_algorithm not in ['raft', 'pbft', 'paxos']:
        return {'error': 'invalid_consensus_algorithm'}
    
    # Extract simulation parameters
    simulation_duration = simulation_params.get('duration', 10.0)
    proposals = simulation_params.get('proposals', [])
    network_events = simulation_params.get('network_events', [])
    
    if simulation_duration <= 0 or simulation_duration > 300:
        return {'error': 'invalid_simulation_duration'}
    
    if len(proposals) > 100:
        return {'error': 'too_many_proposals'}
    
    # Create and start distributed system
    try:
        consensus_alg = ConsensusAlgorithm(consensus_algorithm)
        system = DistributedSystem(num_nodes, consensus_alg)
        system.start_system()
        
        # Let system stabilize
        time.sleep(1.0)
        
        # Execute simulation
        start_time = time.time()
        
        # Submit proposals
        proposal_results = []
        if proposals:
            leader_node = None
            for node in system.nodes.values():
                if node.leader_id == node.node_id:
                    leader_node = node
                    break
            
            if leader_node:
                for i, proposal_value in enumerate(proposals):
                    proposal_id = leader_node.propose_value(proposal_value)
                    if proposal_id:
                        proposal_results.append({
                            'proposal_id': proposal_id,
                            'value': proposal_value,
                            'submitted_at': time.time() - start_time
                        })
                    
                    time.sleep(0.1)  # Small delay between proposals
        
        # Apply network events
        for event in network_events:
            if not isinstance(event, dict):
                continue
            
            event_time = event.get('time', 0)
            event_type = event.get('type', 'partition')
            
            # Wait until event time
            while time.time() - start_time < event_time and system.running:
                time.sleep(0.1)
            
            if event_type == 'partition':
                partition_groups = event.get('groups', [])
                if len(partition_groups) >= 2:
                    groups = [set(group) for group in partition_groups if isinstance(group, list)]
                    system.create_network_partition(groups)
            elif event_type == 'heal':
                system.heal_network_partition()
        
        # Run network simulation
        remaining_time = simulation_duration - (time.time() - start_time)
        if remaining_time > 0:
            system.simulate_network(remaining_time)
        
        # Get final system state
        final_state = system.get_system_state()
        
        # Analyze results
        active_nodes = final_state['active_nodes']
        leader_consensus = final_state['leader_consensus']
        total_proposals = sum(node.metrics['proposals_made'] for node in system.nodes.values())
        total_commits = sum(node.metrics['proposals_committed'] for node in system.nodes.values())
        
        if not leader_consensus:
            status = 'split_brain_detected'
        elif active_nodes < num_nodes * 0.5:
            status = 'majority_node_failure'
        elif total_commits == 0 and total_proposals > 0:
            status = 'consensus_failure'
        elif total_commits == total_proposals and total_proposals > 0:
            status = 'perfect_consensus'
        elif active_nodes == num_nodes and leader_consensus:
            status = 'healthy_cluster'
        else:
            status = 'partial_functionality'
        
        system.stop_system()
        
        return {
            'status': status,
            'final_system_state': final_state,
            'proposal_results': proposal_results,
            'consensus_metrics': {
                'total_proposals': total_proposals,
                'total_commits': total_commits,
                'commit_rate': total_commits / max(total_proposals, 1),
                'average_messages_per_node': sum(node.metrics['messages_sent'] for node in system.nodes.values()) / num_nodes
            },
            'simulation_summary': {
                'duration': simulation_duration,
                'network_events_applied': len(network_events),
                'final_leader_count': len(final_state['leaders'])
            }
        }
        
    except Exception as e:
        return {'error': f'distributed_system_simulation_failed: {str(e)}'}


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original target_function to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: target_function(system_config, simulation_params)
    
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
        d = 1000
    if b is None:
        b = -1000
    if c is None:
        c = 0
    if d is None:
        d = 1000
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-1000 <= param <= 1000):
                param = max(-1000, min(1000, param))  # Clamp to bounds
    
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
