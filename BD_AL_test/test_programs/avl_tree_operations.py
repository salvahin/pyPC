def avl_tree_operations(a, b, c, d):
    """
    AVL tree operations simulator with rebalancing logic.
    Simulates insertions, deletions, and rotations based on input values.
    Target: 20-30% coverage, cyclomatic complexity ~50
    """
    # Tree state representation (height, balance factor, operation count)
    tree_state = {
        'root_height': 0,
        'left_height': 0,
        'right_height': 0,
        'balance_factor': 0,
        'nodes': 0,
        'rotations': 0
    }
    
    operations = []
    result_code = 0
    
    # Convert inputs to operation codes
    ops = [int(a) % 100, int(b) % 100, int(c) % 100, int(d) % 100]
    
    # Process each operation
    for i, op_code in enumerate(ops):
        if op_code == 0:
            # Skip operation
            operations.append('skip')
            continue
        
        operation_type = determine_operation(op_code)
        operations.append(operation_type)
        
        if operation_type == 'insert':
            # Insert operation with balance checking
            value = op_code
            
            if tree_state['nodes'] == 0:
                # First insertion - simple case
                tree_state['nodes'] = 1
                tree_state['root_height'] = 1
                tree_state['balance_factor'] = 0
                result_code += value
            else:
                # Complex insertion with potential rebalancing
                insertion_result = simulate_insertion(tree_state, value, i)
                
                if insertion_result['requires_rebalance']:
                    # Determine rotation type needed
                    if insertion_result['balance_factor'] > 1:
                        # Left heavy
                        if insertion_result['left_balance'] >= 0:
                            # Left-Left case - single right rotation
                            tree_state['rotations'] += 1
                            result_code += perform_right_rotation(tree_state, value)
                        else:
                            # Left-Right case - double rotation
                            tree_state['rotations'] += 2
                            result_code += perform_left_right_rotation(tree_state, value)
                    elif insertion_result['balance_factor'] < -1:
                        # Right heavy
                        if insertion_result['right_balance'] <= 0:
                            # Right-Right case - single left rotation
                            tree_state['rotations'] += 1
                            result_code += perform_left_rotation(tree_state, value)
                        else:
                            # Right-Left case - double rotation
                            tree_state['rotations'] += 2
                            result_code += perform_right_left_rotation(tree_state, value)
                
                tree_state['nodes'] += 1
                update_heights(tree_state, insertion_result)
                
        elif operation_type == 'delete':
            # Delete operation - most complex case
            if tree_state['nodes'] > 0:
                value = op_code
                deletion_result = simulate_deletion(tree_state, value, i)
                
                if deletion_result['found']:
                    tree_state['nodes'] -= 1
                    result_code += deletion_result['replacement_value']
                    
                    # Post-deletion rebalancing
                    if deletion_result['requires_rebalance']:
                        rebalance_after_deletion(tree_state, deletion_result)
                    
                    # Special case: last node deleted
                    if tree_state['nodes'] == 0:
                        reset_tree_state(tree_state)
                        result_code += 1000  # Special completion bonus
                else:
                    result_code += 10  # Not found penalty
            else:
                result_code += 5  # Empty tree penalty
                
        elif operation_type == 'search':
            # Search operation with path-dependent complexity
            value = op_code
            search_result = simulate_search(tree_state, value, i)
            
            if search_result['found']:
                result_code += search_result['depth'] * 10
                
                # Update access patterns for optimizations
                if search_result['depth'] > tree_state['root_height'] // 2:
                    # Deep search - might trigger restructuring
                    if should_restructure(tree_state, search_result):
                        restructure_tree(tree_state, value)
                        result_code += 500
            else:
                result_code += search_result['comparisons']
                
        elif operation_type == 'range_query':
            # Range query - complex traversal simulation
            start_val = op_code
            end_val = (op_code + 20) % 100
            
            if start_val > end_val:
                start_val, end_val = end_val, start_val
            
            range_result = simulate_range_query(tree_state, start_val, end_val)
            result_code += range_result['nodes_visited'] * 5
            
            # Range queries might reveal imbalances
            if range_result['skewed_traversal']:
                if tree_state['nodes'] > 5:
                    # Large tree with skewed access - consider rebalancing
                    global_rebalance_result = perform_global_rebalance(tree_state)
                    tree_state['rotations'] += global_rebalance_result['total_rotations']
                    result_code += global_rebalance_result['improvement_factor'] * 100
        
        # Update tree statistics after each operation
        update_tree_statistics(tree_state)
    
    # Final tree quality assessment
    quality_score = assess_tree_quality(tree_state, operations)
    
    # Bonus for well-balanced final state
    if abs(tree_state['balance_factor']) <= 1 and tree_state['nodes'] > 0:
        result_code += quality_score * 50
    elif tree_state['nodes'] == 0 and len([op for op in operations if op == 'delete']) > 2:
        # Successfully deleted everything
        result_code += 2000
    
    return result_code % 100000

def determine_operation(op_code):
    """Determine operation type from code with specific ranges"""
    if op_code < 25:
        return 'insert'
    elif op_code < 45:
        return 'delete'
    elif op_code < 70:
        return 'search'
    elif op_code < 90:
        return 'range_query'
    else:
        return 'rebalance'

def simulate_insertion(tree_state, value, position):
    """Simulate AVL insertion with balance factor calculation"""
    # Simulate tree traversal to insertion point
    depth = estimate_insertion_depth(tree_state, value)
    
    # Calculate new balance factors
    left_subtree_change = 0
    right_subtree_change = 0
    
    if value % 2 == 0:
        # Even values tend to go left
        left_subtree_change = 1
        tree_state['left_height'] = max(tree_state['left_height'], depth)
    else:
        # Odd values tend to go right
        right_subtree_change = 1
        tree_state['right_height'] = max(tree_state['right_height'], depth)
    
    new_balance = tree_state['left_height'] - tree_state['right_height']
    
    return {
        'requires_rebalance': abs(new_balance) > 1,
        'balance_factor': new_balance,
        'left_balance': calculate_left_balance(tree_state, value),
        'right_balance': calculate_right_balance(tree_state, value),
        'insertion_depth': depth
    }

def estimate_insertion_depth(tree_state, value):
    """Estimate where a value would be inserted"""
    if tree_state['nodes'] == 0:
        return 1
    
    # Simplified depth estimation based on tree size and value
    estimated_depth = 1
    nodes_visited = 0
    
    while nodes_visited < tree_state['nodes'] and estimated_depth < 10:
        if value % (estimated_depth + 1) == 0:
            # Found insertion point
            break
        estimated_depth += 1
        nodes_visited += 1
    
    return min(estimated_depth, calculate_max_depth(tree_state['nodes']))

def calculate_max_depth(nodes):
    """Calculate maximum depth for AVL tree with given nodes"""
    if nodes <= 1:
        return nodes
    
    # AVL tree max depth is approximately 1.44 * log2(n+2) - 0.328
    import math
    return max(1, int(1.44 * math.log2(nodes + 2)))

def perform_right_rotation(tree_state, trigger_value):
    """Simulate right rotation and return cost"""
    rotation_cost = tree_state['left_height'] * 10
    
    # Update heights after rotation
    tree_state['balance_factor'] = max(tree_state['balance_factor'] - 2, -1)
    
    return rotation_cost + trigger_value

def perform_left_rotation(tree_state, trigger_value):
    """Simulate left rotation and return cost"""
    rotation_cost = tree_state['right_height'] * 10
    
    # Update heights after rotation
    tree_state['balance_factor'] = min(tree_state['balance_factor'] + 2, 1)
    
    return rotation_cost + trigger_value

def perform_left_right_rotation(tree_state, trigger_value):
    """Simulate double rotation (left then right)"""
    cost1 = perform_left_rotation(tree_state, trigger_value)
    cost2 = perform_right_rotation(tree_state, trigger_value)
    return cost1 + cost2

def perform_right_left_rotation(tree_state, trigger_value):
    """Simulate double rotation (right then left)"""
    cost1 = perform_right_rotation(tree_state, trigger_value)
    cost2 = perform_left_rotation(tree_state, trigger_value)
    return cost1 + cost2

def calculate_left_balance(tree_state, value):
    """Calculate balance factor of left subtree"""
    if tree_state['left_height'] == 0:
        return 0
    
    # Simplified calculation based on insertion pattern
    return (value % 3) - 1  # Can be -1, 0, or 1

def calculate_right_balance(tree_state, value):
    """Calculate balance factor of right subtree"""
    if tree_state['right_height'] == 0:
        return 0
    
    # Simplified calculation based on insertion pattern
    return 1 - (value % 3)  # Can be -1, 0, or 1

def update_heights(tree_state, insertion_result):
    """Update tree heights after insertion"""
    new_height = max(tree_state['left_height'], tree_state['right_height']) + 1
    tree_state['root_height'] = new_height
    
    # Recalculate balance factor
    tree_state['balance_factor'] = tree_state['left_height'] - tree_state['right_height']

def simulate_deletion(tree_state, value, position):
    """Simulate AVL deletion with complex case handling"""
    if tree_state['nodes'] == 0:
        return {'found': False, 'replacement_value': 0, 'requires_rebalance': False}
    
    # Simulate finding the node
    found = (value % 5) < tree_state['nodes'] % 5  # Probabilistic find
    
    if not found:
        return {'found': False, 'replacement_value': 0, 'requires_rebalance': False}
    
    # Determine deletion case
    deletion_case = value % 3
    replacement_value = 0
    requires_rebalance = True
    
    if deletion_case == 0:
        # Leaf node deletion - simplest case
        replacement_value = value
        requires_rebalance = tree_state['nodes'] > 3
    elif deletion_case == 1:
        # Node with one child
        replacement_value = value + position * 10
        requires_rebalance = True
    else:
        # Node with two children - most complex
        # Find inorder successor
        successor_value = find_inorder_successor(tree_state, value)
        replacement_value = successor_value
        requires_rebalance = True
    
    return {
        'found': True,
        'replacement_value': replacement_value,
        'requires_rebalance': requires_rebalance,
        'deletion_case': deletion_case
    }

def find_inorder_successor(tree_state, value):
    """Find inorder successor for complex deletion"""
    # Simplified successor finding
    base_successor = value + 1
    
    # Adjust based on tree structure
    if tree_state['right_height'] > tree_state['left_height']:
        return base_successor + tree_state['right_height']
    else:
        return base_successor + tree_state['nodes'] // 2

def rebalance_after_deletion(tree_state, deletion_result):
    """Rebalance tree after deletion"""
    if deletion_result['deletion_case'] == 2:  # Two children case
        # More likely to cause imbalance
        tree_state['rotations'] += 1
        
        # Simulate path from deletion point to root
        path_length = min(tree_state['root_height'], 5)
        for level in range(path_length):
            if should_rotate_at_level(tree_state, level):
                tree_state['rotations'] += 1
                break

def should_rotate_at_level(tree_state, level):
    """Determine if rotation needed at specific level"""
    # Complex heuristic based on tree state and level
    imbalance_threshold = 2
    level_factor = (level + 1) % 3
    
    return abs(tree_state['balance_factor']) > imbalance_threshold - level_factor

def simulate_search(tree_state, value, position):
    """Simulate tree search with realistic depth calculation"""
    if tree_state['nodes'] == 0:
        return {'found': False, 'depth': 0, 'comparisons': 1}
    
    # Estimate search depth based on tree balance and value
    max_depth = tree_state['root_height']
    estimated_depth = 1
    comparisons = 0
    
    # Simulate binary search path
    current_range = 100  # Assume values in range 0-99
    target_found = False
    
    while estimated_depth <= max_depth and not target_found:
        comparisons += 1
        
        # Simulate comparison with current node
        if value % current_range == position % current_range:
            target_found = True
        else:
            # Move to child
            estimated_depth += 1
            current_range //= 2
            
            if current_range < 1:
                break
    
    return {
        'found': target_found,
        'depth': estimated_depth,
        'comparisons': comparisons
    }

def should_restructure(tree_state, search_result):
    """Determine if tree restructuring is beneficial"""
    # Restructure if search was unusually deep
    avg_expected_depth = max(1, tree_state['root_height'] * 0.7)
    
    return search_result['depth'] > avg_expected_depth + 2

def restructure_tree(tree_state, trigger_value):
    """Perform tree restructuring for better balance"""
    old_rotations = tree_state['rotations']
    
    # Simulate global rebalancing
    tree_state['rotations'] += max(1, tree_state['nodes'] // 3)
    tree_state['balance_factor'] = 0  # Perfect balance after restructuring
    
    # Update heights to optimal
    import math
    optimal_height = max(1, int(math.log2(tree_state['nodes'] + 1)))
    tree_state['root_height'] = optimal_height
    tree_state['left_height'] = optimal_height - 1
    tree_state['right_height'] = optimal_height - 1

def simulate_range_query(tree_state, start_val, end_val):
    """Simulate range query with traversal pattern analysis"""
    if tree_state['nodes'] == 0:
        return {'nodes_visited': 0, 'skewed_traversal': False}
    
    range_size = end_val - start_val + 1
    estimated_results = min(range_size, tree_state['nodes'])
    
    # Simulate in-order traversal for range
    nodes_visited = estimated_results
    
    # Add overhead for tree navigation
    navigation_overhead = max(1, tree_state['root_height'])
    nodes_visited += navigation_overhead
    
    # Detect skewed traversal patterns
    skewed = False
    if range_size > tree_state['nodes'] // 2:
        # Large range query
        skewed = abs(tree_state['balance_factor']) > 1
    
    return {
        'nodes_visited': nodes_visited,
        'skewed_traversal': skewed,
        'results_found': estimated_results
    }

def perform_global_rebalance(tree_state):
    """Perform complete tree rebalancing"""
    nodes = tree_state['nodes']
    
    if nodes <= 2:
        return {'total_rotations': 0, 'improvement_factor': 0}
    
    # Calculate rotations needed for perfect balance
    import math
    optimal_height = int(math.log2(nodes + 1))
    current_inefficiency = tree_state['root_height'] - optimal_height
    
    rotations_needed = max(0, current_inefficiency)
    improvement = current_inefficiency * 2 if current_inefficiency > 0 else 1
    
    # Apply rebalancing
    tree_state['root_height'] = optimal_height
    tree_state['balance_factor'] = 0
    
    return {
        'total_rotations': rotations_needed,
        'improvement_factor': improvement
    }

def update_tree_statistics(tree_state):
    """Update various tree statistics"""
    # Keep balance factor in valid AVL range
    tree_state['balance_factor'] = max(-2, min(2, tree_state['balance_factor']))
    
    # Ensure heights are consistent
    if tree_state['nodes'] == 0:
        reset_tree_state(tree_state)

def reset_tree_state(tree_state):
    """Reset tree to empty state"""
    tree_state['root_height'] = 0
    tree_state['left_height'] = 0
    tree_state['right_height'] = 0
    tree_state['balance_factor'] = 0

def assess_tree_quality(tree_state, operations):
    """Assess overall tree quality based on operations performed"""
    if tree_state['nodes'] == 0:
        return 10  # Clean slate bonus
    
    # Quality factors
    balance_quality = 10 - abs(tree_state['balance_factor']) * 3
    efficiency_quality = max(0, 10 - tree_state['rotations'])
    
    # Operation diversity bonus
    unique_ops = len(set(operations))
    diversity_bonus = unique_ops * 2
    
    return max(0, balance_quality + efficiency_quality + diversity_bonus)

if __name__ == '__main__':
    # Test cases for different operation patterns
    print("Testing AVL tree operations:")
    
    # Balanced insertions
    result1 = avl_tree_operations(10, 35, 60, 85)
    print(f"Mixed operations: {result1}")
    
    # Heavy insertion pattern
    result2 = avl_tree_operations(5, 15, 25, 35)
    print(f"Insert heavy: {result2}")
    
    # Delete-heavy pattern
    result3 = avl_tree_operations(30, 40, 50, 60)
    print(f"Delete heavy: {result3}")
    
    # Search and range queries
    result4 = avl_tree_operations(65, 75, 85, 95)
    print(f"Query heavy: {result4}")