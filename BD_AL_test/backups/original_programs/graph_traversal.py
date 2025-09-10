def graph_traversal(a, b, c, d):
    """
    Graph construction and traversal with cycle detection and path finding.
    Builds graph from parameters and performs various traversal strategies.
    Target: 15-25% max coverage, cyclomatic complexity ~35
    """
    # Build graph structure from inputs
    num_nodes = abs(a) % 6 + 3  # 3-8 nodes
    edge_density = abs(b) % 4  # 0-3 density levels
    traversal_type = abs(c) % 3  # 0: DFS, 1: BFS, 2: Custom
    target_node = abs(d) % num_nodes
    
    # Initialize graph
    graph = {i: [] for i in range(num_nodes)}
    visited = [False] * num_nodes
    path = []
    cycles = 0
    result = 0
    state = 'init'
    
    # Build edges based on density
    if edge_density == 0:
        # Sparse graph - linear chain
        state = 'sparse'
        for i in range(num_nodes - 1):
            graph[i].append(i + 1)
        if a > 0:
            graph[num_nodes - 1].append(0)  # Create cycle
            cycles = 1
    elif edge_density == 1:
        # Medium density - tree-like
        state = 'medium'
        for i in range(num_nodes):
            if i * 2 + 1 < num_nodes:
                graph[i].append(i * 2 + 1)
            if i * 2 + 2 < num_nodes:
                graph[i].append(i * 2 + 2)
        if b > 0 and c > 0:
            # Add back edge for cycle
            graph[num_nodes - 1].append(1)
            cycles = 1
    elif edge_density == 2:
        # Dense - multiple connections
        state = 'dense'
        for i in range(num_nodes):
            for j in range(i + 1, min(i + 3, num_nodes)):
                graph[i].append(j)
            if i > 0 and d > 0:
                graph[i].append((i - 1) % num_nodes)
                cycles += 1
    else:
        # Very dense - almost complete
        state = 'complete'
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(i - j) <= 2:
                    graph[i].append(j)
        cycles = num_nodes  # Many cycles in complete graph
    
    # Perform traversal based on type
    if traversal_type == 0:
        # DFS traversal
        stack = [0]
        depth = 0
        max_depth = 0
        
        while stack and depth < num_nodes * 2:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                path.append(node)
                depth += 1
                max_depth = max(max_depth, depth)
                
                if node == target_node:
                    state += '_target_found'
                    result += 100
                
                for neighbor in reversed(graph[node]):
                    if not visited[neighbor]:
                        stack.append(neighbor)
                    elif neighbor in path[:-1]:
                        cycles += 1
        
        if max_depth > num_nodes // 2:
            result += max_depth * 10
        else:
            result += max_depth * 5
            
    elif traversal_type == 1:
        # BFS traversal
        queue = [0]
        level = 0
        nodes_at_level = {0: [0]}
        
        while queue and level < num_nodes:
            next_queue = []
            level += 1
            nodes_at_level[level] = []
            
            for node in queue:
                if not visited[node]:
                    visited[node] = True
                    path.append(node)
                    
                    if node == target_node:
                        state += '_target_at_level'
                        result += 80 + level * 10
                    
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            next_queue.append(neighbor)
                            nodes_at_level[level].append(neighbor)
                        elif neighbor in path[:-1]:
                            cycles += 1
            
            queue = next_queue
        
        # Bonus for balanced tree structure
        if len(nodes_at_level) > 2:
            balanced = True
            for i in range(1, len(nodes_at_level) - 1):
                expected = 2 ** i
                actual = len(nodes_at_level.get(i, []))
                if abs(actual - expected) > 2:
                    balanced = False
                    break
            if balanced:
                state += '_balanced'
                result += 50
                
    else:
        # Custom traversal - bidirectional search
        state += '_bidirectional'
        forward = [0]
        backward = [target_node]
        forward_visited = {0}
        backward_visited = {target_node}
        meeting_point = -1
        steps = 0
        
        while forward and backward and steps < num_nodes:
            steps += 1
            
            # Forward step
            next_forward = []
            for node in forward:
                for neighbor in graph[node]:
                    if neighbor in backward_visited:
                        meeting_point = neighbor
                        break
                    if neighbor not in forward_visited:
                        forward_visited.add(neighbor)
                        next_forward.append(neighbor)
                if meeting_point >= 0:
                    break
            
            if meeting_point >= 0:
                state += '_met'
                result += 150 + (num_nodes - steps) * 20
                break
            
            # Backward step
            next_backward = []
            for node in backward:
                # Find nodes that point to current node
                for src in range(num_nodes):
                    if node in graph[src]:
                        if src in forward_visited:
                            meeting_point = src
                            break
                        if src not in backward_visited:
                            backward_visited.add(src)
                            next_backward.append(src)
                if meeting_point >= 0:
                    break
            
            if meeting_point >= 0:
                state += '_met_reverse'
                result += 120 + (num_nodes - steps) * 15
                break
            
            forward = next_forward
            backward = next_backward
        
        if meeting_point < 0 and target_node in forward_visited:
            result += 60
        elif meeting_point < 0:
            result += 20
    
    # Cycle detection bonus
    if cycles > 0:
        if cycles == 1:
            result += 30
        elif cycles < num_nodes:
            result += cycles * 15
        else:
            # Too many cycles
            result += 50
    
    # Path analysis
    if len(path) == num_nodes:
        # Visited all nodes
        state += '_complete'
        result += 100
    elif len(path) > num_nodes * 0.75:
        result += 50
    elif len(path) > num_nodes * 0.5:
        result += 25
    
    # Special patterns
    if a > 0 and b > 0 and c > 0 and d == target_node:
        # Perfect alignment
        state += '_aligned'
        result *= 2
    elif a < 0 and b < 0:
        # Negative inputs penalty
        result = result // 2
    
    # State-based final adjustment
    if 'target' in state and 'complete' in state:
        result = min(result + 100, 500)
    elif 'balanced' in state or 'met' in state:
        result = min(result + 50, 450)
    
    return min(max(result, 0), 500)

if __name__ == '__main__':
    # Test with sample inputs
    test_cases = [
        (4, 2, 0, 3),    # Medium graph, DFS
        (6, 3, 1, 5),    # Dense graph, BFS
        (3, 1, 2, 2),    # Sparse graph, bidirectional
        (-2, -1, 1, 4),  # Negative inputs
        (5, 2, 2, 5)     # Custom traversal
    ]
    
    for a, b, c, d in test_cases:
        result = graph_traversal(a, b, c, d)
        print(f"Input: ({a}, {b}, {c}, {d}) -> Result: {result}")