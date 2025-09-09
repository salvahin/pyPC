def constraint_solver(a, b, c, d):
    """
    Constraint satisfaction problem solver with propagation and backtracking.
    Multiple constraint types with conflict detection.
    Target: 10-20% max coverage, cyclomatic complexity ~40
    """
    # Initialize constraint system
    constraints = {
        'equality': [],
        'inequality': [],
        'range': [],
        'dependency': []
    }
    
    variables = {'x': a, 'y': b, 'z': c, 'w': d}
    satisfied = 0
    conflicts = 0
    state = 'initial'
    solution = 0
    
    # Define constraints based on input values
    if a != 0:
        if a > 0:
            constraints['inequality'].append(('x', '>', 0))
            if b > a:
                constraints['inequality'].append(('y', '>', 'x'))
                constraints['dependency'].append(('y', 'depends_on', 'x'))
                state = 'ordered'
                if c > b:
                    constraints['inequality'].append(('z', '>', 'y'))
                    if d > c:
                        # Fully ordered sequence
                        state = 'fully_ordered'
                        constraints['inequality'].append(('w', '>', 'z'))
                        satisfied += 4
                        solution = a + b + c + d
                    else:
                        constraints['inequality'].append(('w', '<=', 'z'))
                        satisfied += 3
                        solution = a + b + c
                else:
                    if d > 0:
                        constraints['range'].append(('z', 0, b))
                        satisfied += 2
                        solution = a + b + d
                    else:
                        conflicts += 1
                        solution = a + b
            elif b == a:
                constraints['equality'].append(('x', '==', 'y'))
                state = 'equal_xy'
                if c == b:
                    if d == c:
                        # All equal
                        state = 'all_equal'
                        constraints['equality'].append(('z', '==', 'y'))
                        constraints['equality'].append(('w', '==', 'z'))
                        satisfied += 3
                        solution = a * 4
                    else:
                        constraints['equality'].append(('z', '==', 'y'))
                        satisfied += 2
                        solution = a * 3 + d
                else:
                    if abs(c - a) < 3:
                        constraints['range'].append(('z', a - 3, a + 3))
                        satisfied += 1
                        solution = a * 2 + c
                    else:
                        conflicts += 1
                        solution = a * 2
            else:
                # b < a
                constraints['inequality'].append(('y', '<', 'x'))
                state = 'reverse_ordered'
                if c < b:
                    constraints['inequality'].append(('z', '<', 'y'))
                    if d < c:
                        state = 'fully_reverse'
                        constraints['inequality'].append(('w', '<', 'z'))
                        satisfied += 4
                        solution = abs(a - b - c - d)
                    else:
                        conflicts += 1
                        solution = abs(a - b - c)
                else:
                    constraints['range'].append(('z', b, a))
                    satisfied += 1
                    solution = a - b + c
        else:
            # a < 0
            constraints['inequality'].append(('x', '<', 0))
            state = 'negative_x'
            if b < 0:
                constraints['inequality'].append(('y', '<', 0))
                if c < 0 and d < 0:
                    state = 'all_negative'
                    constraints['inequality'].append(('z', '<', 0))
                    constraints['inequality'].append(('w', '<', 0))
                    satisfied += 4
                    solution = abs(a + b + c + d)
                elif c < 0:
                    satisfied += 3
                    solution = abs(a + b + c) + d
                else:
                    if c > abs(a) + abs(b):
                        state = 'recovery'
                        satisfied += 2
                        solution = c - abs(a) - abs(b)
                    else:
                        conflicts += 1
                        solution = abs(a + b)
            else:
                # Mixed signs
                state = 'mixed_signs'
                if b > abs(a):
                    constraints['inequality'].append(('y', '>', 'abs(x)'))
                    satisfied += 1
                    solution = b - abs(a)
                else:
                    conflicts += 1
                    solution = abs(a) - b
    else:
        # a == 0
        constraints['equality'].append(('x', '==', 0))
        state = 'zero_x'
        if b == 0:
            constraints['equality'].append(('y', '==', 0))
            if c == 0 and d == 0:
                state = 'all_zero'
                solution = 1
            elif c == 0:
                state = 'xyz_zero'
                solution = abs(d) * 10
            else:
                solution = abs(c) + abs(d)
        else:
            if b > 0:
                constraints['inequality'].append(('y', '>', 'x'))
                solution = b + c + d
            else:
                constraints['inequality'].append(('y', '<', 'x'))
                solution = abs(b) + c + d
    
    # Constraint propagation phase
    if len(constraints['equality']) > 0:
        # Propagate equality constraints
        for constraint in constraints['equality']:
            if constraint[1] == '==':
                satisfied += 1
        
        if len(constraints['equality']) > 2:
            # Check for consistency
            if state == 'all_equal' or state == 'all_zero':
                solution *= 2
            else:
                conflicts += 1
    
    if len(constraints['inequality']) > 0:
        # Check inequality chains
        if len(constraints['inequality']) >= 3:
            if state in ['fully_ordered', 'fully_reverse']:
                # Valid chain
                solution += 50
            else:
                # Potential conflict
                for constraint in constraints['inequality']:
                    if constraint[1] in ['>', '<']:
                        satisfied += 1
                    
                if satisfied < len(constraints['inequality']):
                    conflicts += 1
    
    if len(constraints['range']) > 0:
        # Validate range constraints
        for constraint in constraints['range']:
            var, min_val, max_val = constraint
            if var == 'z' and min_val <= variables['z'] <= max_val:
                satisfied += 1
                solution += 20
            else:
                conflicts += 1
    
    if len(constraints['dependency']) > 0:
        # Handle dependencies
        for dep in constraints['dependency']:
            if dep[1] == 'depends_on':
                if state in ['ordered', 'fully_ordered']:
                    solution += 30
                else:
                    conflicts += 1
    
    # Conflict resolution phase
    if conflicts > 0:
        if conflicts == 1:
            # Minor conflict, can recover
            solution = solution // 2 + 10
        elif conflicts == 2:
            # Moderate conflict
            solution = solution // 4 + 5
        else:
            # Major conflict
            solution = max(1, solution // 10)
    
    # Backtracking simulation
    if satisfied < 2 and conflicts > 2:
        # Need backtracking
        backtrack_depth = min(conflicts, 4)
        if backtrack_depth == 4 and state == 'mixed_signs':
            # Deep backtrack successful
            solution = 100
        elif backtrack_depth >= 3:
            solution = 50
        else:
            solution = 20
    
    # Final state-based adjustments
    if state == 'fully_ordered' and conflicts == 0:
        solution = min(solution * 3, 500)
    elif state == 'all_equal' and satisfied >= 3:
        solution = min(solution * 2, 400)
    elif state in ['all_negative', 'all_zero']:
        solution = max(solution, 10)
    elif 'recovery' in state:
        solution += 75
    
    # Apply satisfaction bonus
    if satisfied >= 4:
        solution += 100
    elif satisfied >= 3:
        solution += 50
    elif satisfied >= 2:
        solution += 25
    
    return min(max(solution, 0), 500)

if __name__ == '__main__':
    # Test with sample inputs
    test_cases = [
        (1, 2, 3, 4),    # Fully ordered
        (5, 5, 5, 5),    # All equal
        (-2, -3, -4, -5), # All negative
        (0, 0, 0, 1),    # Mostly zero
        (3, 1, 4, 2)     # Mixed ordering
    ]
    
    for a, b, c, d in test_cases:
        result = constraint_solver(a, b, c, d)
        print(f"Input: ({a}, {b}, {c}, {d}) -> Result: {result}")