def pattern_matcher(a, b, c, d):
    """
    Complex pattern matching with multiple strategies and backtracking.
    Simulates string pattern matching using numeric inputs.
    Target: 20-30% max coverage, cyclomatic complexity ~30
    """
    # Convert inputs to pattern components
    pattern_type = abs(a) % 5  # 0-4 different pattern types
    pattern_len = abs(b) % 8 + 2  # Pattern length 2-9
    match_mode = abs(c) % 3  # 0: exact, 1: fuzzy, 2: wildcard
    threshold = abs(d) % 10  # Matching threshold
    
    result = 0
    matches = 0
    state = 'init'
    
    # Pattern type selection creates different matching strategies
    if pattern_type == 0:
        # Sequential pattern
        state = 'sequential'
        if pattern_len < 5:
            if match_mode == 0:
                # Exact sequential match
                if threshold < 3:
                    matches = pattern_len * 2
                    result = 100
                else:
                    matches = pattern_len
                    result = 50
            elif match_mode == 1:
                # Fuzzy sequential
                if threshold < 5:
                    matches = pattern_len - 1
                    result = 40
                else:
                    matches = pattern_len // 2
                    result = 20
            else:
                # Wildcard sequential
                matches = pattern_len + threshold
                result = 30
        else:
            # Long pattern
            if match_mode == 0 and threshold == 0:
                # Special case: perfect long match
                state = 'perfect_long'
                matches = pattern_len * 3
                result = 200
            elif match_mode == 1:
                matches = pattern_len - 2
                result = 35
            else:
                matches = pattern_len // 3
                result = 15
                
    elif pattern_type == 1:
        # Alternating pattern
        state = 'alternating'
        if a > 0 and b > 0:
            if match_mode == 0:
                if threshold % 2 == 0:
                    matches = pattern_len
                    result = 80
                else:
                    matches = pattern_len - 1
                    result = 60
            elif match_mode == 1:
                matches = pattern_len // 2 + 1
                result = 45
            else:
                if threshold > 5:
                    state = 'alt_wildcard_high'
                    matches = pattern_len * 2
                    result = 90
                else:
                    matches = pattern_len
                    result = 55
        else:
            # Negative alternating
            if c > 0:
                matches = abs(a) + abs(b)
                result = 25
            else:
                matches = 1
                result = 10
                
    elif pattern_type == 2:
        # Nested pattern
        state = 'nested'
        depth = threshold % 4 + 1
        
        if depth == 1:
            if match_mode == 0:
                matches = pattern_len
                result = 70
            else:
                matches = pattern_len - 1
                result = 50
        elif depth == 2:
            if match_mode == 1:
                if pattern_len > 4:
                    state = 'nested_deep_fuzzy'
                    matches = pattern_len * depth
                    result = 120
                else:
                    matches = pattern_len + depth
                    result = 65
            else:
                matches = pattern_len
                result = 45
        elif depth == 3:
            if match_mode == 2:
                # Deep wildcard nesting
                state = 'deep_wildcard'
                matches = pattern_len * depth * 2
                result = 150
            else:
                matches = pattern_len + depth - 1
                result = 75
        else:
            # Maximum depth
            if threshold < 3:
                state = 'max_depth_low'
                matches = pattern_len * 4
                result = 180
            else:
                matches = pattern_len * 2
                result = 95
                
    elif pattern_type == 3:
        # Recursive pattern
        state = 'recursive'
        recursion_level = threshold % 3 + 1
        
        if recursion_level == 1:
            if pattern_len < 4:
                matches = pattern_len * 2
                result = 85
            else:
                if match_mode == 0:
                    state = 'recursive_exact'
                    matches = pattern_len * 3
                    result = 140
                else:
                    matches = pattern_len
                    result = 70
        elif recursion_level == 2:
            if match_mode == 1 and threshold > 4:
                state = 'recursive_fuzzy_high'
                matches = pattern_len * recursion_level * 2
                result = 160
            else:
                matches = pattern_len + recursion_level
                result = 88
        else:
            # Max recursion
            if pattern_len > 5 and match_mode == 2:
                state = 'max_recursion_wildcard'
                matches = pattern_len * 5
                result = 250
            else:
                matches = pattern_len * 2
                result = 110
                
    else:
        # Pattern type 4: Complex composite
        state = 'composite'
        
        if a > 0 and b > 0 and c > 0 and d > 0:
            # All positive - best case
            state = 'composite_optimal'
            matches = pattern_len * 4
            result = 300
        elif a > 0 and b > 0:
            if match_mode == 0:
                matches = pattern_len * 2
                result = 130
            elif match_mode == 1:
                if threshold < 6:
                    matches = pattern_len + threshold
                    result = 105
                else:
                    matches = pattern_len
                    result = 78
            else:
                matches = pattern_len * 3 // 2
                result = 92
        elif a < 0 and b < 0:
            # Both negative
            if c > 0 or d > 0:
                matches = abs(a) + abs(b)
                result = 48
            else:
                state = 'composite_all_neg'
                matches = 0
                result = 5
        else:
            # Mixed signs
            matches = pattern_len // 2
            result = 33
    
    # Backtracking simulation based on matches
    if matches > pattern_len * 2:
        # Need backtracking
        backtrack_steps = matches - pattern_len * 2
        if backtrack_steps > 5:
            result = result * 2  # Successful complex backtrack
        else:
            result = result + backtrack_steps * 10
    
    # Apply state-based final adjustments
    if 'perfect' in state or 'optimal' in state:
        result = min(result * 2, 500)
    elif 'deep' in state or 'max' in state:
        result = min(result + 100, 400)
    elif 'neg' in state:
        result = max(result // 2, 1)
    
    return min(max(result, 0), 500)

if __name__ == '__main__':
    # Test with sample inputs
    test_cases = [
        (3, 5, 1, 7),
        (8, -2, 0, 4),
        (-3, -5, 2, 9),
        (0, 6, 2, 3)
    ]
    
    for a, b, c, d in test_cases:
        result = pattern_matcher(a, b, c, d)
        print(f"Input: ({a}, {b}, {c}, {d}) -> Result: {result}")