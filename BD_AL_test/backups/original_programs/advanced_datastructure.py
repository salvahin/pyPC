def advanced_datastructure(a, b, c, d):
    """
    Complex nested data structure operations with dynamic access patterns.
    Designed to be extremely difficult for random test generation.
    Target: 15-25% max coverage, cyclomatic complexity ~35
    """
    # Initialize complex nested structure
    data = {
        'level1': {
            'items': [a, b, c, d],
            'matrix': [[0] * 4 for _ in range(4)],
            'flags': [False] * 4
        },
        'level2': {},
        'counter': 0,
        'state': 'init'
    }
    
    result = 0
    
    # First phase: structure initialization based on input relationships
    if a > 0 and b > 0:
        if a > b:
            data['level1']['matrix'][0][0] = a
            data['level1']['flags'][0] = True
            if c > a + b:
                data['level2']['special'] = c * 2
                data['state'] = 'advanced'
                if d > c:
                    data['level2']['nested'] = {'value': d, 'depth': 1}
                    result += d * 10
                else:
                    data['level2']['nested'] = {'value': c, 'depth': 2}
                    result += c * 5
            else:
                data['state'] = 'basic'
                if d < 0:
                    data['level1']['matrix'][1][1] = abs(d)
                    result += abs(d)
        elif b > a:
            data['level1']['matrix'][0][1] = b
            data['level1']['flags'][1] = True
            if c < 0:
                data['level2']['negative'] = abs(c)
                if d > 0:
                    data['counter'] = d % 4
                    if data['counter'] == 0:
                        data['state'] = 'zero_mod'
                        result += 100
                    elif data['counter'] == 1:
                        data['state'] = 'one_mod'
                        result += 50
                    else:
                        data['state'] = 'other_mod'
                        result += 25
                else:
                    data['level2']['both_neg'] = True
                    result -= 10
            else:
                data['level2']['positive'] = c
                result += c * 2
        else:  # a == b
            data['level1']['flags'][2] = True
            data['state'] = 'equal'
            if c > d:
                data['level2']['comparison'] = 'c_greater'
                result = (a + b) * c
            elif d > c:
                data['level2']['comparison'] = 'd_greater'
                result = (a + b) * d
            else:
                data['level2']['comparison'] = 'equal'
                result = a * b * 2
    elif a < 0 or b < 0:
        if a < 0 and b < 0:
            data['state'] = 'both_negative'
            sum_val = abs(a) + abs(b)
            if sum_val > 10:
                data['level1']['matrix'][2][2] = sum_val
                if c > 0 and d > 0:
                    data['level2']['recovery'] = True
                    result = sum_val + c + d
                elif c > 0:
                    data['level2']['partial'] = c
                    result = sum_val + c
                else:
                    result = -sum_val
            else:
                data['level1']['matrix'][3][3] = sum_val
                result = sum_val // 2
        elif a < 0:
            data['state'] = 'a_negative'
            if b > abs(a):
                data['level2']['b_dominates'] = True
                result = b - abs(a)
            else:
                data['level2']['a_dominates'] = True
                result = abs(a) - b
        else:  # b < 0
            data['state'] = 'b_negative'
            if a > abs(b):
                data['level2']['a_dominates'] = True
                result = a - abs(b)
            else:
                data['level2']['b_dominates'] = True
                result = abs(b) - a
    else:  # a == 0 or b == 0
        data['state'] = 'has_zero'
        if a == 0 and b == 0:
            if c != 0 or d != 0:
                data['level2']['fallback'] = True
                result = (c + d) * 10
            else:
                data['level2']['all_zero'] = True
                result = 1
        elif a == 0:
            data['level1']['flags'][3] = True
            result = b + c + d
        else:
            data['level1']['flags'][3] = True
            result = a + c + d
    
    # Second phase: matrix operations based on state
    if data['state'] == 'advanced':
        for i in range(4):
            for j in range(4):
                if i == j:
                    data['level1']['matrix'][i][j] = result % 10
        result *= 2
    elif data['state'] == 'equal':
        diagonal_sum = sum(data['level1']['matrix'][i][i] for i in range(4))
        if diagonal_sum > 0:
            result += diagonal_sum * 5
    
    # Third phase: flag-based modifications
    active_flags = sum(data['level1']['flags'])
    if active_flags == 0:
        result = 10
    elif active_flags == 1:
        result *= 3
    elif active_flags == 2:
        result *= 2
    elif active_flags == 3:
        result += 50
    else:
        result += 100
    
    # Fourth phase: nested dictionary access
    if 'nested' in data.get('level2', {}):
        nested_val = data['level2']['nested'].get('value', 0)
        nested_depth = data['level2']['nested'].get('depth', 0)
        if nested_depth == 1:
            result += nested_val * 2
        elif nested_depth == 2:
            result += nested_val
    
    # Fifth phase: final state-based adjustment
    if data['counter'] > 0:
        result = result * (5 - data['counter'])
    
    # Boundary checks
    if result > 1000:
        return 1000
    elif result < -1000:
        return -1000
    else:
        return result

if __name__ == '__main__':
    # Test with sample inputs
    test_cases = [
        (5, 3, 8, 2),
        (-2, -4, 1, 3),
        (0, 0, 0, 0),
        (10, 10, -5, 15)
    ]
    
    for a, b, c, d in test_cases:
        result = advanced_datastructure(a, b, c, d)
        print(f"Input: ({a}, {b}, {c}, {d}) -> Result: {result}")