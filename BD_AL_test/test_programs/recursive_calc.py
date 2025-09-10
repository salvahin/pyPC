def recursive_calc(a, b, c, d, depth=0):
    if depth > 5:
        return 0
    
    result = 0
    
    if a > 10:
        if b > 5:
            result = a + b
            if c > 0:
                # Recursive call with modified parameters
                sub_result = recursive_calc(a//2, b//2, c-1, d, depth+1)
                result += sub_result
            else:
                result *= 2
        else:
            if c > b:
                result = a - b + c
            else:
                result = a // 2
    elif a > 0:
        if b < 0:
            result = abs(b) * 2
            if c > 0 and d > 0:
                # Another recursive branch
                sub_result = recursive_calc(c, d, a, b, depth+1)
                result += sub_result // 2
            elif c > 0:
                result += c * 3
            else:
                result -= d
        else:
            if b == 0:
                result = a * 10
            else:
                result = a + b
                if c > a + b:
                    # Deep recursion possibility
                    sub_result = recursive_calc(b, c, d, a, depth+1)
                    result += sub_result
    else:
        if a == 0:
            if b > 0 and c > 0:
                result = b * c
                if d > result:
                    result = d
            elif b > 0:
                result = b * 5
            elif c > 0:
                result = c * 3
            else:
                result = abs(d)
        else:
            # a < 0
            if b > abs(a):
                result = b - a
                if depth < 3:
                    # Limited recursion
                    sub_result = recursive_calc(abs(a), b, c//2, d//2, depth+1)
                    result += sub_result // 4
            else:
                result = abs(a * 2)
    
    # Additional complexity based on depth
    if depth == 0:
        if result > 100:
            result = 100
        elif result < -100:
            result = -100
    elif depth < 3:
        if result > 50:
            result = result // 2
    else:
        if result > 25:
            result = result // 4
    
    return result

def test_recursive(a, b, c, d):
    result = recursive_calc(a, b, c, d)
    
    # Additional processing based on result
    if result > 50:
        if a > 0 and b > 0:
            return result * 2
        else:
            return result + 50
    elif result > 0:
        if c > 0 or d > 0:
            return result + 10
        else:
            return result
    elif result == 0:
        return 1
    else:
        return abs(result)

if __name__ == '__main__':
    result = test_recursive(15, 8, 4, 2)
    print(f"Final result: {result}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original recursive_calc to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: recursive_calc(a, b, c, d, depth)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from recursive_calc
    """
    # Set defaults for optional parameters based on function requirements
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
        return recursive_calc(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return recursive_calc(a, b, c)
            except:
                pass
            try:
                return recursive_calc(a, b)
            except:
                pass
            try:
                return recursive_calc(a)
            except:
                pass
            return recursive_calc(a)
        raise e
