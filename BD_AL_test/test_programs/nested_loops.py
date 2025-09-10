def nested_loops(a, b, c, d):
    result = 0
    counter = 0
    
    if a > 0:
        for i in range(int(abs(a)) % 5 + 1):
            if i > 2:
                for j in range(int(abs(b)) % 3 + 1):
                    if j > 1:
                        result += i * j
                        counter += 1
                    else:
                        result += i
            else:
                if b > 0:
                    result += i * 2
                else:
                    result -= i
    
    if result > 10:
        if c > 0:
            for k in range(int(abs(c)) % 4 + 1):
                if k % 2 == 0:
                    result *= 2
                    if result > 100:
                        break
                else:
                    result += k
                    if d > 0:
                        counter += k
        else:
            result = result // 2
    elif result < -10:
        if d < 0:
            result = abs(result)
        else:
            for m in range(3):
                if m > 0:
                    result += m * d
    else:
        if counter > 5:
            result = counter * 10
        elif counter > 2:
            result = counter * 5
        else:
            result = 1
    
    if result > 1000:
        return 1000
    elif result < 0:
        return 0
    else:
        return result

if __name__ == '__main__':
    test_result = nested_loops(4, 3, 2, 1)
    print(f"Result: {test_result}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original nested_loops to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: nested_loops(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from nested_loops
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
        return nested_loops(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return nested_loops(a, b, c)
            except:
                pass
            try:
                return nested_loops(a, b)
            except:
                pass
            try:
                return nested_loops(a)
            except:
                pass
            return nested_loops(a)
        raise e
