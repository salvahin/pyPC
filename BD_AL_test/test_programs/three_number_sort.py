def three_number_sort(a,b,c):
    if a > b:
        a,b = b,a
    if a > c:
        a,c = c,a
    if b > c:
        b,c = c,b     
    print (a, "<", b, "<", c)


if __name__ == '__main__':
    a=6
    b=0
    c=1
    three_number_sort(a,b,c)


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original three_number_sort to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: three_number_sort(a, b, c)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from three_number_sort
    """
    # Set defaults for optional parameters based on function requirements
    if d is None:
        d = 999999
    if b is None:
        b = -999999
    if c is None:
        c = 0
    if d is None:
        d = 999999
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-999999 <= param <= 999999):
                param = max(-999999, min(999999, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return three_number_sort(a, b, c)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return three_number_sort(a, b)
            except:
                pass
            try:
                return three_number_sort(a)
            except:
                pass
            return three_number_sort(a)
        raise e
