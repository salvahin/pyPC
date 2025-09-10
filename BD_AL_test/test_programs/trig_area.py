from cmath import sqrt

def trig_area(a, b, c):
    match = p = s = h = 0
    if ((a+b)>=c and b+c >= a) and ((a+c >= b and a > 0) and (b > 0 and c >0)):
        if a == b:
            match += 1
        if a == c:
            match += 2
        if b == c:
            match += 3
        if match == 0:
            p = (a+b+c)/2
            s = sqrt(p * (p-a) * (p-b) * (p-c))
            print("Anomalistic")
        elif match == 1:
            h = sqrt((a**2) - ((c/2)**2))
            s = c*h/2
            print("Isoceles and First=Second")
        elif match == 2:
            h = sqrt((a**2) - ((b/2)**2))
            s = b*h/2
            print("Isoceles and First=Third")
        elif match == 3:
            h = sqrt((b**2) - ((a/2)**2))
            s = a*h/2
            print("Isoceles and Second=Third")
        else:
            s = sqrt(3) * a * a/4
            print("Equilateral")
    else:
        print("Not a Triangle")
    return s


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original trig_area to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: trig_area(a, b, c)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from trig_area
    """
    # Set defaults for optional parameters based on function requirements
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
        return trig_area(a, b, c)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return trig_area(a, b)
            except:
                pass
            try:
                return trig_area(a)
            except:
                pass
            return trig_area(a)
        raise e
