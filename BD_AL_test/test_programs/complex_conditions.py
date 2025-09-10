def complex_conditions(a, b, c, d):
    score = 0
    level = 0
    bonus = 0
    
    # Complex nested conditions
    if a > 0 and b > 0:
        if a > b:
            score = a * 2
            if c > 0:
                score += c
                if d > c:
                    bonus = d - c
                    level = 3
                else:
                    bonus = c // 2
                    level = 2
            else:
                if d > 0:
                    score += d * 2
                    level = 2
                else:
                    score = score // 2
                    level = 1
        elif b > a:
            score = b * 3
            if c < 0:
                score -= abs(c)
                if d < c:
                    bonus = abs(d) * 2
                    level = 4
                else:
                    bonus = abs(c)
                    level = 3
            else:
                if d > b:
                    score += d
                    bonus = d - b
                    level = 5
                else:
                    score += c
                    level = 2
        else:
            score = a + b
            if c == 0:
                if d != 0:
                    score *= abs(d)
                    level = 2
                else:
                    score = 10
                    level = 1
            else:
                score += c
                level = 1
    elif a < 0 or b < 0:
        if a < 0 and b < 0:
            score = abs(a * b)
            if c > 0 and d > 0:
                score += c + d
                level = 6
            elif c > 0 or d > 0:
                if c > d:
                    score += c
                else:
                    score += d
                level = 4
            else:
                score = score // 4
                level = 1
        elif a < 0:
            score = abs(a) + b
            if c > b:
                score *= 2
                level = 3
            else:
                score += 5
                level = 2
        else:
            score = a + abs(b)
            if d < a:
                score *= 3
                level = 4
            else:
                score += 10
                level = 2
    else:
        if c != 0 or d != 0:
            if c > d:
                score = c * 10
                level = 2
            elif d > c:
                score = d * 10
                level = 3
            else:
                score = (c + d) * 5
                level = 1
        else:
            score = 1
            level = 0
    
    # Apply bonus and level modifiers
    final_score = score + bonus
    if level > 4:
        final_score *= 2
    elif level > 2:
        final_score += 50
    elif level > 0:
        final_score += 10
    
    return final_score

if __name__ == '__main__':
    result = complex_conditions(5, -3, 2, 7)
    print(f"Final score: {result}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original complex_conditions to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: complex_conditions(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from complex_conditions
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
        return complex_conditions(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return complex_conditions(a, b, c)
            except:
                pass
            try:
                return complex_conditions(a, b)
            except:
                pass
            try:
                return complex_conditions(a)
            except:
                pass
            return complex_conditions(a)
        raise e
