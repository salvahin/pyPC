def bubbleSort(a, b, c, d):
    n = 4
    arr = [a, b, c, d]
    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j + 1]:
                swapped = True
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original bubbleSort to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: bubbleSort(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from bubbleSort
    """
    # Set defaults for optional parameters based on function requirements
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
        return bubbleSort(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return bubbleSort(a, b, c)
            except:
                pass
            try:
                return bubbleSort(a, b)
            except:
                pass
            try:
                return bubbleSort(a)
            except:
                pass
            return bubbleSort(a)
        raise e
