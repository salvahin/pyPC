def cryptographic_hash(a, b, c, d):
    """
    Simplified cryptographic hash function with multiple rounds and state transformations.
    Based on real hash algorithms (SHA-like structure).
    Target: 15-25% coverage, high cyclomatic complexity ~45
    """
    # Initialize state with magic numbers (simplified SHA constants)
    h0 = 0x67452301 + int(a) if a != 0 else 0x67452301
    h1 = 0xEFCDAB89 + int(b) if b != 0 else 0xEFCDAB89
    h2 = 0x98BADCFE + int(c) if c != 0 else 0x98BADCFE
    h3 = 0x10325476 + int(d) if d != 0 else 0x10325476
    
    # Input preprocessing with multiple paths
    message_blocks = []
    if abs(a) < 256 and abs(b) < 256:
        # Small input path - requires specific range
        message_blocks = [int(a) & 0xFF, int(b) & 0xFF, int(c) & 0xFF, int(d) & 0xFF]
        if a > 0 and b > 0:
            # Positive small values - rare path
            if c > a and d > b:
                # Ascending pattern - very specific condition
                message_blocks.append(0x80)  # Padding marker
                h0 ^= 0x5A827999  # Round constant
            else:
                message_blocks.append(0x40)
        elif a < 0 or b < 0:
            # Mixed sign values
            if abs(a) > abs(b):
                message_blocks.append(0xC0)
                h1 ^= 0x6ED9EBA1
            else:
                message_blocks.append(0xA0)
        else:
            # Zero values
            message_blocks.append(0x00)
            if c == 0 and d == 0:
                # All small or zero - special case
                h2 ^= 0x8F1BBCDC
                h3 ^= 0xCA62C1D6
    else:
        # Large input path
        message_blocks = [
            (int(a) >> 8) & 0xFF, int(a) & 0xFF,
            (int(b) >> 8) & 0xFF, int(b) & 0xFF
        ]
        
        # Complex state-dependent processing
        temp = int(c) ^ int(d)
        if temp > 1000:
            # High magnitude XOR - uncommon
            if (temp & 0x1) == 0:
                # Even XOR result - nested condition
                h0 = rotleft(h0, 5)
                if temp > 10000:
                    # Extremely high magnitude
                    h1 = rotleft(h1, 30)
                    message_blocks.extend([0xFF, 0xEE])
                else:
                    h1 = rotright(h1, 7)
                    message_blocks.extend([0xDD, 0xCC])
            else:
                # Odd XOR result
                h2 = rotleft(h2, 13)
                if (temp % 3) == 0:
                    # Divisible by 3 - rare mathematical property
                    h3 = rotright(h3, 17)
                    message_blocks.extend([0xBB, 0xAA])
                else:
                    h3 = rotleft(h3, 21)
        elif temp < -1000:
            # Negative XOR - different processing path
            temp = abs(temp)
            if is_prime_like(temp):
                # Prime-like property check - computationally intensive path
                h0, h1 = h1, h0  # Swap state
                message_blocks.append(0x99)
            else:
                h2, h3 = h3, h2
                message_blocks.append(0x88)
        else:
            # Normal range XOR
            message_blocks.extend([0x11, 0x22])
    
    # Multi-round processing (simplified)
    rounds = 4
    for round_num in range(rounds):
        for i, block in enumerate(message_blocks):
            # Complex mixing function with multiple branches
            if round_num == 0:
                # First round - choice function
                f_result = choice_function(h1, h2, h3)
                k = 0x5A827999
            elif round_num == 1:
                # Second round - parity function
                f_result = parity_function(h1, h2, h3)
                k = 0x6ED9EBA1
                if block > 200:  # Rare high block values
                    f_result ^= 0xDEADBEEF
            elif round_num == 2:
                # Third round - majority function
                f_result = majority_function(h1, h2, h3)
                k = 0x8F1BBCDC
                if is_power_of_two(block + 1):
                    # Power of 2 check - mathematical property
                    f_result = rotleft(f_result, 11)
            else:
                # Final round - mixed function
                f_result = mixed_function(h1, h2, h3, block)
                k = 0xCA62C1D6
                if (h0 & h1 & h2 & h3) != 0:
                    # All states have common bits - rare condition
                    f_result ^= collision_resistance_mixer(h0, h1, h2, h3)
            
            # Main hash update with carry propagation
            temp = add_with_carry(h0, f_result, block, k, i)
            
            # State rotation - different patterns per round
            if round_num % 2 == 0:
                h0, h1, h2, h3 = temp, h0, rotleft(h1, 30), h2
            else:
                h0, h1, h2, h3 = temp, h0, rotright(h1, 30), h2
    
    # Final output computation
    final_hash = h0 ^ h1 ^ h2 ^ h3
    
    # Output post-processing with validation
    if final_hash == 0:
        # Collision prevention - should never be zero
        final_hash = 0xDEADBEEF
    elif final_hash == 0xFFFFFFFF:
        # All ones prevention
        final_hash = 0x12345678
    
    # Return bounded result for testing
    return abs(final_hash) % 1000000

def rotleft(value, amount):
    """Rotate left with 32-bit wrapping"""
    return ((value << amount) | (value >> (32 - amount))) & 0xFFFFFFFF

def rotright(value, amount):
    """Rotate right with 32-bit wrapping"""
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF

def choice_function(x, y, z):
    """Choose y or z based on x bits"""
    return (x & y) | (~x & z)

def parity_function(x, y, z):
    """XOR parity function"""
    return x ^ y ^ z

def majority_function(x, y, z):
    """Majority vote function"""
    return (x & y) | (x & z) | (y & z)

def mixed_function(x, y, z, block):
    """Complex mixing function with block dependency"""
    if block % 4 == 0:
        return choice_function(x, y, z)
    elif block % 4 == 1:
        return parity_function(x, y, z)
    elif block % 4 == 2:
        return majority_function(x, y, z)
    else:
        return x ^ (y | ~z)

def is_prime_like(n):
    """Simplified primality-like test for specific path triggering"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    # Test only small factors for complexity
    for i in [3, 5, 7, 11, 13]:
        if n % i == 0:
            return n == i
    return True  # Consider "prime-like" for our purposes

def is_power_of_two(n):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0

def add_with_carry(a, b, c, d, index):
    """Addition with carry propagation and index dependency"""
    result = (a + b + c + d) & 0xFFFFFFFF
    
    # Carry-dependent processing
    if result > 0x80000000:
        # High bit set - carry occurred
        if index % 2 == 0:
            result ^= 0x1B873593
        else:
            result ^= 0xCC9E2D51
    
    return result

def collision_resistance_mixer(h0, h1, h2, h3):
    """Anti-collision mixing function - very rarely called"""
    mix = h0 ^ rotleft(h1, 13) ^ rotright(h2, 19) ^ rotleft(h3, 7)
    
    # Multiple avalanche rounds
    mix ^= mix >> 16
    mix *= 0x85EBCA6B
    mix ^= mix >> 13
    mix *= 0xC2B2AE35
    mix ^= mix >> 16
    
    return mix & 0xFFFFFFFF

if __name__ == '__main__':
    # Test cases that should trigger different paths
    print("Testing cryptographic hash function:")
    
    # Small positive values (should trigger specific padding path)
    result1 = cryptographic_hash(50, 100, 150, 200)
    print(f"Small ascending: {result1}")
    
    # Large values with high XOR
    result2 = cryptographic_hash(5000, 3000, 8000, 2000)
    print(f"Large XOR path: {result2}")
    
    # Mixed signs
    result3 = cryptographic_hash(-100, 50, -200, 75)
    print(f"Mixed signs: {result3}")
    
    # Edge case - all zeros
    result4 = cryptographic_hash(0, 0, 0, 0)
    print(f"All zeros: {result4}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original cryptographic_hash to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: cryptographic_hash(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from cryptographic_hash
    """
    # Set defaults for optional parameters based on function requirements
    if b is None:
        b = -100
    if c is None:
        c = 0
    if d is None:
        d = 100
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-100 <= param <= 100):
                param = max(-100, min(100, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return cryptographic_hash(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return cryptographic_hash(a, b, c)
            except:
                pass
            try:
                return cryptographic_hash(a, b)
            except:
                pass
            try:
                return cryptographic_hash(a)
            except:
                pass
            return cryptographic_hash(a)
        raise e
