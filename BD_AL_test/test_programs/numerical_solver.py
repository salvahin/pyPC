def numerical_solver(a, b, c, d):
    """
    Numerical equation solver using multiple iterative methods.
    Solves equations like f(x) = ax³ + bx² + cx + d = 0 using Newton-Raphson,
    Bisection, and Secant methods with convergence analysis.
    Target: 25-35% coverage, cyclomatic complexity ~55
    """
    import math
    
    # Initialize solver state
    solver_state = {
        'method': 'auto',
        'iterations': 0,
        'convergence': False,
        'precision': 1e-6,
        'max_iterations': 100,
        'roots_found': [],
        'method_switches': 0
    }
    
    result_value = 0
    coefficients = [float(a), float(b), float(c), float(d)]
    
    # Input validation and preprocessing
    if all(coef == 0 for coef in coefficients[:3]):
        # Degenerate case: not a proper equation
        if d == 0:
            # 0 = 0, infinite solutions
            return 999999
        else:
            # Constant = 0, no solutions
            return -999999
    
    # Determine initial method based on coefficients
    if coefficients[0] == 0:
        if coefficients[1] == 0:
            # Linear equation: cx + d = 0
            if coefficients[2] != 0:
                root = -coefficients[3] / coefficients[2]
                return int(abs(root * 1000)) % 100000
            else:
                return 0
        else:
            # Quadratic equation: bx² + cx + d = 0
            return solve_quadratic(coefficients[1], coefficients[2], coefficients[3])
    
    # Cubic equation - use multiple methods
    initial_guesses = generate_initial_guesses(coefficients)
    
    for guess_idx, initial_x in enumerate(initial_guesses):
        solver_state['iterations'] = 0
        solver_state['convergence'] = False
        
        # Try different methods based on problem characteristics
        method_sequence = determine_method_sequence(coefficients, initial_x, guess_idx)
        
        for method_name in method_sequence:
            solver_state['method'] = method_name
            
            if method_name == 'newton_raphson':
                root_result = newton_raphson_method(coefficients, initial_x, solver_state)
            elif method_name == 'bisection':
                # Need to find interval first
                interval = find_sign_change_interval(coefficients, initial_x)
                if interval:
                    root_result = bisection_method(coefficients, interval, solver_state)
                else:
                    root_result = {'converged': False, 'root': initial_x}
            elif method_name == 'secant':
                second_guess = initial_x + 1.0 if initial_x != -1.0 else initial_x - 1.0
                root_result = secant_method(coefficients, initial_x, second_guess, solver_state)
            elif method_name == 'hybrid':
                root_result = hybrid_method(coefficients, initial_x, solver_state)
            else:
                # Fallback method
                root_result = simple_iteration_method(coefficients, initial_x, solver_state)
            
            if root_result['converged']:
                # Verify the solution
                verification_result = verify_solution(coefficients, root_result['root'])
                
                if verification_result['valid']:
                    solver_state['roots_found'].append(root_result['root'])
                    result_value += int(abs(root_result['root']) * 1000)
                    
                    # Check for multiple roots
                    deflated_coeffs = deflate_polynomial(coefficients, root_result['root'])
                    if should_continue_search(deflated_coeffs, solver_state):
                        # Continue with deflated polynomial
                        additional_roots = solve_deflated_system(deflated_coeffs, solver_state)
                        for additional_root in additional_roots:
                            solver_state['roots_found'].append(additional_root)
                            result_value += int(abs(additional_root) * 500)
                    
                    break  # Success with this method
                else:
                    # False convergence - try next method
                    solver_state['method_switches'] += 1
                    if solver_state['method_switches'] > 3:
                        break  # Too many switches, give up
            else:
                # Method failed - switch to next
                solver_state['method_switches'] += 1
                if solver_state['method_switches'] > 2:
                    # Try more robust method before giving up
                    if can_use_robust_fallback(coefficients, solver_state):
                        fallback_result = robust_fallback_method(coefficients, initial_x, solver_state)
                        if fallback_result['converged']:
                            solver_state['roots_found'].append(fallback_result['root'])
                            result_value += int(abs(fallback_result['root']) * 750)
                    break
        
        # Early termination if we found enough roots
        if len(solver_state['roots_found']) >= 2:
            break
    
    # Final result processing
    result_value += calculate_convergence_bonus(solver_state)
    result_value += calculate_accuracy_penalty(coefficients, solver_state['roots_found'])
    
    return abs(result_value) % 100000

def solve_quadratic(b, c, d):
    """Solve quadratic equation bx² + cx + d = 0"""
    discriminant = c * c - 4 * b * d
    
    if discriminant < 0:
        # Complex roots
        return 5000
    elif discriminant == 0:
        # One repeated root
        root = -c / (2 * b)
        return int(abs(root * 1000)) % 10000
    else:
        # Two real roots
        import math
        sqrt_disc = math.sqrt(discriminant)
        root1 = (-c + sqrt_disc) / (2 * b)
        root2 = (-c - sqrt_disc) / (2 * b)
        return int(abs(root1 * 1000 + root2 * 1000)) % 10000

def generate_initial_guesses(coefficients):
    """Generate multiple initial guesses based on coefficient analysis"""
    guesses = []
    
    # Standard guesses
    guesses.extend([0.0, 1.0, -1.0])
    
    # Coefficient-based guesses
    a, b, c, d = coefficients
    
    if a != 0:
        # For cubic, rough estimate: x ≈ -d/c if c != 0
        if c != 0:
            guesses.append(-d / c)
        
        # Another estimate: x ≈ -b/(3a) (critical point of cubic)
        guesses.append(-b / (3 * a))
    
    # Range-based guesses
    max_coeff = max(abs(coef) for coef in coefficients if coef != 0)
    if max_coeff > 0:
        estimate_range = abs(d) / max_coeff + 1
        guesses.extend([-estimate_range, estimate_range])
    
    return guesses[:5]  # Limit to 5 guesses

def determine_method_sequence(coefficients, initial_x, guess_index):
    """Determine the sequence of methods to try"""
    a, b, c, d = coefficients
    
    # Analyze derivative at initial point for Newton's method suitability
    derivative_at_x = 3 * a * initial_x * initial_x + 2 * b * initial_x + c
    
    if abs(derivative_at_x) > 0.1:
        # Good for Newton's method
        if guess_index == 0:
            return ['newton_raphson', 'secant', 'bisection']
        else:
            return ['newton_raphson', 'hybrid', 'secant']
    else:
        # Derivative too small, avoid Newton's method initially
        if abs(initial_x) < 10:
            return ['secant', 'bisection', 'newton_raphson']
        else:
            return ['bisection', 'secant', 'hybrid']

def newton_raphson_method(coefficients, initial_x, solver_state):
    """Newton-Raphson method implementation"""
    a, b, c, d = coefficients
    x = initial_x
    
    for iteration in range(solver_state['max_iterations']):
        # Calculate function value
        f_x = a * x**3 + b * x**2 + c * x + d
        
        # Calculate derivative
        f_prime_x = 3 * a * x**2 + 2 * b * x + c
        
        # Check for zero derivative
        if abs(f_prime_x) < solver_state['precision'] * 10:
            # Derivative too small, method fails
            return {'converged': False, 'root': x}
        
        # Newton's update
        x_new = x - f_x / f_prime_x
        
        # Check convergence
        if abs(x_new - x) < solver_state['precision']:
            solver_state['iterations'] += iteration + 1
            return {'converged': True, 'root': x_new}
        
        # Check for divergence
        if abs(x_new) > 1000:
            return {'converged': False, 'root': x_new}
        
        x = x_new
    
    solver_state['iterations'] += solver_state['max_iterations']
    return {'converged': False, 'root': x}

def find_sign_change_interval(coefficients, center, search_range=10):
    """Find interval where function changes sign"""
    a, b, c, d = coefficients
    
    def eval_function(x):
        return a * x**3 + b * x**2 + c * x + d
    
    # Search around center point
    left = center - search_range
    right = center + search_range
    step = search_range / 10
    
    prev_x = left
    prev_f = eval_function(left)
    
    x = left + step
    while x <= right:
        f_x = eval_function(x)
        
        if prev_f * f_x < 0:  # Sign change detected
            return (prev_x, x)
        
        prev_x = x
        prev_f = f_x
        x += step
    
    return None  # No sign change found

def bisection_method(coefficients, interval, solver_state):
    """Bisection method implementation"""
    a, b, c, d = coefficients
    
    def eval_function(x):
        return a * x**3 + b * x**2 + c * x + d
    
    left, right = interval
    
    if eval_function(left) * eval_function(right) >= 0:
        return {'converged': False, 'root': (left + right) / 2}
    
    for iteration in range(solver_state['max_iterations']):
        mid = (left + right) / 2
        f_mid = eval_function(mid)
        
        if abs(f_mid) < solver_state['precision'] or abs(right - left) < solver_state['precision']:
            solver_state['iterations'] += iteration + 1
            return {'converged': True, 'root': mid}
        
        f_left = eval_function(left)
        if f_left * f_mid < 0:
            right = mid
        else:
            left = mid
    
    solver_state['iterations'] += solver_state['max_iterations']
    return {'converged': False, 'root': (left + right) / 2}

def secant_method(coefficients, x0, x1, solver_state):
    """Secant method implementation"""
    a, b, c, d = coefficients
    
    def eval_function(x):
        return a * x**3 + b * x**2 + c * x + d
    
    for iteration in range(solver_state['max_iterations']):
        f_x0 = eval_function(x0)
        f_x1 = eval_function(x1)
        
        # Check for zero denominator
        if abs(f_x1 - f_x0) < solver_state['precision'] * 10:
            return {'converged': False, 'root': x1}
        
        # Secant update
        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        
        # Check convergence
        if abs(x_new - x1) < solver_state['precision']:
            solver_state['iterations'] += iteration + 1
            return {'converged': True, 'root': x_new}
        
        # Check for divergence
        if abs(x_new) > 1000:
            return {'converged': False, 'root': x_new}
        
        # Update for next iteration
        x0, x1 = x1, x_new
    
    solver_state['iterations'] += solver_state['max_iterations']
    return {'converged': False, 'root': x1}

def hybrid_method(coefficients, initial_x, solver_state):
    """Hybrid method combining Newton and bisection"""
    # Start with Newton's method
    newton_result = newton_raphson_method(coefficients, initial_x, solver_state)
    
    if newton_result['converged']:
        return newton_result
    
    # If Newton fails, try to find interval and use bisection
    interval = find_sign_change_interval(coefficients, initial_x)
    if interval:
        return bisection_method(coefficients, interval, solver_state)
    else:
        # Last resort: try secant from different starting points
        x1 = initial_x + 0.1 if initial_x != -0.1 else initial_x - 0.1
        return secant_method(coefficients, initial_x, x1, solver_state)

def simple_iteration_method(coefficients, initial_x, solver_state):
    """Simple fixed-point iteration method"""
    a, b, c, d = coefficients
    x = initial_x
    
    # Rearrange to x = g(x) form: x = -ax³ - bx² - d)/c if c != 0
    if abs(coefficients[2]) < solver_state['precision']:
        return {'converged': False, 'root': x}
    
    for iteration in range(solver_state['max_iterations'] // 2):  # Slower method, fewer iterations
        x_new = -(a * x**3 + b * x**2 + d) / coefficients[2]
        
        if abs(x_new - x) < solver_state['precision']:
            solver_state['iterations'] += iteration + 1
            return {'converged': True, 'root': x_new}
        
        if abs(x_new) > 1000:  # Divergence check
            return {'converged': False, 'root': x_new}
        
        x = x_new
    
    solver_state['iterations'] += solver_state['max_iterations'] // 2
    return {'converged': False, 'root': x}

def verify_solution(coefficients, root):
    """Verify that the found root is actually a solution"""
    a, b, c, d = coefficients
    
    function_value = a * root**3 + b * root**2 + c * root + d
    tolerance = 1e-4
    
    is_valid = abs(function_value) < tolerance
    
    return {
        'valid': is_valid,
        'residual': abs(function_value),
        'tolerance_met': abs(function_value) < tolerance
    }

def deflate_polynomial(coefficients, root):
    """Remove known root from polynomial (synthetic division)"""
    # For cubic ax³ + bx² + cx + d, divide by (x - root)
    a, b, c, d = coefficients
    
    # Resulting quadratic: ax² + (b + ar)x + (c + br + ar²)
    new_a = a
    new_b = b + a * root
    new_c = c + b * root + a * root * root
    
    return [new_a, new_b, new_c]

def should_continue_search(deflated_coeffs, solver_state):
    """Determine if we should search for more roots"""
    # Continue if we have less than 2 roots and coefficients are significant
    if len(solver_state['roots_found']) >= 2:
        return False
    
    max_coeff = max(abs(coef) for coef in deflated_coeffs if coef != 0)
    return max_coeff > 1e-6 and solver_state['iterations'] < solver_state['max_iterations'] * 0.7

def solve_deflated_system(deflated_coeffs, solver_state):
    """Solve the deflated quadratic system"""
    if len(deflated_coeffs) >= 3:
        a, b, c = deflated_coeffs[0], deflated_coeffs[1], deflated_coeffs[2]
        
        if a == 0:
            # Linear equation
            if b != 0:
                return [-c / b]
            else:
                return []
        
        # Quadratic formula
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            import math
            sqrt_disc = math.sqrt(discriminant)
            root1 = (-b + sqrt_disc) / (2 * a)
            root2 = (-b - sqrt_disc) / (2 * a)
            
            # Verify both roots
            valid_roots = []
            for root in [root1, root2]:
                if abs(root) < 100:  # Reasonable range
                    valid_roots.append(root)
            
            return valid_roots
    
    return []

def can_use_robust_fallback(coefficients, solver_state):
    """Check if robust fallback method can be used"""
    # Use robust method if we haven't tried too many iterations
    return solver_state['iterations'] < solver_state['max_iterations'] * 0.8

def robust_fallback_method(coefficients, initial_x, solver_state):
    """Robust fallback method for difficult cases"""
    # Grid search combined with local optimization
    best_root = initial_x
    best_residual = float('inf')
    
    # Coarse grid search
    for test_x in [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]:
        a, b, c, d = coefficients
        residual = abs(a * test_x**3 + b * test_x**2 + c * test_x + d)
        
        if residual < best_residual:
            best_residual = residual
            best_root = test_x
    
    # Local refinement using Newton's method
    if best_residual < 10:  # Promising candidate
        refined_result = newton_raphson_method(coefficients, best_root, solver_state)
        if refined_result['converged']:
            return refined_result
    
    # Return best grid point
    converged = best_residual < solver_state['precision'] * 100
    return {'converged': converged, 'root': best_root}

def calculate_convergence_bonus(solver_state):
    """Calculate bonus based on convergence quality"""
    bonus = 0
    
    if len(solver_state['roots_found']) > 0:
        bonus += len(solver_state['roots_found']) * 1000
    
    # Efficiency bonus
    if solver_state['iterations'] < 20:
        bonus += 500
    elif solver_state['iterations'] < 50:
        bonus += 200
    
    # Method stability bonus
    if solver_state['method_switches'] <= 1:
        bonus += 300
    
    return bonus

def calculate_accuracy_penalty(coefficients, roots):
    """Calculate penalty for inaccurate solutions"""
    penalty = 0
    
    for root in roots:
        verification = verify_solution(coefficients, root)
        if not verification['valid']:
            penalty += int(verification['residual'] * 1000)
    
    return penalty

if __name__ == '__main__':
    print("Testing numerical solver:")
    
    # Test cubic with real roots
    result1 = numerical_solver(1, -6, 11, -6)  # Roots at 1, 2, 3
    print(f"Cubic with known roots: {result1}")
    
    # Test difficult cubic
    result2 = numerical_solver(2, -4, -22, 24)
    print(f"Difficult cubic: {result2}")
    
    # Test quadratic case
    result3 = numerical_solver(0, 1, -3, 2)  # x² - 3x + 2 = 0
    print(f"Quadratic case: {result3}")
    
    # Test linear case
    result4 = numerical_solver(0, 0, 2, -6)  # 2x - 6 = 0
    print(f"Linear case: {result4}")