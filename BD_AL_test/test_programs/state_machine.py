class StateMachine:
    def __init__(self):
        self.state = 'INIT'
        self.counter = 0
        self.history = []
        self.error_count = 0
    
    def transition(self, event):
        self.history.append((self.state, event))
        
        if self.state == 'INIT':
            if event > 0:
                self.state = 'ACTIVE'
                self.counter = 1
            elif event < 0:
                self.state = 'ERROR'
                self.error_count += 1
            else:
                self.state = 'IDLE'
        
        elif self.state == 'IDLE':
            if event > 5:
                self.state = 'ACTIVE'
                self.counter += 1
            elif event < -5:
                self.state = 'ERROR'
                self.error_count += 1
            elif event != 0:
                self.counter += 1
                if self.counter > 3:
                    self.state = 'PROCESSING'
        
        elif self.state == 'ACTIVE':
            if event > 10:
                self.state = 'PROCESSING'
                self.counter *= 2
            elif event < -10:
                self.state = 'ERROR'
                self.error_count += 1
            elif event == 0:
                self.state = 'IDLE'
            else:
                self.counter += event
                if self.counter > 20:
                    self.state = 'COMPLETE'
                elif self.counter < 0:
                    self.state = 'ERROR'
                    self.error_count += 1
        
        elif self.state == 'PROCESSING':
            if event > 0:
                self.counter += event
                if self.counter > 50:
                    self.state = 'COMPLETE'
                elif self.counter > 30:
                    if event % 2 == 0:
                        self.state = 'VALIDATING'
            elif event < 0:
                self.counter += event
                if self.counter < 0:
                    self.state = 'ERROR'
                    self.error_count += 1
                elif self.counter < 10:
                    self.state = 'ACTIVE'
            else:
                if len(self.history) > 5:
                    self.state = 'VALIDATING'
        
        elif self.state == 'VALIDATING':
            if event > 0:
                if self.error_count == 0:
                    self.state = 'COMPLETE'
                else:
                    self.state = 'ACTIVE'
                    self.error_count = max(0, self.error_count - 1)
            elif event < 0:
                self.state = 'ERROR'
                self.error_count += 1
            else:
                if self.counter > 25:
                    self.state = 'COMPLETE'
                else:
                    self.state = 'PROCESSING'
        
        elif self.state == 'ERROR':
            if event > 10:
                self.state = 'RECOVERING'
                self.error_count = max(0, self.error_count - 1)
            elif event > 0:
                if self.error_count < 3:
                    self.state = 'IDLE'
            elif event == 0:
                if self.error_count > 5:
                    self.state = 'FAILED'
                else:
                    self.state = 'RECOVERING'
        
        elif self.state == 'RECOVERING':
            if event > 5:
                self.state = 'IDLE'
                self.counter = 0
            elif event > 0:
                self.error_count = max(0, self.error_count - 1)
                if self.error_count == 0:
                    self.state = 'IDLE'
            elif event < 0:
                self.error_count += 1
                if self.error_count > 10:
                    self.state = 'FAILED'
                else:
                    self.state = 'ERROR'
        
        elif self.state == 'COMPLETE':
            if event < 0:
                self.state = 'VALIDATING'
                self.counter = max(0, self.counter + event)
            elif event == 0:
                return self.counter
        
        elif self.state == 'FAILED':
            return -1
        
        return self.counter

def test_state_machine(a, b, c, d):
    sm = StateMachine()
    
    events = [a, b, c, d]
    final_result = 0
    
    for event in events:
        result = sm.transition(event)
        if result == -1:
            return -1
        final_result = result
    
    if sm.state == 'COMPLETE':
        return final_result * 10
    elif sm.state == 'FAILED':
        return -1
    elif sm.state == 'ERROR':
        return -sm.error_count
    elif sm.state == 'PROCESSING' or sm.state == 'VALIDATING':
        return final_result
    elif sm.state == 'ACTIVE':
        return final_result // 2
    else:
        return 0

if __name__ == '__main__':
    result = test_state_machine(5, 12, -3, 8)
    print(f"State machine result: {result}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original test_state_machine to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: test_state_machine(a, b, c, d)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from test_state_machine
    """
    # Set defaults for optional parameters based on function requirements
    if b is None:
        b = -20
    if c is None:
        c = 0
    if d is None:
        d = 20
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-20 <= param <= 20):
                param = max(-20, min(20, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return test_state_machine(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return test_state_machine(a, b, c)
            except:
                pass
            try:
                return test_state_machine(a, b)
            except:
                pass
            try:
                return test_state_machine(a)
            except:
                pass
            return test_state_machine(a)
        raise e
