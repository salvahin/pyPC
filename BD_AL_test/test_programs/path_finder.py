def find_path(maze_size, start_x, start_y, obstacles):
    if maze_size <= 0 or maze_size > 10:
        return None
    
    if start_x < 0 or start_x >= maze_size:
        return None
    
    if start_y < 0 or start_y >= maze_size:
        return None
    
    maze = [[0 for _ in range(maze_size)] for _ in range(maze_size)]
    
    for obs in obstacles:
        if obs < maze_size * maze_size:
            x = obs // maze_size
            y = obs % maze_size
            if x < maze_size and y < maze_size:
                maze[x][y] = 1
    
    if maze[start_x][start_y] == 1:
        return []
    
    visited = set()
    path = []
    
    def dfs(x, y, depth):
        if depth > 20:
            return False
        
        if x < 0 or x >= maze_size or y < 0 or y >= maze_size:
            return False
        
        if (x, y) in visited:
            return False
        
        if maze[x][y] == 1:
            return False
        
        visited.add((x, y))
        path.append((x, y))
        
        if x == maze_size - 1 and y == maze_size - 1:
            return True
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if nx >= 0 and nx < maze_size and ny >= 0 and ny < maze_size:
                if (nx, ny) not in visited and maze[nx][ny] == 0:
                    if depth < 5:
                        if dfs(nx, ny, depth + 1):
                            return True
                    elif depth < 10:
                        if x > y:
                            if dfs(nx, ny, depth + 1):
                                return True
                        else:
                            if ny > nx:
                                if dfs(nx, ny, depth + 1):
                                    return True
                    else:
                        if (x + y) % 2 == 0:
                            if dfs(nx, ny, depth + 1):
                                return True
        
        path.pop()
        return False
    
    if dfs(start_x, start_y, 0):
        if len(path) > 5:
            if len(path) > 10:
                return path[:10]
            return path
        else:
            return path * 2
    else:
        return []

def test_pathfinder(a, b, c, d):
    maze_size = abs(int(a)) % 8 + 2
    start_x = abs(int(b)) % maze_size
    start_y = abs(int(c)) % maze_size
    
    obstacles = []
    if d > 0:
        obstacles.append(abs(int(d)) % (maze_size * maze_size))
    
    result = find_path(maze_size, start_x, start_y, obstacles)
    
    if result is None:
        return 0
    elif len(result) == 0:
        return 1
    elif len(result) < 5:
        return 2
    else:
        return len(result)

if __name__ == '__main__':
    result = test_pathfinder(5, 0, 0, 3)
    print(f"Path finder result: {result}")


def target_function(a, b=None, c=None, d=None):
    """
    Standardized test function interface for experimental methodology.
    
    This function wraps the original find_path to provide
    a consistent 4-parameter interface for automated test generation.
    
    Original function: find_path(maze_size, start_x, start_y, obstacles)
    
    Args:
        a: First parameter (required)
        b: Second parameter (optional)
        c: Third parameter (optional) 
        d: Fourth parameter (optional)
    
    Returns:
        Result from find_path
    """
    # Set defaults for optional parameters based on function requirements
    if b is None:
        b = -50
    if c is None:
        c = 0
    if d is None:
        d = 50
    
    # Validate parameter ranges
    for param, name in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        if param is not None and isinstance(param, (int, float)):
            if not (-50 <= param <= 50):
                param = max(-50, min(50, param))  # Clamp to bounds
    
    # Call original function with appropriate parameters
    try:
        return find_path(a, b, c, d)
    except Exception as e:
        # Handle potential errors gracefully for test generation
        if "too many" in str(e).lower() or "unexpected keyword" in str(e).lower():
            # Try with fewer parameters
            try:
                return find_path(a, b, c)
            except:
                pass
            try:
                return find_path(a, b)
            except:
                pass
            try:
                return find_path(a)
            except:
                pass
            return find_path(a)
        raise e
