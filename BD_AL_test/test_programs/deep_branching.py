def deep_branching(a, b, c, d):
    score = 0
    path = 0
    
    # First level branching
    if a > 0:
        if a > 5:
            if a > 10:
                score = a * 3
                path = 1
            else:
                score = a * 2
                path = 2
        else:
            score = a
            path = 3
            
        # Second level based on b
        if b > 0:
            if path == 1:
                if b > a:
                    score += b * 2
                else:
                    score += b
            elif path == 2:
                if b > 3:
                    score += b * 3
                else:
                    score += 1
            else:
                score += b // 2
    else:
        if a == 0:
            score = 10
            path = 4
        else:
            score = abs(a)
            path = 5
            
        # Different branch for negative a
        if b < 0:
            if path == 4:
                score += abs(b) * 2
            else:
                score += abs(b)
    
    # Third level based on c
    if c > 0:
        if score > 20:
            if c > 5:
                score *= 2
            else:
                score += c * 5
        elif score > 10:
            if c > 3:
                score += c * 3
            else:
                score += c
        else:
            if c > 7:
                score = c * 10
            else:
                score += c * 2
    elif c < 0:
        if score > 15:
            score -= abs(c)
        else:
            score += abs(c) // 2
    
    # Fourth level based on d
    if d != 0:
        if d > 0:
            if score > 30:
                if d > 5:
                    score += d * 4
                else:
                    score += d * 2
            elif score > 15:
                if d > 3:
                    score += d * 3
                else:
                    score += d
            else:
                score += d // 2
        else:
            if score > 25:
                score -= abs(d) * 2
            elif score > 10:
                score -= abs(d)
            else:
                score = abs(d)
    
    # Final adjustments
    if score > 100:
        if path < 3:
            score = 100
        else:
            score = 90
    elif score < 0:
        score = 0
    
    return score

if __name__ == '__main__':
    result = deep_branching(7, 3, 5, -2)
    print(f"Score: {result}")