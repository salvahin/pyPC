def factorial(n):
    if n > 1:
        return n * factorial(n-1)
    return 1

if __name__ == '__main__':
    n = 9
    print(factorial(n))