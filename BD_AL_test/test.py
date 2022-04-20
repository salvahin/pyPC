def excecute(x: int, y, a=2, *defaultlistx, **args) -> int:
    #function execute
    if x == 3 and y > 100:
        x = x ** 2
    elif x == 2:
        test = y
    test = x
    if y > 100:
        test = 1
        if x > 2 and y < 200:
            test = 0
    else:
        print("success")
        print("s")
        a = 6
    return 0

if __name__ == '__main__':
    """
    call to function
    """
    x = 1
    y = 3
    excecute(x, y)