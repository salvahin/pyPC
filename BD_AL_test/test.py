def excecute(x: int, y, a=2, *defaultlistx, **args) -> int:
    #function execute
    for i in range(10):
        for asd in range(10,20):
            print(i)
    for ar in range(2):
        for are in range(2):
            if x:
                print(3)
    if x > 2 and y > 100:
        x = x ** 2
        for i in range(5):
            for df in range(5):
                if i == 0:
                    print(i)
    elif x == 2:
        test = y
    elif y == 3:
        test = 0
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