from cmath import sqrt


def trig_area(a, b, c):
    match = p = s = h = 0
    if ((a+b)>=c and b+c >= a) and ((a+c >= b and a > 0) and (b > 0 and c >0)):
        if a == b:
            match += 1
        if a == c:
            match += 2
        if b == c:
            match += 3
        if match == 0:
            p = (a+b+c)/2
            s = sqrt(p * (p-a) * (p-b) * (p-c))
            print("Anomalistic")
        elif match == 1:
            h = sqrt((a**2) - ((c/2)**2))
            s = c*h/2
            print("Isoceles and First=Second")
        elif match == 2:
            h = sqrt((a**2) - ((b/2)**2))
            s = b*h/2
            print("Isoceles and First=Third")
        elif match == 3:
            h = sqrt((b**2) - ((a/2)**2))
            s = a*h/2
            print("Isoceles and Second=Third")
        else:
            s = sqrt(3) * a * a/4
            print("Equilateral")
    else:
        print("Not a Triangle")
    return s