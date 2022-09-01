def three_number_sort(a,b,c):
    if a > b:
        a,b = b,a
    if a > c:
        a,c = c,a
    if b > c:
        b,c = c,b     
    print (a, "<", b, "<", c)


if __name__ == '__main__':
    a=b=c=1
    three_number_sort(a,b,c)