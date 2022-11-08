def min(x,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,u,v):
    a = [x,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,u,v]
    # smallest number
    smallest = a[0]
    
    # find smallest
    for i in a:
    	if i<smallest:
    		smallest=i
    
    print(f"Smallest element is: {smallest}")
    return smallest

if __name__ == '__main__':
    min(1,2,3,4,45,6,76,8,9,0,10,11,234,55,7,7,83,2,5,6)