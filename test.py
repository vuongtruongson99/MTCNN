import numpy as np

x = 9
y = 4
z = 2

if x == y and z > 0:
    z = x
    x = y
    y = z
else:
    z = x*2
    x = 2*y
    y =z

print(x, y, z)