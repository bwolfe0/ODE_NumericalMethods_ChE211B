import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos

# Define the derivative functions
def f1(y1, t):
    return -y1

def f2(y2, t):
    return -100*(y2 - sin(t)) + cos(t)

# Euler method solver for a system of two ODEs
def euler_system(f1, f2, y1_0, y2_0, h, t_end):
    y1 = [y1_0]
    y2 = [y2_0]
    t = np.arange(0, t_end + h, h)
    for i in range(1, len(t)):
        y1.append(y1[-1] + h * f1(y1[-1], t[i-1]))
        y2.append(y2[-1] + h * f2(y2[-1], t[i-1]))
    return t, y1, y2

def forward_euler(f,y_0,h,t_end):
    y = [y_0]
    t = np.arange(0,t_end + h/2,h)
    for i in range(1, len(t)):
        y.append(
            y[-1] + h*f(y[-1],t[i-1])
        )
    
    return(t,y)