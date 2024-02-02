import numpy as np
from numpy.linalg import solve

def theta_method(f,theta,h,t_end,y0, dfdy):
    t = np.arange(start=h,stop=t_end+h/2, step=h)
    y = np.zeros(shape=(len(y0),len(t)))

    for i in range(len(t)):
        y[:,i] = solve_for_next_step(y,f,theta,h,t[i],t[i-1],dfdy)

    return y


def solve_for_next_step(y,f,theta,h,t,t_previous,dfdy):
    #theta = 1 => Forward Euler (explicit solution)
    if theta == 1:
        #y_n+1 = y_n + theta * h * f(y_n,t_n)
        return y[:,-1] + 1 * h * f(y[:,-1], t_previous)

    #otherwise, implicit solution => Newton's Method
    else:
        return newton_method(y,f,theta,h,t,t_previous,dfdy)

def newton_method(y,f,theta,h,t,t_previous,dfdy,tol=1e-3):
    #take initial guess as y_previous
    next_iter = y[:,-1]

    #Until convergence condition is met, iterate over newton method
    while abs(current_iter - next_iter) > tol:
        current_iter = next_iter
        
        jac = np.identity(len(y[:,0])) - (1-theta) * h * dfdy(current_iter)

        g_previous = (current_iter - (y[:,-1] + theta * h * f(y[:,-1], t_previous) + 
                                          (1-theta) * h * f(current_iter, t)))
        
        next_iter = solve(a=jac, b=-g_previous)

