import numpy as np
from numpy.linalg import solve

def theta_method(f,theta,h,t_end,y0, dfdy):
    t = np.arange(start=0,stop=t_end+h/2, step=h)
    y = np.zeros(shape=(len(y0),len(t)))
    y[:,0] = y0

    for i in range(1,len(t)):
        y[:,i] = solve_for_next_step(y,f,theta,h,t[i],t[i-1],dfdy, i)

    return y


def solve_for_next_step(y,f,theta,h,t,t_previous,dfdy,i):
    #theta = 1 => Forward Euler (explicit solution)
    if theta == 1:
        #y_n+1 = y_n + theta * h * f(y_n,t_n)
        return y[:,i-1] + 1 * h * f(y[:,i-1], t_previous)

    #otherwise, implicit solution => Newton's Method
    else:
        return newton_method(y,f,theta,h,t,t_previous,dfdy, i)


def newton_method(y,f,theta,h,t,t_previous,dfdy,i,tol=1e-3):
    #take initial guess as y_previous
    next_iter = y[:,i-1]
    current_iter = next_iter + tol * 10
    count = 0

    #Until convergence condition is met, iterate over newton method
    while np.linalg.norm(current_iter - next_iter, ord=2) > tol:
        current_iter = next_iter.copy()

        jac = np.identity(len(y[:,0])) - (1-theta) * h * np.array([d(current_iter,t) for d in dfdy])

        # g_previous = (current_iter - (y[:,-1] + theta * h * np.array([ff(y[:,-1], t_previous) for ff in f]) + 
        #                                   (1-theta) * h * np.array([ff(current_iter, t) for ff in f])))
        g_previous = (current_iter - (y[:,i-1] + 
                              (1-theta) * h * (-1 * current_iter)))
        
        next_iter = solve(a=jac, b=-g_previous) + current_iter
        
        count += 1
        if count > 1e5: print("Too many iterations for Newton's Method")

    return next_iter