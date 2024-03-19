import numpy as np
from numpy.linalg import solve
from numpy import sin, cos, exp

# Main function that "takes" the steps
def fixed_h_bdf(f,h,t_end,y0, dfdy, tol=1e-3):
    t = np.arange(start=0,stop=t_end+h/2, step=h)
    y = np.zeros(shape=(len(y0),len(t)))
    y[:,0:len(y0)] = y0

    # Take step i
    for i in range(len(y0),len(t)):
        y[:,i] = calculate_next_step(y,f,h,t[i],dfdy,i,tol)

    return y


# Determine solution at next step
def calculate_next_step(y,f,h,t,dfdy,i,tol):

    # The first guess is our last solution
    next_iter = y[:,i-1]
    current_iter = next_iter + tol * 10
    count = 0

    # Until Newton Method Converges relative to our error tolerance
    while np.linalg.norm(current_iter - next_iter, ord=2) > tol:
        current_iter = next_iter.copy()

        # Calculate the jacobian at this step
        jac = get_jac(len(y[:,0]), dfdy, current_iter, h, t)

        # Evauate f1(y1,y2,t) and f2(y1,y2,t) at this step for use in g
        f_eval_current = np.array([ff(current_iter[0],current_iter[1], t) for q, ff in enumerate(f)]).flatten()

        # Evaluate the convergence function, should go to zero
        g_previous = current_iter + (-4/3*y[:,i-1] + 
                    1/3*y[:,i-2] - 2/3*h*f_eval_current)

        # Find for the next guess for y_{n+1}
        next_iter = solve(a=jac, b=-g_previous) + current_iter
        
        count += 1
        if count > 1e5: print("Too many iterations for Newton's Method")

    return next_iter



def get_jac(size, dfdy, current_iter, h,t):
    jac = np.identity(size)
    for i, dfdy_row in enumerate(dfdy):
        for j, df in enumerate(dfdy_row):
            jac[i,j] -= (2/3)*h * df(current_iter[0],current_iter[1],t)
    return jac