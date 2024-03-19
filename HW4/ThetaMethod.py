import numpy as np
from numpy.linalg import solve
from numpy import sin, cos, exp


#### NOTE: This was hard coded for a 2-equation system. Modify lines 24, 43, 44 and 63 if needed#### 


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
        return y[:,i-1] + 1 * h * np.array([ff(y[0,i-1],y[1,i-1], t_previous) for p, ff in enumerate(f)])

    #otherwise, implicit solution => Newton's Method
    else:
        return newton_method(y,f,theta,h,t,t_previous,dfdy, i)


def newton_method(y,f,theta,h,t,t_previous,dfdy,i,tol=1e-6):
    #take initial guess as y_previous
    next_iter = y[:,i-1]
    current_iter = next_iter + tol * 10
    count = 0

    #Until convergence condition is met, iterate over newton method
    while np.linalg.norm(current_iter - next_iter, ord=2) > tol:
        # print(f"t: {t}")
        # print(f"ci: {current_iter}")
        # print(f"ni: {next_iter})")
        current_iter = next_iter.copy()

        jac = get_jac(len(y[:,0]), dfdy, theta, current_iter, h, t)

        f_eval_previous = np.array([ff(y[0,i-1],y[1,i-1], t_previous) for p, ff in enumerate(f)]).flatten()
        f_eval_current = np.array([ff(current_iter[0],current_iter[1], t) for q, ff in enumerate(f)]).flatten()

        g_previous = current_iter - (y[:,i-1] + 
                    theta * h *  f_eval_previous +
                    (1-theta) * h * f_eval_current)

        next_iter = solve(a=jac, b=-g_previous) + current_iter
        
        count += 1
        if count > 1e5: print("Too many iterations for Newton's Method")

    return next_iter



def get_jac(size, dfdy, theta, current_iter, h,t):
    jac = np.identity(size)
    for i, dfdy_row in enumerate(dfdy):
        for j, df in enumerate(dfdy_row):
            jac[i,j] -= (1-theta) * h * df(current_iter[0],current_iter[1],t)
    return jac