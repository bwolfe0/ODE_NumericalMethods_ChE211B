import numpy as np
from numpy.linalg import solve
from numpy import sin, cos, exp


def variable_h_bdf(f,h,t_start,t_end,y0, dfdy,h_matrix, tol=1e-6):
    t = t_start

    # Solution values will be stored as a list embedded in the list y:
    # y = [[y1], [y2]]. This list is appended at each step with the new 
    # solution values
    y = y0
    i = len(y0[0])
    run = True

    # Until t_end is reached
    while run is True:

        # Determine solution at next step
        result = calculate_next_step(y,f,h,t[-1],dfdy,i,tol, h_matrix)

        # Extract values from result
        y = result['y_list']
        last_h = result['last_h']
        h_matrix.append(last_h)
        h = result['next_h']

        # Keep track of y size
        i += 1

        # Once t_end is passed, interpolate back to desired endpoint
        if t [-1] > t_end:
            y[0][-1] = y[0][-2] + (t_end - t[-2]) * (y[0][-1] - y[0][-2]) / (t[-1] - t[-2])
            y[1][-1] = y[1][-2] + (t_end - t[-2]) * (y[1][-1] - y[1][-2]) / (t[-1] - t[-2])
            return {'y': y, 't':t, 'h_list': h_matrix}
        
        # Store the current time
        t.append(
            t[-1] + last_h
        )


def calculate_next_step(y,f,h,t,dfdy,i,tol, h_list):

    # The list "y" is converted to an array for convience, and returned back 
    # to a list at the end of this function
    y_array = np.array(y)

    # Our first guess will be the last value
    next_iter = y_array[:,i-1]
    current_iter = next_iter + tol * 10

    count = 0
    size = len(y)

    # Until the Newton Method Converges sufficiently relative to our error tolerance
    while np.max(abs(current_iter - next_iter)) > tol:
        current_iter = next_iter.copy()

        # Calculate the jacobian at this step
        jac = get_jac(size, dfdy, current_iter, h, t)

        # Evauate f1(y1,y2,t) and f2(y1,y2,t) at this step for use in g
        f_eval_current = np.array([ff(current_iter[0],current_iter[1], t) for q, ff in enumerate(f)]).flatten()

        # Evaluate the convergence function, should go to zero
        g_previous = (current_iter + (-4/3*y_array[:,i-1] + 
                    1/3*y_array[:,i-2] - 2/3*h*f_eval_current))

        # Find for the next guess for y_{n+1}
        next_iter = solve(a=jac, b=-g_previous) + current_iter
        
        count += 1
        if count > 1e5: print("Too many iterations for Newton's Method")

    # Determine the estimated error for the step just taken
    go = estimate_error(y_array,h,i,tol, next_iter,h_list)

    # If the error is acceptable, proceed to next step
    if go['signal'] is True:
        if size == 2:
            y[0].append(next_iter.tolist()[0])
            y[1].append(next_iter.tolist()[1])
        else:
            y.append(next_iter.tolist())
        last_h = h
        next_h = calc_next_step_size(h,tol,go['est'])
        return {'y_list': y, 'last_h':last_h, 'next_h': next_h}
    
    # If the error is unacceptable, determine new step size
    # and recursively call the function until the error is small enough
    else:
        h = calc_next_step_size(h,tol,go['est'])
        if h < 1e-20: 
            raise ValueError(f'h became too small: {h}')
        return calculate_next_step(y,f,h,t,dfdy,i,tol, h_list)



def get_jac(size, dfdy, current_iter, h,t):
    # Jac = I - 2/3 * h * df/dy
    jac = np.identity(size)
    for i, dfdy_row in enumerate(dfdy):
        for j, df in enumerate(dfdy_row):
            jac[i,j] -= (2/3) * h * df(current_iter[0],current_iter[1],t)
    return jac


# Reduce the stepsize if the error is too large
def calc_next_step_size(h,tol,est):
    return h * (.9*tol/np.max(abs(est)))**(1/3)


# Estimate the error of the BDF2 method: this is done by evaluating
# the interpolating polynomial with one more term, i.e.
# h_n(h_n + h_{n-1}) * [y_n, y_{n-1}, y_{n-2}, y_{n-3}]

def estimate_error(y,h,i,tol, next_iter, h_list):

    frac11 = (next_iter - y[:,i-1]) / h
    frac12 = (y[:,i-1] - y[:,i-2]) / h_list[-1]
    frac1 = (frac11 - frac12) / (h + h_list[-1])

    frac21 = (y[:,i-1] - y[:,i-2]) / h_list[-1]
    frac22 = (y[:,i-2] - y[:,i-3]) / h_list[-2]
    frac2 = (frac21 - frac22) / (h_list[-1] + h_list[-2])

    frac = (frac1 - frac2) / (h + h_list[-1] + h_list[-2])
    est = (h*(h + h_list[-1]) * frac) * h**2 * 2/3

    # Accept or reject the step depending on size of the error relative to tolerance
    if max(abs(est)) < tol:
        return {'signal': True, 'est': est}
    else:
        return {'signal': False, 'est':est}