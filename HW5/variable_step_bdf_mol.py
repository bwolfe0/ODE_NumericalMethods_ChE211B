import numpy as np
from numpy.linalg import solve
import pandas as pd
import matplotlib.pyplot as plt


######## NOTE: I found this code more complicated, so it is somewhat hardcoded compred to variable_step_bdf #########


def variable_h_bdf_mol(dx, tol=1e-6):
    # Length of x, and t intervals
    L = 1
    t_end = 1

    # Setup the x values and intialize BC and IC given in the problem
    x_vals = np.arange(0, L+dx, dx)
    boundary_conditions = [0,0]
    initial_conditions = [2*x if x <=.5 else 2-2*x for x in x_vals]

    # Setup the solution matrix. It starts with only one column, with an additional
    # one added at each step

    y = np.zeros((len(x_vals),1))
    y[0,:] = boundary_conditions[0]
    y[-1,:] = boundary_conditions[1]
    y[:,0] = initial_conditions

    # A is our matrix that is defined by the central difference method
    # We have that dy_i/dt = A * y_i where i represents the grid values for x
    # except for the two boundaries A, is a sparse diagonal matrix with [1, -2, 1] in each row
    # and -2 on the true diagonal

    A = np.diag([-2]*(len(x_vals)-2)) + np.diag([1]*(len(x_vals)-3),1) + np.diag([1]*(len(x_vals)-3),-1)
    A = A/dx**2

    t = [0]
    i = 1
    h = .001
    run = True

    # We require at least 3 past values to start BDF2 and retreive and error estimate for the first step
    # We get these values using BEM here

    for j in range(1,3):
        y_copy = y.copy()

        # Here we extend y by an additional column at each step
        zero_col = np.zeros((len(x_vals), 1))
        y = np.hstack((y_copy, zero_col))

        ####### BEM #########
        b = (y_copy[1:-1,j-1])
        system_matrix = np.eye(len(x_vals) - 2) - h * A
        #####################
        
        soln = np.linalg.solve(system_matrix,b)
        y[1:-1,j] = soln

        t.append(
            t[-1] + h
        )
        i+=1

    # List to keep track of different step sizes
    h_list = [.001] * 2
    
    # Until t_end is surpassed
    while run is True:

        # Once t_end is surpassed, interpolate back to the desired endpoint

        if t[-1] > t_end:
            y[:,-1] = y[:,-2] + (t_end - t[-2]) * (y[:,-1] - y[:,-2]) / (t[-1] - t[-2])
            t[-1] = t_end
            return {'y': y, 't': t, 'h_list': h_list}
        

        # Our first guess is the last value
        next_iter = y[:,i-1]
        current_iter = next_iter + tol * 1000

        # Here we extend y by an additional column at each step. Important since total
        # number of steps to be taken is ambiguous for variable step method

        zero_col = np.zeros((len(x_vals), 1))
        y = np.hstack((y, zero_col))

        # Determine next solution and keep track of h values

        result = calc_next_step(y,i,tol,x_vals,h,A,next_iter,current_iter,h_list)
        y[1:-1,i] = result['next_iter']
        last_h = result['last_h']
        h = result['next_h']
        h_list.append(last_h)

        # Keep track of the time
        t.append(
            t[-1] + last_h
        )
        
        # Keep track of y size
        i+=1


# Here we run the Newton Method for a given stepsize and then estimate the error
# The function is called recursively until the error is sufficiently small

def calc_next_step(y,i,tol,x_vals,h,A,next_iter,current_iter,h_list): 

    # Run the Newton Method until it converges   
    while abs(np.max(current_iter - next_iter)) > tol:
        current_iter = next_iter.copy()

        ####### BDF2 ###########
        b = 4 * y[1:-1, i-1] - y[1:-1, i-2]
        system_matrix = 3 * np.eye(len(x_vals) - 2) - 2 * h * A
        ########################

        soln = np.linalg.solve(system_matrix,b)
        next_iter = np.hstack((0,soln,0))

    # Estimate error for the step just taken
    res = estimate_error(y,h,i,tol, next_iter,h_list)
    go = res['signal']
    est = res['est']

    # If step is accepted, proceed to next step with default step size
    if go is True:
        last_h = h
        h = calc_next_step_size(h,tol,est)
        return {'next_iter': next_iter[1:-1], 'last_h':last_h, 'next_h': h}
    
    # If step is denied, recursively call this function with a new, smaller step
    else:
        # Find the next step to try
        h = calc_next_step_size(h,tol,est)
        return calc_next_step(y,i,tol,x_vals,h,A,next_iter,current_iter,h_list)


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


# Reduce the stepsize if the error is too large

def calc_next_step_size(h,tol,est):
    return h * (.9*tol/np.max(abs(est)))**(1/3)