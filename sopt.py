# Sparse Optimization
import numpy as np
import scipy.fftpack as spfft
import cvxpy as cvx
import matplotlib.pyplot as plt
from sparse_reconstruction import reconstruct_from_signal, sparse_reconstruct

def sparse_minimize(cost, x0, cutoff, samples, itt = None, args=None):
    # Sample points
    # The cutoff is the frequency cutoff
    # If the frequency is in 2\pi[0, n], then sample should
    # be taken between 0 and n.
    if samples > cutoff:
        samples = cutoff

    # Number of parameters
    num_param = len(x0)
    
    # Default number of iterations = num_param * 5
    if itt:
        itt_opt = itt
    else:
        itt_opt = num_param * 5

    history = []
    for j in range(itt_opt):
        for which in range(num_param):
            # Sample random angles.
            rand_angles = np.array([2*np.pi * i / cutoff for i in range(cutoff)])
            np.random.shuffle(rand_angles)
            angles_sampled = rand_angles[:samples]
            # Sample random signals.
            signals = []
            for angle in angles_sampled:
                xtemp = [value for value in x0]
                xtemp[which] = angle
                signals.append(cost(xtemp, args))

            angles_sorted = np.sort(angles_sampled)
            signals_sorted = []
            for angle in angles_sorted:
                xtemp = [value for value in x0]
                xtemp[which] = angle
                signals_sorted.append(cost(xtemp, args))

#            plt.plot(angles_sorted, signals_sorted)
#            plt.show()

            signal = sparse_reconstruct(angles_sampled, signals, magnify = 1, error=0)
            # Find the minimum.
            idx = np.argmin(signal)
            angle_opt = idx * 2 * np.pi / len(signal)
            
#            plt.plot(signal)
#            plt.plot(idx, signal[idx], marker='o', markersize=3, color="red")
#            plt.show()
            
            # Replace the angle.
            x0[which] = angle_opt
            mycost=cost(x0, args)
            history.append(mycost)

            print("itt={}, which={}, cost={}".format(j, which, mycost), end='\r')
        
    return x0, cost(x0, args), history

    
    
