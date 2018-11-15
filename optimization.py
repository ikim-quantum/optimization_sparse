import numpy as np
from qaoa import qaoa_maxcut_randsample, qaoa_maxcut
from sparse_reconstruction import reconstruct_from_signal
import matplotlib.pyplot as plt

def qaoa_maxcut_opt_single(
        edges, angles, which, sample_size, sample_size_reduced, shots=1000, draw=False):
    """
    Optimizes a single angle for the MAXCUT QAOA.

    Args:
        edges (list): Edges that define the graph of the MAXCUT problem.
        angles (list): QAOA angles.
        which (int): The index of the angle we are optimizing.
        sample_size (int): Number of spacings for the variable angle.
        sample_size_reduced (int): Reduced number of samples over the
                                   variable angle.
        shots (int): Total number of single-shot measurements.

    Returns:
        angles (list): Optimized angles.
    """
    randangles, randsignals = qaoa_maxcut_randsample(
        edges, angles, shots=shots, which=which, sample_size=sample_size,
        sample_size_reduced=sample_size_reduced)
    # Estimate the error in 2-norm.
    error = pow(len(edges), 2) * len(randangles) / shots
    # Reconstruct the signal
    sparse_signal = reconstruct_from_signal(
        randangles, randsignals, error=0)
    # Find the index that maximizes the signal.
    idx = np.argmax(sparse_signal)
    angle_opt = idx * 2 * np.pi / len(sparse_signal)

    if draw:
        plt.plot(sparse_signal)
        plt.plot(idx, sparse_signal[idx], marker='o', markersize=3, color="red")
        plt.show()

    return angle_opt


def sparse_maxcut_opt(edges, angles, sample_size,
                      sample_size_reduced, shots=1000, sweeps=5, draw=False):
    """
    Optimizing each angles sequentially for the MAXCUT QAOA

    Args:
        edges (list): Edges that define the graph of the MAXCUT problem.
        angles (list): QAOA angles.
        sample_size (int): Number of spacings for the variable angle.
        sample_size_reduced (int): Reduced number of samples over the
                                   variable angle.
        shots (int): Total number of single-shot measurements.
        sweeps (int): Total number of sweeps.

    Returns:
        angles (list): List of optimal angles.
    """
    for i in range(sweeps):
        for j in range(len(angles)//2):
            angles[j] = qaoa_maxcut_opt_single(
                edges, angles, j, sample_size, sample_size_reduced,
                shots=shots, draw=draw)
            angles[j+len(angles)//2] = qaoa_maxcut_opt_single(
                edges, angles, j+len(angles)//2, sample_size, sample_size_reduced,
                shots=shots, draw=draw)
            
            current_value = qaoa_maxcut(edges, angles, 1000)
            print(
                "{}th sweep, {}th angle update, sample size={}: {}".format(
                    i, j, shots * sample_size_reduced * (i * len(angles) + j),current_value))

    print("Total number of samples: {}".format(shots * sample_size_reduced * sweeps * len(angles)))
    return angles

def sparse_maxcut_opt_simple(edges, angles, shots=1000, sweeps=5, draw=False):
    """
    Optimizing each angles sequentially for the MAXCUT QAOA

    Args:
        edges (list): Edges that define the graph of the MAXCUT problem.
        angles (list): QAOA angles.
        shots (int): Total number of single-shot measurements.
        sweeps (int): Total number of sweeps.

    Returns:
        angles (list): List of optimal angles.
    """
    sample_size = 100
    sample_size_reduced = int(np.sqrt(sample_size))
    for i in range(sweeps):
        for j in range(len(angles) // 2):
            angles[j] = qaoa_maxcut_opt_single(
                edges, angles, j, sample_size, sample_size_reduced,
                shots=sample_size_reduced, draw=draw)
            angles[j + len(angles)//2] = qaoa_maxcut_opt_single(
                edges, angles, j+len(angles)//2, sample_size, sample_size_reduced,
                shots=sample_size_reduced, draw=draw) 
            current_value = qaoa_maxcut(edges, angles, 1000)
            print(
                "{}th sweep, {}th angle update, sample size={}: {}".format(
                    i, j, shots * (i * len(angles) + j),current_value))

    print("Total number of samples: {}".format(shots * sweeps * len(angles)))
    return angles

def spsa_maxcut_opt(edges, angles, shots, alpha = 0.602, gamma=0.101, itt=1000):
    """
    Optimizing a MAXCUT problem with SPSA.

    Args:
        edges (list): Edges that define the graph of the MAXCUT problem.
        angles (list): QAOA angles.
        shots (int): Total number of single-shot measurements.
        alpha (float): Parameter for SPSA
        gamma (float): Parameter for SPSA
        itt (int): Total number of iterations
    """

    for k in range(itt):
        c_k = 0.01 / pow((1+k), gamma)
        a_k = 0.01 / pow((1+k), alpha)
        randombit = np.random.randint(2, size=len(angles))
        randpm = 2* randombit - 1
        angles_p = angles + c_k * randpm
        angles_m = angles - c_k * randpm
        g_k = (qaoa_maxcut(edges, angles_p, shots=shots) - qaoa_maxcut(edges, angles_m)) / (2*c_k)

        angles = [angle + g_k * mybit for (angle, mybit) in zip(angles,randpm)]
        total_shots = k * shots * 2
        print("Total shots = {}: {}".format(total_shots, qaoa_maxcut(edges, angles)))

#def powell_maxcut_opt(edges, angles, shots):
    
