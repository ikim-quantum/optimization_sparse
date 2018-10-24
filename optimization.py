import numpy as np
from qaoa import qaoa_maxcut_randsample, qaoa_maxcut
from sparse_reconstruction import reconstruct_from_signal


def qaoa_maxcut_opt_single(
        edges, angles, which, sample_size, sample_size_reduced, shots=1000):
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
        randangles, randsignals, error=error)
    # Find the index that maximizes the signal.
    idx = np.argmax(sparse_signal)
    angle_opt = idx * 2 * np.pi / len(sparse_signal)

    return angle_opt


def qaoa_maxcut_opt(edges, angles, sample_size,
                    sample_size_reduced, shots=1000, sweeps=5):
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
        for j in range(len(angles)):
            angles[j] = qaoa_maxcut_opt_single(
                edges, angles, j, sample_size, sample_size_reduced,
                shots=shots)
            current_value = qaoa_maxcut(edges, angles, 1000)
            print(
                "{}th sweep, {}th angle update: {}".format(
                    i, j, current_value))

    return angles
