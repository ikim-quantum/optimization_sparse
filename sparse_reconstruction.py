import numpy as np
import scipy.fftpack as spfft
import cvxpy as cvx


def min_spacing(mylist):
    """
    Find the minimum spacing in the list.

    Args:
        mylist (list): A list of integer/float.

    Returns:
        int/float: Minimum spacing within the list.
    """
    # Set the maximum of the minimum spacing.
    min_space = max(mylist) - min(mylist)

    # Iteratively find a smaller spacing.
    for item in mylist:
        spaces = [abs(item - item2) for item2 in mylist if item != item2]
        min_space = min(min_space, min(spaces))

    # Return the answer.
    return min_space


def reconstruct_from_signal(angles, signals, error=0, verb=False):
    """
    Reconstruct the signal from the given angles and signals.

    Args:
        angles (list): List of angles.
        signals (list): List of signals.
        error: An error strength in the 2-norm distance.

    Returns:
        list: Reconstructed signal.
    """
    # Find the minimum spacing.
    spacing = min_spacing(angles)
    itt = int(2 * np.pi / spacing)
    # Make the discret Fourier transform matrix.
    idct_matrix = spfft.idct(np.identity(itt), norm='ortho', axis=0)
    # Make the angles integer-valued.
    angles_discrete = [int(angle / spacing) for angle in angles]
    m_vecs = idct_matrix[angles_discrete]
    vx = cvx.Variable(itt)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [cvx.norm(m_vecs * vx - signals, 2) <= error]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=verb)
    return idct_matrix @ np.squeeze(np.array(vx.value))

def sparse_reconstruct(angles, signals, magnify = 10, error=0, verb=False):
    """
    Reconstruct the signal from the given angles and signals.

    Args:
        angles (list): List of angles.
        signals (list): List of signals.
        error: An error strength in the 2-norm distance.

    Returns:
        list: Reconstructed signal.
    """
    # Find the minimum spacing.
    spacing = min_spacing(angles)
    itt = int(2 * np.pi / spacing)
    # Make the discret Fourier transform matrix.
    idct_matrix = spfft.idct(np.identity(itt), norm='ortho', axis=0)
    # Make the angles integer-valued.
    angles_discrete = [int(angle / spacing) for angle in angles]
    m_vecs = idct_matrix[angles_discrete]
    vx = cvx.Variable(itt)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [cvx.norm(m_vecs * vx - signals, 2) <= error]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=verb)
    fcoefficients = np.squeeze(np.array(vx.value))
    dim_enlarged = itt * magnify
    fcoefficients_enlarged = np.zeros(dim_enlarged)
    for i in range(itt):
        fcoefficients_enlarged[i*magnify] = fcoefficients[i]

    idct_matrix_enlarged = spfft.idct(np.identity(dim_enlarged), norm='ortho', axis=0)
    
    return idct_matrix_enlarged @ fcoefficients_enlarged

