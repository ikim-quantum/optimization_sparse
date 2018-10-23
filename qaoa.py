from pyquil.quil import Program
from pyquil.api import get_qc
import numpy as np
import matplotlib.pyplot as plt


def setup_forest_objects(device="9q-generic-qvm"):
    """
    Set up a quantum computer/simulator.

    Args:
        device (str): Name of the device

    Returns:
        forest object.
    """

    qc = get_qc(device)
    return qc


def qaoa_maxcut(edges, angles, shots=1000, device="9q-generic-qvm"):
    """
    Estimate the objective function of the MAXCUT problem specified in terms
    of a graph over a QAOA ansatz. The graph is specified in terms of its
    edges. The QAOA ansatz is specified in terms of its rotation angles.

    Args:
        edges (list): list of edges
        angles (list): 2n real numbers. The first n are the betas and the last
                       n are the gammas.
        shots (int): Number of one-shot measurements.
        device (str): Name of the device we are using.

    Returns:
        float: the objective function of the MAXCUT problem.
    """

    # Number of QAOA layers and angles.
    n = len(angles) // 2
    gammas = angles[:n]
    betas = angles[n:2 * n]

    # Set up a quantum computer/simulator.
    qc = setup_forest_objects(device)

    # Generate a Quil program that estimates the objective function.
    program = qaoa_maxcut_2quil(edges, n)
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    # Fill in the variable angles.
    for i in range(n):
        qc.qam.write_memory(region_name="gamma{}".format(i), value=gammas[i])
        qc.qam.write_memory(region_name="beta{}".format(i), value=betas[i])

    # Run the QAOA algorithm.
    qc.qam.run()
    qc.qam.wait()

    # Estimate the objective function.
    f_sum = 0.0
    for array in qc.qam.read_from_memory_region(
            region_name="ro", offsets=True):
        array_pm = [2 * value - 1 for value in array]
        for e in edges:
            f_sum += (1 - array_pm[e[0]] * array_pm[e[1]]) / 2

    # Return the objective function.
    return f_sum / shots


def qaoa_maxcut_randsample(edges, angles, shots=1000, which=0,
                           sample_size=100, sample_size_reduced=60,
                           device="9q-generic-qvm"):
    """
    Sample the objective function over random angles.

    Args:
        edges (list): list of edges
        angles (list): 2n real numbers. The first n are the betas and the last
                       n are the gammas.
        shots (int): Number of one-shot measurements for each angles.
        which (int): The variable angle.
        sample_size (int): Number of spacings over the variable angle.
        sample_size_reduced (int): Number of samples.
        device (str): Name of the device we are using.

    Returns:
        angles_sampled (list): Randomly chosen angles.
        qaoa_sampled (list): MAXCUT objective function for each angles.
    """

    # Pick the random angles.
    rand_angles = np.array(
        [2 * np.pi * i / sample_size for i in range(sample_size)])
    np.random.shuffle(rand_angles)
    angles_sampled = rand_angles[:sample_size_reduced]

    # Estimate the objective function for each angles.
    qaoa_sampled = []
    for angle in angles_sampled:
        angles[which] = angle
        qaoa_sampled.append(qaoa_maxcut(edges, angles, shots, device))

    # Return the random angles and the associated objective functions.
    return angles_sampled, qaoa_sampled


def qaoa_maxcut_uniformsample(
        edges, angles, shots=1000, which=0, sample_size=60,
        device="9q-generic-qvm"):
    """
    Sample the objective function over different angles. The spacing between
    two adjacent angles are fixed.

    Args:
        edges (list): list of edges
        angles (list): 2n real numbers. The first n are the betas and the last
                       n are the gammas.
        shots (int): Number of one-shot measurements for each angles.
        which (int): The variable angle.
        sample_size (int): Numbr of spacings over the variable angle.
        device (str): Name of the device we are using.

    Returns:
        rot_angles (list): Different angles.
        qaoa_sampled (list): MAXCUT objective function for each angles.
    """

    # Pick the angles uniformly.
    rot_angles = np.array(
        [2 * np.pi * i / sample_size for i in range(sample_size)])

    # Estimate the objective function for each angles.
    qaoa_sampled = []
    for angle in rot_angles:
        angles[which] = angle
        qaoa_sampled.append(qaoa_maxcut(edges, angles, shots, device))

    # Return the list of angles and the associated objective function.
    return rot_angles, qaoa_sampled


def qaoa_maxcut_2quil(edges, p):
    """
    Generate a QAOA instance for MAXCUT problem.

    Args:
        edges (list): List of edges
        p (int): Number of QAOA steps

    Returns:
        Quil program that encodes QAOA for MAXCUT.
    """
    vertices = set([])

    # Find all the vertices
    for edge in edges:
        vertices = vertices | set(edge)

    # Declare all the variables
    program = ""

    for k in range(p):
        program += "DECLARE beta{} REAL\n".format(k)
        program += "DECLARE gamma{} REAL\n".format(k)

    program += "DECLARE ro BIT[{}]\n".format(len(vertices))

    # Set up initial states
    for v in vertices:
        program += "X {}\n".format(v)

    # Apply gamma and beta gates
    for k in range(p):
        # gamma gate
        for e in edges:
            program += "CNOT {} {}\n".format(e[0], e[1])
            program += "RZ(gamma{}) {}\n".format(k, e[1])
            program += "CNOT {} {}\n".format(e[0], e[1])
        # beta gate
        for v in vertices:
            program += "RX(beta{}) {}\n".format(k, v)

    # Measure
    for v in vertices:
        program += "MEASURE {} ro[{}]\n".format(v, v)

    # Return a quil script
    return Program(program)
