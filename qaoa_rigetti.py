from pyquil.quil import Program
from pyquil.api import get_qc
import numpy as np
import scipy.linalg as la
from scipy import sparse
import matplotlib.pyplot as plt
import scipy.fftpack as spfft
import cvxpy as cvx

def setup_forest_objects():
    qc = get_qc("9q-generic-qvm")
    return qc

# Cosine transform
# I have no idea what the hell is going on here.
def reconstruct_sparse_cosine(m_vecs,y_vals,verb=False):
    itt=100
    idct_matrix = spfft.idct(np.identity(itt), norm='ortho', axis=0)
    m_vecs = idct_matrix[rand_idx] #-> Needs to be fixed!!!!!!!!!
    vx = cvx.Variable(itt)
    objective = cvx.Minimize(cvx.norm(vx,1))
    constraints = [m_vecs*vx == y_vals]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=verb)
    return np.squeeze(np.array(vx.value))

def min_spacing(mylist):
    min_space = max(mylist) - min(mylist)
    for item in mylist:
        spaces = [abs(item-item2) for item2 in mylist if item!=item2]
        min_space = min(min_space,min(spaces))

    return min_space

# reconstruction: my version
def reconstruct_from_signal(angles, signals, verb=False):
    spacing = min_spacing(angles)
    itt = int(2*np.pi / spacing)
    # itt is the number of distinct angles.
    idct_matrix = spfft.idct(np.identity(itt), norm='ortho', axis=0)
    angles_discrete = [int(angle / spacing) for angle in angles]
    m_vecs = idct_matrix[angles_discrete]
    vx = cvx.Variable(itt)
    objective = cvx.Minimize(cvx.norm(vx,1))
    constraints = [m_vecs*vx == signals]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=verb)
    return idct_matrix @ np.squeeze(np.array(vx.value))
    
# Outputs random sample of angles for qaoa.
# Only one angle is changed(which)
def qaoa_maxcut_randsample(edges, angles, shots, which, sample_size=100, sample_size_reduced=60):
    rand_angles = np.array([2*np.pi*i/sample_size for i in range(sample_size)])
    np.random.shuffle(rand_angles)

    angles_sampled = rand_angles[:sample_size_reduced]
    qaoa_sampled = []
    for angle in angles_sampled:
        angles[which] = angle
        qaoa_sampled.append(qaoa_maxcut(edges, angles, shots))

    return angles_sampled, qaoa_sampled

# Outputs uniform sample of angles for qaoa.
# Only one angle is changed(which)
def qaoa_maxcut_uniformsample(edges, angles, shots, which, sample_size=60):
    rot_angles = np.array([2*np.pi*i/sample_size for i in range(sample_size)])
    qaoa_sampled = []
    for angle in rot_angles:
        angles[which] = angle
        qaoa_sampled.append(qaoa_maxcut(edges, angles, shots))

    return rot_angles, qaoa_sampled


# qaoa reconstruction
def qaoa_maxcut_opt(edges, angles, shots=1000, sweeps=5):
    for i in range(sweeps):
        for j in range(len(angles)):
            # Get random samples
            randangles, randsignals = qaoa_maxcut_randsample(edges, angles, shots=shots, which=j, sample_size=1000,sample_size_reduced=30)
            # Retrieve the sparse Fourier coefficients
            sparse_signal = reconstruct_from_signal(randangles, randsignals, verb=False)
            # Find the index that minimizes sparse_signal.
            idx = np.argmax(sparse_signal)
            # Find the actual value of the angle corresponding to idx.
            angle_opt = idx * 2 * np.pi / len(sparse_signal)
            # Stick in the optimal angle.
            angles[j] = angle_opt
            current_value = qaoa_maxcut(edges, angles, 1000)
            print("{}th sweep, {}th angle update: {}".format(i,j,current_value))
    
            

def qaoa_customquil(edges, p):
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

def qaoa_quil():
    program = Program("""
# set up memory
# beta: X
# gamma: ZZ
DECLARE beta REAL
DECLARE gamma REAL
DECLARE ro BIT[3]

# set up initial state
X 0
X 1
X 2

# Apply gamma gate
CNOT 0 1
RZ(gamma) 1
CNOT 0 1

CNOT 1 2
RZ(gamma) 2
CNOT 1 2

# Apply beta gate
RX(beta) 0
RX(beta) 1
RX(beta) 2

# measure out the results
MEASURE 0 ro[0]
MEASURE 1 ro[1]
MEASURE 2 ro[2]
""")
    return program

def run():
    shots = 1000
    qc = setup_forest_objects()

    program = qaoa_quil()
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    qc.qam.write_memory(region_name="gamma", value=50)
    qc.qam.write_memory(region_name="beta", value=30)
    qc.qam.run()
    qc.qam.wait()

    totals = [0] * 3
    for array in qc.qam.read_from_memory_region(region_name="ro", offsets=True):
        for k in range(3):
            totals[k] += array[k]

    print([total/shots for total in totals])

def qaoa_maxcut(edges, angles, shots=1000):
    """
    Returns the objective function of the maxcut problem.
    edges: Set of edges
    angles: 2n real numbers. The first n are the betas and the last n 
            are the gammas.
    """
    n = len(angles) // 2
    gammas = angles[:n]
    betas = angles[n:2*n]
    qc = setup_forest_objects()

    program = qaoa_customquil(edges, n)
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    for i in range(n):
        qc.qam.write_memory(region_name="gamma{}".format(i), value=gammas[i])
        qc.qam.write_memory(region_name="beta{}".format(i), value=betas[i])
    qc.qam.run()
    qc.qam.wait()

    f_sum = 0.0
    for array in qc.qam.read_from_memory_region(region_name="ro", offsets=True):
        array_pm = [2 * value - 1 for value in array]
        for e in edges:
            f_sum += (1- array_pm[e[0]] * array_pm[e[1]])/2

    # Note that f_sum needs to be MAXIMIZED, not minimized.
    return f_sum/shots
    
def qaoa_ising(angles):
    """
    Returns the energy of a transverse field Ising model on a length=3 chain.
    """

    gamma = angles[0]
    beta = angles[1]
    
    shots = 1000
    qc = setup_forest_objects()


    program = qaoa_quil()
    program.wrap_in_numshots_loop(shots=shots)
    nq_program = qc.compiler.quil_to_native_quil(program)
    binary = qc.compiler.native_quil_to_executable(nq_program)
    qc.qam.load(binary)

    qc.qam.write_memory(region_name="gamma", value=gamma)
    qc.qam.write_memory(region_name="beta", value=beta)
    qc.qam.run()
    qc.qam.wait()

    energy_sum=0.0
    for array in qc.qam.read_from_memory_region(region_name="ro", offsets=True):
        array_pm = [2 * value - 1 for value in array]
        energy_sum += - array_pm[0] * array_pm[1] - array_pm[1] * array_pm[2]

    return energy_sum / shots
    
