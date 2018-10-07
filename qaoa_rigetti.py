from pyquil.quil import Program
from pyquil.api import get_qc

def setup_forest_objects():
    qc = get_qc("9q-generic-qvm")
    return qc

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
    
