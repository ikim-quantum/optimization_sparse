import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import minimize
from sparse_reconstruction import reconstruct_from_signal

# Number of Qubits
N = 4

# Define Operators 
def gen_cij(edge):
    i,j = edge
    Id = [qeye(2) for n in range(N)]
    si_n = tensor(Id)
    Id[i] = sigmaz()
    Id[j] = sigmaz()
    zij = tensor(Id)
    return 0.5*(si_n - zij)

def gen_B():
    b_op = 0*tensor([qeye(2) for j in range(N)])
    for i in range(N):
        Id = [qeye(2) for j in range(N)]
        Id[i] = sigmax()
        b_op += tensor(Id)
    return b_op

def gen_init():
    init = tensor([basis(2,0) for i in range(N)])
    x_all = tensor([hadamard_transform(1) for i in range(N)])
    return (x_all*init).unit()

def gen_U(angles):
    edges = [[0,1],[1,2],[2,3],[3,0]]
    C = np.sum(gen_cij(edge) for edge in edges)
    B = gen_B()
    L = len(angles)
    gammas = angles[:int(L/2)] 
    betas = angles[int(L/2):]
    U = np.prod([(-1j*betas[i]*B).expm()*(-1j*gammas[i]*C).expm() for i in range(int(L/2))]) 
    return U

def cost(angles, sample_size, info):
    edges = [[0,1],[1,2],[2,3],[3,0]]
    C = np.sum(gen_cij(edge) for edge in edges)
    psi_init = gen_init()
    U_temp = gen_U(angles)
    psi_temp = U_temp*psi_init
    energy = -expect(C,psi_temp) + np.random.normal() / np.sqrt(sample_size)

    info['Nfeval'] += 1
#    print(energy)
    print(info, sep=' ', end='\r', flush=True)
    return energy

def opt_sparse(samples):
    myangle= np.random.rand(2)

    for i in range(5):
        # Get Random Samples
        sample_size = 1000
        sample_size_reduced = 50

        rand_angles = np.array([2 * np.pi  * i/sample_size for i in range(sample_size)])
        np.random.shuffle(rand_angles)
        angles_sampled = rand_angles[:sample_size_reduced]
        
        qaoa_sampled = []
        for angle in angles_sampled:
            myangle[0] = angle
            qaoa_sampled.append(cost(myangle, samples, {'Nfeval':0}))

        # Reconstruct and update
        sparse_signal = reconstruct_from_signal(angles_sampled, qaoa_sampled, error=0)
        idx = np.argmin(sparse_signal)
        angle_opt = idx * 2 * np.pi / len(sparse_signal)
        myangle[0]=angle_opt

        # Get Random Samples
        rand_angles = np.array([2 * np.pi  * i/sample_size for i in range(sample_size)])
        np.random.shuffle(rand_angles)
        angles_sampled = rand_angles[:sample_size_reduced]
        
        qaoa_sampled = []
        for angle in angles_sampled:
            myangle[1] = angle
            qaoa_sampled.append(cost(myangle, samples, {'Nfeval':0}))

        # Reconstruct and update
        sparse_signal = reconstruct_from_signal(angles_sampled, qaoa_sampled, error=0)
        idx = np.argmin(sparse_signal)
        angle_opt = idx * 2 * np.pi / len(sparse_signal)
        myangle[1] = angle_opt

        print(cost(myangle, 1000000, {'Nfeval':0}))

    
        


#num_layers = 1
#in_fun = lambda angles, info: cost(angles, 10000, info)
#sol_NM = minimize(in_fun,np.random.rand(2*num_layers),args=({'Nfeval':0}),method='Nelder-Mead',options={'maxiter':1000})
#print(sol_NM.x,sol_NM.fun)

#samples_per_strategy = 100
#totalitt = 2**12
#for k in range(6,12):
#    in_fun = lambda angles, info: cost(angles, 2**k, info)
#    cumulative = []
#    for j in range(samples_per_strategy):
#        sol_NM = minimize(in_fun,np.random.rand(2*num_layers),args=({'Nfeval':0}),method='Nelder-Mead',options={'maxiter':2**(12-k)})
#        cumulative.append(sol_NM.fun)
#    print("Min from {} samples = {} +- {}".format(2**k, np.mean(cumulative), np.std(cumulative)))
    

