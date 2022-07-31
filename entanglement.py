import time

import numpy as np
import pickle
import matplotlib.pyplot as pl
import itertools
import importlib


def configuration_to_index(confi):
    if len(np.shape(confi)) != 1:
        L = len(confi[0])
    else:
        L = len(confi)
    indices = 2**(np.arange(L-1,-1,-1))
    confi = np.array(confi)
    index = np.sum((confi)*indices,axis=len(confi.shape)-1)
    return index


def index_to_confi(index, L, system=None):
    if system is None:
        system = np.arange(0,L,1)
    system = np.array(system)
    len_sys = len(system)
    if type(index)!=int:
        index = np.array(index)
        len_index = len(index)
        temp =  index.reshape((len_index,1)) / (2 ** (L - 1 - system.reshape((1,len_sys))))
        return temp.astype(int) % 2

    else:
        return ((index / (2 ** (L - 1 - system))).astype(int)) % 2


def reduced_density_matrix(vector, sub_system):
    #sub_system is a list of indices belonging to sub-system A
    sub_system = np.array(sub_system)
    L = int(np.log2(len(vector)))
    A = len(sub_system)
    
    # psi matrix is writing psi = psi_matrix_ij |i>_A |j>_B
    psi_matrix = np.zeros((2**A,2**(L-A)),dtype=complex)
    
    system_indices = list(range(L))
    complement = np.array([i for i in system_indices if i not in sub_system])

    temp = np.arange(0, 2**L, 1)
    A_config = index_to_confi(temp, L, sub_system)
    B_config = index_to_confi(temp, L, complement)
    A_index = configuration_to_index(A_config)
    B_index = configuration_to_index(B_config)
    psi_matrix[A_index, B_index] = vector[temp]
    # for i in range(2**L):
    #     A_config = ((i/(2**(L-1-sub_system))).astype(int))%2
    #     B_config = ((i/(2**(L-1-complement))).astype(int))%2
    #     A_index = configuration_to_index(A_config)
    #     B_index = configuration_to_index(B_config)
    #     psi_matrix[A_index,B_index] = vector[i]
    
    u,schmidt_values,v = np.linalg.svd(psi_matrix,compute_uv=True,full_matrices=False)
    return u, (schmidt_values), v


def renyi_entropy(vector, sub_system, renyi_index):
    _,schmidt_values,_ = reduced_density_matrix(vector, sub_system)
    if np.round(np.sum(schmidt_values**2),6)!=1:
        print('ah, Schimdt values not normalized',sub_system, np.sum(schmidt_values**2))
    if renyi_index == 1:
        schmidt_values[schmidt_values<1e-20]=1
        entropy = np.sum(-2*schmidt_values**2*np.log2(schmidt_values))
    else:
        entropy = np.log2(np.sum(schmidt_values**(2*renyi_index)))/(1-renyi_index)
    
    return entropy



def sum_tuple(tuple):
    i = ()
    for j in tuple:
        # print(j)
        i += j
    return i


def sub_system_negativity(vector, sys_A: list, sys_B: list):
    sys_AB = sorted(sys_A + sys_B)
    L_A = len(sys_A)
    L_B = len(sys_B)
    dim = 2 ** (L_A + L_B)
    dim_A = 2 ** (L_A)
    dim_B = 2 ** (L_B)
    temp = np.arange(0,L_A+L_B,1)
    sys_A = [i for i in temp if sys_AB[i] in sys_A]
    sys_B = [i for i in temp if sys_AB[i] in sys_B]

    u_AB, sch_val, v_C = reduced_density_matrix(vector, sys_AB) # C is complement of AB
    # column of u_AB are schmidt states in AB
    no_of_schmidt_states = u_AB.shape[1]
    rho_AB = np.zeros((no_of_schmidt_states,2**L_A,2**L_B,2**L_A,2**L_B),dtype=complex)

    ##converting AB basis to A + B product basis index
    ind_AB = np.arange(0,dim,1)
    configuration_A = index_to_confi(ind_AB, L_A+L_B, system=sys_A)
    configuration_B = index_to_confi(ind_AB, L_A+L_B, system=sys_B)
    ## another cheaper way is to get index_A,B directly from index but this works only if AB is divided in contiguous region.
    index_A = configuration_to_index(configuration_A)
    index_B = configuration_to_index(configuration_B)
    # print(index_A)

    state = u_AB.T
    # state = np.sum(np.abs(state))
    state_AB = np.zeros((no_of_schmidt_states,dim_A,dim_B),dtype=complex)
    state_AB[:,index_A,index_B] = state[:]

    #
    start = time.time()
    # indices_AB = list(itertools.product(np.arange(0,dim_A,1), np.arange(0,dim_B,1)))
    # indices_ABAB = list(map(sum_tuple, itertools.product(indices_AB, indices_AB)))
    # print(index_A.shape,index_B.shape,np.array(indices_AB).shape,np.array(indices_ABAB).shape)
    rho_AB[:,:,:,:,:] += sch_val.reshape((no_of_schmidt_states,1,1,1,1))**2 * state_AB.reshape((no_of_schmidt_states,dim_A, dim_B, 1, 1)) * np.conjugate(state).reshape((no_of_schmidt_states,1, 1, dim_A, dim_B))
    rho_AB = np.sum(rho_AB,axis=0)

    # for i, state in enumerate(u_AB.T):
        #normalizing state vector; state is schmidt vector on AB.
        # state = state/np.sqrt(np.sum(np.abs(state)**2))
        #
        # state_AB = np.zeros((dim_A,dim_B),dtype=complex)
        # state_AB[index_A,index_B] = state
        #
        # indices_AB = list(itertools.product(index_A,index_B))
        # indices_ABAB =list(map(sum_tuple,itertools.product(indices_AB,indices_AB)))
        # rho_AB[indices_ABAB] += sch_val[i]*state_AB.reshape((dim_A,dim_B,1,1))*state.reshape((1,1,dim_A,dim_B))


    rho_AB_T = rho_AB.transpose(0,3,2,1)

    rho_AB_T_2 = np.empty((dim,dim),dtype=complex)
    for i in range(dim):
        rho_AB_T_2[i,:] = rho_AB_T[index_A[i],index_B[i],index_A,index_B]
    rho_AB_2 = np.empty((dim,dim),dtype=complex)
    for i in range(dim):
        rho_AB_2[i, :] = rho_AB[index_A[i], index_B[i], index_A, index_B]
    #
    # rho_AB_T_2 = rho_AB_T.reshape((dim,dim))
    # rho_AB_2 = rho_AB.reshape((dim, dim))
    # print(np.sum(sch_val**2),np.trace(np.trace(rho_AB,axis1=1,axis2=3)),)

    print(time.time() - start)

    trace_T = (np.trace(np.linalg.matrix_power(rho_AB_T_2,3),axis1=1,axis2=0))
    trace = (np.trace(np.linalg.matrix_power(rho_AB_2,3), axis1=0, axis2=1))
    negativity = -0.5*np.log(trace_T/trace)
    return negativity


    
def z_measurement(measurement_result, measurement_position, vector, spin_array):
    spin_not_measured = 1 - measurement_result
    uncompatible_confi = np.where(spin_array[measurement_position]==spin_not_measured)[0]
    vector[uncompatible_confi] = 0
    vector = vector/np.sqrt(np.sum(np.abs(vector)**2))
    return vector
    
    
def entropy_vs_measurements(L, renyi_index,integrable=False,window=10):
    values, vectors, _, _  = Level_statistics.run(L,integrable)
    bipartite_ent = []
    N = len(values)
    index_array = np.arange(0,2**L,1)
    spin_array = ((index_array.reshape((1,2**L))/(2**(L-1-np.arange(0,L,1).reshape((L,1))))).astype(int))%2
    # sub_system = np.arange(0,int(L),2)
    sub_system = np.arange(0,int(L/2),1)
    
    for i in range(int(N/2)-window,int(N/2)+window):
        vector = vectors[i]
        entropy_for_given_state = []
        measurement_indices = np.random.choice(np.arange(0,L,1),L,replace=False)
        measured_vector = np.array(vector.copy())
        entropy_for_given_state.append(renyi_entropy(measured_vector,sub_system,renyi_index))
        
        for m in measurement_indices:
            spin_measured = np.random.choice([0,1],1)
            measured_vector = z_measurement(spin_measured, m, measured_vector, spin_array)
            entropy_for_given_state.append(renyi_entropy(measured_vector,sub_system,renyi_index))
            
        bipartite_ent.append(entropy_for_given_state)
    
    return bipartite_ent
            
            
def run(L,integrable=False):
    values, vectors, _, _  = Level_statistics.run(L,integrable)
    bipartite_ent = []
    N = len(values)
    sub_system = np.arange(0,int(L/2),1)

    for i in range(int(N/2)-100,int(N/2)+100):
        entropy = renyi_entropy(vectors[i],sub_system,1)
        bipartite_ent.append(entropy)
    return bipartite_ent


# ent = entropy_vs_measurements(14,2,False)
# with open('data/entanglement/half_system_L=14' 'rb') as f:
#     pickle.dump(ent,f)