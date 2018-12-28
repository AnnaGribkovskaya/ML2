import numpy as np



def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E

def generate_data_set(states,L,n_samples):
    energies=ising_energies(states,L)
    states=np.einsum('...i,...j->...ij', states, states)
    shape=states.shape
    states=states.reshape((shape[0],shape[1]*shape[2]))
    data=[states,energies]
    X_train=data[0][:n_samples]
    Y_train=data[1][:n_samples]
    X_test=data[0][n_samples:3*n_samples//2]
    Y_test=data[1][n_samples:3*n_samples//2] 
    states=states.reshape((shape[0],shape[1]*shape[2]))
    data=[states,energies]

    return data, X_train, X_test, Y_train, Y_test