
from lib_N1 import DaC_eigen_N1, DaC_dyn_N1, energies_ED, PR_ED
import random
import numpy as np


def check_eigenvalues_DaC_N1( L, M, W):

    potential = np.random.uniform(-W, W, L)     
    hopping = np.ones(L-1)                    

    E, PR = DaC_eigen_N1( system = L, subsystem = M, potential = potential, hopping = hopping )

    E = np.sort(E)

    if( len(E) == L and L < 6000):
    
        energies = energies_ED( potential = potential, hopping = hopping)
        assert np.allclose( energies, E ) == True



def check_PR_dyn_N1( L, M, W ):

    h = np.random.uniform(-W, W, L)
    Jxx = 1
    precision = 1e-4
    time = np.arange(0, 10, 0.5)

    variance = 1e-30
    min_jump = 1
    error_propagation_ratio = 10


    PR, sites = DaC_dyn_N1( system = L, subsystem = M, potential = h, Jxx = Jxx, precision = precision, time = time, variance = variance, min_jump = min_jump, error_propagation_ratio = error_propagation_ratio )


    if( len(sites) == L and L < 6000):

        PR_T_ED = PR_ED( potential = h, hopping = np.zeros(L-1) + Jxx, time_interest = time )
        max_error = np.amax( np.fabs(PR_T_ED - PR) )

        assert (max_error < precision) == True





def test_eigen_N1():

    M = 200                       

    L_array = [600, 800]
    W_array = [10, 20]

    for L in L_array:
        for W in W_array:
            
            check_eigenvalues_DaC_N1(L, M, W)





def test_dyn_N1():

    M = 200                       

    L_array = [600, 800]
    W_array = [10, 20]

    for L in L_array:
        for W in W_array:
            
            check_PR_dyn_N1(L, M, W)












