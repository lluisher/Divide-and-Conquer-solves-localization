'''
File with the test, both for eigenstates and dynamics of the Anderson model.
'''
from lib_N1 import DaC_eigen_N1, DaC_dyn_N1, energies_ED, PR_ED
import random
import numpy as np
from classes_DaC import System_Parameters_Anderson, Technical_Parameters_Anderson_Eigen, Technical_Parameters_Anderson_Dyn



def check_eigenvalues_DaC_N1( L, M, W):
    '''
    Check that we obtain the same eigenstates using DaC and ED
    (system should be small enough to do ED).
    '''

    potential = np.random.uniform(-W, W, L)
    hopping_dist = np.ones(L-1)

    Physical_parameters = System_Parameters_Anderson( L, W, potential, hopping_dist)

    DaC_paramenters = Technical_Parameters_Anderson_Eigen(M)


    E, PR, population = DaC_eigen_N1( Physical_parameters, DaC_paramenters )

    E = np.sort(E)

    if( len(E) == L and L < 6000):

        energies = energies_ED( potential = potential, hopping = hopping_dist)
        assert np.allclose( energies, E ) == True

    if( len(E) == L ):
        assert np.allclose( population, np.ones(L) ) == True


def check_PR_dyn_N1( L, M, W ):
    '''
    Check that we obtain the dynamics using DaC and ED
    (system should be small enough to do ED).
    '''

    h = np.random.uniform(-W, W, L)
    Jxx = 1
    precision = 1e-4

    Physical_parameters = System_Parameters_Anderson( L, W, h, hopping_strength=Jxx)

    DaC_paramenters = Technical_Parameters_Anderson_Dyn(M, precision = precision)

    time = np.arange(0, 10, 0.5)


    PR, sites = DaC_dyn_N1( Physical_parameters, time, DaC_paramenters )


    if( len(sites) == L and L < 6000):

        PR_T_ED = PR_ED( potential = h, hopping = np.zeros(L-1) + Jxx, time_interest = time )
        max_error = np.amax( np.fabs(PR_T_ED - PR) )

        assert (max_error < precision) == True





def test_eigen_N1():
    '''
    Provide input for some test (eigenstates).
    '''
    M = 200

    L_array = [600, 800]
    W_array = [10, 20]

    for L in L_array:
        for W in W_array:

            check_eigenvalues_DaC_N1(L, M, W)





def test_dyn_N1():
    '''
    Provide input for some test (dynamics).
    '''
    M = 200

    L_array = [600, 800]
    W_array = [10, 20]

    for L in L_array:
        for W in W_array:

            check_PR_dyn_N1(L, M, W)
