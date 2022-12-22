'''
Examples how to use the functions ***DaC_eigen_N1()*** and ***DaC_dyn_N1()***,
to obtain eigenstates and the dynamics of the Anderson problem.
'''

from lib_N1 import DaC_eigen_N1, DaC_dyn_N1, energies_ED, PR_ED
import random
import numpy as np

from classes_DaC import System_Parameters_Anderson, Technical_Parameters_Anderson_Eigen, Technical_Parameters_Anderson_Dyn


def example_eigen_N1():
    '''
    Remember, to use the ***DaC_eigen_N1()***, we need to provide:\n
    1. Object of class **System_Parameters_Anderson**
    (physical paremeters, like size of system).\n
    2. Object of class **Technical_Parameters_Anderson_Eigen**
    (parameters needed for the DaC algorithm).\n
    This function calls the ***DaC_eigen_N1()***
    and also calculates the energies using ED, from ***energies_ED()***.\n
    It prints the maximum differences between the two methods.
    '''

    L = 700
    W = 10
    potential = np.random.uniform(-W, W, L)
    hopping_dist = np.ones(L-1)

    Physical_parameters = System_Parameters_Anderson( L, W, potential, hopping_dist)

    M = 200
    DaC_paramenters = Technical_Parameters_Anderson_Eigen(M)

    E, PR, population = DaC_eigen_N1( Physical_parameters, DaC_paramenters)
    N = len(E)

    print("Obtained number of eigenstates with DaC:", N)

    if( len(E) == L and L < 5000):
        energies = energies_ED( potential = potential, hopping = hopping_dist)

        E = np.sort(E)

        print( "Maximum difference in the eigenvalues compared with ED:", np.amax( np.fabs(E - energies) ) )

    if( len(E) == L ):
        print( "Infinity norm of |population - identity|:", np.amax( np.fabs(population - np.ones(L)) ) )


example_eigen_N1()



def example_dyn_N1():
    '''
    Remember, to use the ***DaC_dyn_N1()***, we need to provide:\n
    1. Object of class **System_Parameters_Anderson**
    (physical paremeters, like size of system).\n
    2. Numpy array with the times of interest.\n
    3. Object of class **Technical_Parameters_Anderson_Dyn**
    (parameters needed for the DaC algorithm).\n
    This function calls the ***DaC_dyn_N1()*** to calculate the PR of considered
    initial states and the considered times. It compares the results with the
    ones from ED, using ***PR_ED()***.\n
    It prints the maximum differences between the two methods,
    and checks if indeed the difference is smaller than the requiered precision.
    '''

    time = np.arange(0, 10, 0.5)

    L = 500
    W = 10
    h = np.random.uniform(-W, W, L)
    Jxx = 1
    Physical_parameters = System_Parameters_Anderson( L, W, h, hopping_strength=Jxx)

    M = 200
    precision = 1e-7
    DaC_paramenters = Technical_Parameters_Anderson_Dyn(M, precision = precision)

    PR, sites = DaC_dyn_N1( Physical_parameters, time, DaC_paramenters)

    print("Number of initial states whose time evolution have been calculated:", len(sites))


    if( len(sites) == L and L < 5000):
        PR_T_ED = PR_ED( potential = h, hopping = np.zeros(L-1) + Jxx, time_interest = time )

        max_error = np.amax( np.fabs(PR_T_ED - PR) )

        print( "Maximum difference in the values of PR compared with ED in all the considered times:", max_error )
        print( "Precision requested was:", precision )

        if(max_error < precision):
            print("Maximum error PR smaller than needed precision. Great :)")
        else:
            print("ATTENTION! Error in PR exceeds needed precision! Try to reduce the cutoff of the variance, until precision is larger than the square root of the variance, or increase value of \"error_propagation_ratio\".")


example_dyn_N1()
