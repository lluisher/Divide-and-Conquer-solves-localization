'''
Examples how to use the functions ***DaC_eigen_N2()*** and ***DaC_dyn_N2()***,
to obtain eigenstates and the dynamics of the TIP problem.
'''



from lib_N2 import DaC_eigen_N2, DaC_N2_dyn
import random
import numpy as np
from classes_DaC import System_Parameters_TIP, Technical_Parameters_Eigen, Technical_Parameters_Dyn

def example_eigen_N2():
    '''
    Remember, to use the ***DaC_eigen_N2()***, we need to provide:\n
    1. Object of class **System_Parameters_TIP**
    (physical paremeters, like size of system).\n
    2. Object of class **Technical_Parameters_Eigen**
    (parameters needed for the DaC algorithm).\n
    This function calls the ***DaC_eigen_N2()***.\n
    It returns the set of obtained energies, the Observables of interest
    (as an object of the class **Observables_TIP_class**),
    the population in each consecutive site (i,i+1) and in which interval
    each of the obtained eigenstate is localized.
    '''

    L = 100
    W = 15
    potential = np.random.uniform(-W, W, L)
    Jxx = 1
    Jz = 1

    Physical_parameters = System_Parameters_TIP( L, W, Jxx, Jz, potential)

    M = 50
    cutoff_variance = 1e-16
    DaC_paramenters = Technical_Parameters_Eigen(M, cutoff_variance = cutoff_variance)

    E, Obs, popu, begin_site = DaC_eigen_N2( Physical_parameters, DaC_paramenters)

example_eigen_N2()



def example_dyn_N2():
    '''
    Remember, to use the ***DaC_dyn_N2()***, we need to provide:\n
    1. Object of class **System_Parameters_TIP**
    (physical paremeters, like size of system).\n
    2. Numpy array with the times of interest.\n
    3. Object of class **Technical_Parameters_Dyn**
    (parameters needed for the DaC algorithm).\n
    This function calls the ***DaC_dyn_N2()*** to calculate the PR of considered
    initial states and the considered times.
    '''

    L = 200
    W = 20
    potential = np.random.uniform(-W, W, L)
    Jxx = 1
    Jz = 1

    Physical_parameters = System_Parameters_TIP( L, W, Jxx, Jz, potential)

    time = np.linspace(0, 10, 50)

    M = 50
    precision = 1e-5
    cutoff_variance = 1e-16
    min_jump = 1
    error_propagation = 10

    DaC_paramenters = Technical_Parameters_Dyn(M, cutoff_variance = cutoff_variance,
                        min_jump = min_jump, error_propagation = error_propagation,
                        precision = precision)



    PR, real_sites = DaC_N2_dyn( Physical_parameters, time, DaC_paramenters)

    print(f"The number of initial wavefunctions whose time evolution" +
    f"can be calculated, with the required precision of 10^({np.log10(precision)})" +
    "and subsystem size {M} is {len(real_sites)}." +
    "The total number of initial states is {L-1}")


example_dyn_N2()
