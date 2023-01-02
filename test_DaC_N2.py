'''
File with the tests, for the dynamics of the TIP problem.
'''

from lib_N2 import DaC_eigen_N2, DaC_N2_dyn, PR_ED_N2
import random
import numpy as np
from classes_DaC import System_Parameters_TIP, Technical_Parameters_Eigen, Technical_Parameters_Dyn



def check_PR_dyn_N2( L, M, W ):
    '''
    Check that we obtain the dynamics using DaC and ED
    (system should be small enough to do ED).
    '''

    potential = np.random.uniform(-W, W, L)
    Jxx = 1
    Jz = 1

    Physical_parameters = System_Parameters_TIP( L, W, Jxx, Jz, potential)


    time = np.arange(0, 10, 0.5)

    cutoff_variance = 1e-16
    min_jump = 5
    error_propagation = 10
    precision = 1e-5

    DaC_paramenters = Technical_Parameters_Dyn(M, cutoff_variance = cutoff_variance,
                        min_jump = min_jump, error_propagation = error_propagation,
                        precision = precision)

    PR, sites = DaC_N2_dyn( Physical_parameters, time, DaC_paramenters )


    if( len(sites) != 0 and L <= 200):

        PR_T_ED = PR_ED_N2( potential = potential, Jxx = Jxx, Jz = Jz, time_interest = time )
        max_error = 0

        for j in range( 0, len(sites) ):
            real_site = sites[j]
            max_error = max(max_error, np.amax( np.fabs(PR_T_ED[real_site] - PR[j]) ) )

        assert (max_error < precision) == True


    if( L > 200):
        print("Too large system to do ED.")
        assert True


    if( len(sites) == 0 ):
        print("Too small subsystem size, not possible to obtain any time evolution with the desired precision.")
        assert True




def test_dyn_N2():
    '''
    Provide input for some test (dynamics).
    '''
    M = 40

    L_array = [60, 70]
    W_array = [35, 40]

    for L in L_array:
        for W in W_array:
            print(L, W)
            check_PR_dyn_N2(L, M, W)
