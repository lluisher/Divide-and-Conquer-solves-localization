
from lib_N2 import DaC_eigen_N2, DaC_N2_dyn, PR_ED_N2
import random
import numpy as np



def check_PR_dyn_N2( L, M, W ):

    potential = np.random.uniform(-W, W, L)
    Jxx = 1
    Jz = 1
    precision = 1e-5
    time = np.arange(0, 10, 0.5)

    variance = 1e-16
    min_jump = 1
    error_propagation_ratio = 10


    PR, sites = DaC_N2_dyn( system = L, subsystem = M, potential = potential, Jxx = Jxx, Jz = Jz, time = time, precision = precision, variance = variance, min_jump = min_jump, error_propagation_ratio = error_propagation_ratio )


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

    M = 30                       

    L_array = [45, 50]
    W_array = [40, 50]

    for L in L_array:
        for W in W_array:
            
            check_PR_dyn_N2(L, M, W)








