
#INPUT FOR FUNCTION DaC_eigen_N1

#Size (L), size of the system.
#Subsystem (M), size of the subsystem. M should be even and 2L/M should be an integer
#potential, array of L real numbers (potential at site i)
#hopping, array of L-1 real numbers (hopping between sites i, i+1)
#variance, variance of the eigenfunctions, as DEFAULT = 1e-32
#cutoff_overlap, if scalar product between eigenfunctions larger than cutoff_overlap, then they are considered as different eigenfunctions, as DEFAULT = 1e-7.
#cutoff_E, if eigenvalues between eigenfunctions are larger than cutoff_E, then they are considered as different eigenfunctions, as DEFAULT = 1e-7.


#Return E, the energies, PR, the Participation Ratio

from lib_N1 import DaC_eigen_N1, DaC_dyn_N1, energies_ED, PR_ED
import random
import numpy as np


L = 700
M = 200
W = 10
potential = np.random.uniform(-W, W, L)
hopping = np.ones(L-1)
shift = 0.4

E, PR, population = DaC_eigen_N1( system = L, subsystem = M, potential = potential, hopping = hopping, shift = shift )

N = len(E)

print("Obtained number of eigenstates with DaC:", N)


if( len(E) == L and L < 5000):
    energies = energies_ED( potential = potential, hopping = hopping)

    E = np.sort(E)

    print( "Maximum difference in the eigenvalues compared with ED:", np.amax( np.fabs(E - energies) ) )

if( len(E) == L ):
    print( "Infinity norm of (population - identity):", np.amax( np.fabs(population - np.ones(L)) ) )



#INPUT FOR FUNCTION DaC_eigen_N1

#Size (L), size of the system.
#Subsystem (M), size of the subsystem. M should be even
#potential, array of L numbers (potential at site i)
#Jxx, value of the hopping (it is the same in all sites)
#variance, variance of the eigenfunctions, as DEFAULT = 1e-32
#precision, upper bound of the error when calculating the Participation Ratio (PR)
#time, array with the time of interest, where the PR is calculated
#min_jump, minimum shift between subsystems.
#error_propagation_ratio, relates the maximum error amplitude wavefunction with the upper bound of the observable PR. DEFAULT  = 10.

L = 500
M = 200
W = 10
h = np.random.uniform(-W, W, L)
Jxx = 1
precision = 1e-4
time = np.arange(0, 10, 0.5)


variance = 1e-30
min_jump = 5
error_propagation_ratio = 10


PR, sites = DaC_dyn_N1( system = L, subsystem = M, potential = h, Jxx = Jxx, precision = precision, time = time, variance = variance, min_jump = min_jump, error_propagation_ratio = error_propagation_ratio )

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
