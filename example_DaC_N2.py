#INPUT FOR FUNCTION DaC_eigen_N2

#system (L), size of the system.
#subsystem (M), size of the subsystem. M should be even
#potential, array of L numbers (potential at site i)
#Jxx, hopping coefficient
#Jz, interaction strength

#variance, variance of the eigenfunctions, as DEFAULT = 1e-32
#cutoff_overlap, if scalar product between eigenfunctions larger than cutoff_overlap, then they are considered as different eigenfunctions, as DEFAULT = 1e-7.
#cutoff_E, if eigenvalues between eigenfunctions are larger than cutoff_E, then they are considered as different eigenfunctions, as DEFAULT = 1e-7.
#min_jump, the shift between consecutive subsystems. DEFAULT is half the subsystem size.


#Return energies (E), several observables (Obs, in README are specified which one are), the population in consecutive sites (popu) and in which site the eigenstates obtained start (begin_site)

from lib_N2 import DaC_eigen_N2, DaC_N2_dyn
import random
import numpy as np

L = 100
M = 50
W = 10
potential = np.random.uniform(-W, W, L)
Jxx = 1
Jz = 1


variance = 1e-16
cutoff_overlap = 1e-7
cutoff_E = 1e-7
min_jump = 25

E, Obs, popu, begin_site = DaC_eigen_N2( system = L, subsystem = M, potential = potential, Jxx = Jxx, Jz = Jz, variance = variance, cutoff_overlap = cutoff_overlap, cutoff_E = cutoff_E, min_jump = min_jump )


#INPUT FOR FUNCTION DaC_N2_dyn
#system (L), size of the system.
#subsystem (M), size of the subsystem. M should be even
#potential, array of L numbers (potential at site i)
#Jxx, hopping coefficient
#Jz, interaction strength
#time, array with the time of interest
#precision, upper bound for the error when calculating the time evolution
#variance, variance of the eigenfunctions, as DEFAULT = 1e-32.
#min_jump, the shift between consecutive subsystems. DEFAULT is 1.
#error_propagation_ratio, constant to relate how error propagates from eigenfunctions to the observable. DEFAULT = 10

#Return PR (a 2D-array with the Participation Ratio for the several times of interest, where the 0-axis refers to the initial states (maximum L-1) and the 1-axis refers to the instance of the time ) and real_sites, an array specifying which are the initial states of the wavefunctions solved.


L = 200
M = 50
W = 20
potential = np.random.uniform(-W, W, L)
Jxx = 1
Jz = 1
time = np.linspace(0, 10, 50)
precision = 1e-5

variance = 1e-16
min_jump = 1
error_propagation_ratio = 10


PR, real_sites = DaC_N2_dyn( system = L, subsystem = M, potential = potential, Jxx = Jxx, Jz = Jz, time = time, precision = precision, variance = variance, min_jump = min_jump, error_propagation_ratio = error_propagation_ratio )



print("The number of initial wavefunctions whose time evolution can be calculated, with the required precision of 10^(%5.1f) and subsystem size %5.0f is %5.0f. The total number of initial states is %5.0f"%(np.log10(precision),M,len(real_sites),L-1) )
