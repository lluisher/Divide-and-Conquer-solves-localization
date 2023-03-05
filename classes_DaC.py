'''
File with useful classes for the Anderson and TIP problem.
'''

import numpy as np


class System_Parameters_Anderson():
    '''
    Parameters describing the system (Anderson problem).\n
    * **system**: Number of sites in the system (L).\n
    * **disorder**: Disorder strength (W).\n
    * **hopping_strength**: Hopping (Jxx).\n
    * **hopping_anisotropy**: Amount of anisotropy (Delta_Jxx).\n
    * **potential**: Site dependent potential (h).\n
    * **hopping_dist**: Site dependent hopping (No hopping between first and last site)\n
    '''

    def __init__(self, system, disorder, potential, hopping_dist=[0], hopping_strength = 1, hopping_anisotropy=0):
        self.system = system
        self.disorder = disorder
        self.potential = potential
        self.hopping_strength = hopping_strength
        self.hopping_anisotropy = hopping_anisotropy
        if(len(hopping_dist) == system - 1):
            self.hopping_dist = np.concatenate( ([0], np.concatenate((hopping_dist, [0]))) )
        else:
            self.hopping_dist = hopping_dist




class Technical_Parameters_Eigen():
    '''
    Parameters needed for the Divide-and-Conquer algorithm ("unphysical" parameters).
    Valid for both Anderson and TIP problem.\n
    * **subsystem**: Number of sites in the subsystem.\n
    * **shift**: Shift between consecutive subsystems.\n
    * **cutoff_variance**: Cutoff for the variance of the eigenstates (accept only those with lower variance).\n
    * **cutoff_overlap**: If the scalar product between eigenfunctions larger than cutoff_overlap, then they are considered as different eigenfunctions.\n
    * **cutoff_E**: if eigenvalues between eigenfunctions are larger than cutoff_E, then they are considered as different eigenfunctions.\n
    '''


    def __init__(self, subsystem, shift = 0.5, cutoff_variance=1e-32, cutoff_overlap=1e-7, cutoff_E = 1e-7):
        self.subsystem = subsystem
        self.shift = shift
        self.cutoff_variance = cutoff_variance
        self.cutoff_overlap = cutoff_overlap
        self.cutoff_E = cutoff_E




#time, array with the time of interest, where the PR is calculated


class Technical_Parameters_Dyn():
    '''
    Parameters needed for the Divide-and-Conquer algorithm ("unphysical" parameters),
    when interested in the dynamics of both the Anderson and TIP problem.\n
    * **subsystem**: Number of sites in the subsystem (subtract one if the number is odd).\n
    * **min_jump**: Minimum jump between consecutive subsystems.\n
    * **precision**: Upper bound of the error when calculating the Participation Ratio (PR).\n
    * **cutoff_variance**: Cutoff for the variance of the eigenstates (accept only those with lower variance).\n
    * **error_propagation**: Relates the maximum error amplitude wavefunction with the upper bound of the observable PR. It is observable dependent.\n
    * **reduce_memory**: Boolean, how to calculate the dynamics. If True, using for loop (Numba), otherwise, via matrix multiplication (default).
    '''


    def __init__(self, subsystem, min_jump = 1, precision = 1e-4, cutoff_variance=1e-32, error_propagation=10, reduce_memory=False):
        self.subsystem = subsystem
        self.min_jump = min_jump
        self.precision = precision
        self.cutoff_variance = cutoff_variance
        self.error_propagation = error_propagation
        self.reduce_memory = reduce_memory



class System_Parameters_TIP():
    '''
    Parameters describing the system.\n
    * **system**: Number of sites in the system (L)\n
    * **disorder**: Disorder strength (W)\n
    * **hopping**: Constant hopping (Jxx)\n
    * **interaction**: Interaction strength (Jz)\n
    * **potential**: Site dependent potential (h)
    '''

    def __init__(self, system=0, disorder=0, hopping=1, interaction = 1, potential=0):
        self.system = system
        self.disorder = disorder
        self.hopping = hopping
        self.interaction = interaction
        self.potential = potential


class Observables_TIP_class():
    '''Class with the different quantities of interest, for the Two Interacting Particle (TIP) problem.\n
        * **meanDist**: the mean distance between the 2 particles.\n
        * **flucCoM**: fluctuations of the Center of Mass (CoM).\n
        * **PRDensity**: Participation Ratio (PR) in real space.\n
        * **PRFock**: Participation Ratio (PR) in Fock space.\n
        * **probTogether**: Probability to find the 2 particles at consecutive sites.
    '''

    def __init__(self, meanDist=np.zeros(0), flucCoM=np.zeros(0), PRDensity=np.zeros(0),
                PRFock=np.zeros(0), probTogether=np.zeros(0)):
        self.meanDist = meanDist
        self.flucCoM = flucCoM
        self.PRDensity = PRDensity
        self.PRFock = PRFock
        self.probTogether = probTogether

    #Override + symbol, to concatenate two of the objects
    def __add__(self, other):
        new_dist = np.concatenate( (self.meanDist, other.meanDist) )
        new_CoM = np.concatenate( (self.flucCoM, other.flucCoM) )
        new_PR_Density = np.concatenate( (self.PRDensity, other.PRDensity) )
        new_PR_Fock = np.concatenate( (self.PRFock, other.PRFock) )
        new_prob = np.concatenate( (self.probTogether, other.probTogether) )

        return Observables_TIP_class( new_dist, new_CoM, new_PR_Density, new_PR_Fock, new_prob )
