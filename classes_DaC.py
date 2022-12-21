import numpy as np


class System_Parameters_Anderson():
    '''
    Parameters describing the system.
    size: Number of sites in the system (L)
    disorder: Disorder strength (W)
    hopping: Site dependent hopping (Jxx)
    potential: Site dependent potential (h)
    '''

    def __init__(self, size=0, disorder=0, hopping=0, potential=0):
        self.size = size
        self.disorder = disorder
        self.hopping = hopping
        self.potential = potential




class Technical_Parameters_Anderson():
    '''
    Parameters needed for the Divide-and-Conquer algorithm ("unphysical").
    subsystem: Number of sites in the subsystem.
    shift: Shift between consecutive subsystems.
    cutoff_variance: Cutoff for the variance of the eigenstates (accept only those with lower variance).
    cutoff_overlap: If the scalar product between eigenfunctions larger than cutoff_overlap, then they are considered as different eigenfunctions.
    cutoff_E: if eigenvalues between eigenfunctions are larger than cutoff_E, then they are considered as different eigenfunctions.
    '''


    def __init__(self, subsystem=0, shift = 0.5, cutoff_variance=1e-32, cutoff_overlap=1e-7, cutoff_E = 1e-7):
        self.subsystem = subsystem
        self.shift = shift
        self.cutoff_variance = cutoff_variance
        self.cutoff_overlap = cutoff_overlap
        self.cutoff_E = cutoff_E




#########################
#########################
#########################

#change
class System_Parameters_TIP():
    '''
    Parameters describing the system.
    size: Number of sites in the system (L)
    disorder: Disorder strength (W)
    hopping: Site dependent hopping (Jxx)
    potential: Site dependent potential (h)
    '''

    def __init__(self, size=0, disorder=0, hopping=0, potential=0):
        self.size = size
        self.disorder = disorder
        self.hopping = hopping
        self.potential = potential



#change
class Technical_Parameters_TIP():
    '''
    Parameters needed for the Divide-and-Conquer algorithm ("unphysical").
    subsystem: Number of sites in the subsystem.
    shift: Shift between consecutive subsystems.
    cutoff_variance: Cutoff for the variance of the eigenstates (accept only those with lower variance).
    cutoff_overlap: If the scalar product between eigenfunctions larger than cutoff_overlap, then they are considered as different eigenfunctions.
    cutoff_E: if eigenvalues between eigenfunctions are larger than cutoff_E, then they are considered as different eigenfunctions.
    '''


    def __init__(self, subsystem=0, shift = 0.5, cutoff_variance=1e-32, cutoff_overlap=1e-7, cutoff_E = 1e-7):
        self.subsystem = subsystem
        self.shift = shift
        self.cutoff_variance = cutoff_variance
        self.cutoff_overlap = cutoff_overlap
        self.cutoff_E = cutoff_E




class Observables_TIP_class():
    '''Class with the different quantities of interest, for the Two Interacting Particle (TIP) problem.
        meanDist: the mean distance between the 2 particles.
        flucCoM: fluctuations of the Center of Mass (CoM).
        PRDensity: Participation Ratio (PR) in real space.
        PRFock: Participation Ratio (PR) in Fock space.
        probTogether: Probability to find the 2 particles at consecutive sites.
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
