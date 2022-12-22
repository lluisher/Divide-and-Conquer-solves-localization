'''
Module with the functions to implement the DaC for the Anderson problem.
'''
import numpy as np

from scipy.linalg import eigh_tridiagonal

from classes_DaC import System_Parameters_Anderson, Technical_Parameters_Anderson_Eigen


def energies_ED( potential, hopping ):
    '''
    Calculate the eigenvalues (energies) of the Anderson model (tridiagonal matrix), using the
    function ***eigh_tridiagonal()*** defined in scipy.\n
    * **potential**: Values in the diagonal (chemical potential or magnetic field).\n
    * **hopping**: Values of the off-diagonal (hopping). They might be site-dependent.\n
    It returns an array with the energies.
    '''
    E, V = eigh_tridiagonal(-potential, hopping)

    return E


def PR_ED( potential, hopping, time_interest ):
    '''
    Calculate the PR using ED, for the Anderson model (tridiagonal matrix).\n
    * **potential**: Values in the diagonal (chemical potential or magnetic field).\n
    * **hopping**: Values of the off-diagonal (hopping). They might be site-dependent
    (but note that at least one eigenstate will be fully delocalized).\n
    * **time_interest**: For which time calculate the time evolution.\n
    It returns a 2D numpy array. Each row represent a different initial state,
    each column a different value of time.
    '''
    E, V = eigh_tridiagonal(-potential, hopping)

    V = V.T

    PR_T_ED = np.zeros( (len(E), len(time_interest)) )

    for j in range(0, len(E)):
        PR_T_ED[j] = 1/np.sum ( np.abs( np.dot( (V[:, j ][:, np.newaxis] * V).T, np.exp( -np.outer(E, time_interest)*1j ) ) )**4, axis = 0 )

    return PR_T_ED




def compare_vec( v_old, E_old, v_new, E_new, M, last_old, last_new, tol_overlap, Delta_E ):
    '''
    Function to check which new eigenstates (v_new) have been already obtained (in v_old).
    '''


    dif_E = np.fabs(np.tensordot(E_old, np.ones(len(E_new)), axes = 0) - np.tensordot(np.ones(len(E_old)), E_new, axes = 0)) < Delta_E

    sure_old = np.sum( dif_E, axis = 1) == 0
    sure_new = np.sum( dif_E, axis = 0) == 0

    dif_E = 0

    store_v_new = v_new[sure_new]
    store_E_new = E_new[sure_new]

    v_old_local = v_old[np.logical_not(sure_old)]
    v_new = v_new[np.logical_not(sure_new)]
    E_new = E_new[np.logical_not(sure_new)]

    real = np.sum( np.fabs( np.dot( v_new[ :, : last_old - (last_new - M) ], np.transpose( v_old_local[ :, -(last_old - (last_new - M) ): ] ) ) ) > tol_overlap, axis = 1 ) == 0

    v_new = v_new[real]
    E_new = E_new[real]

    if(np.sum(sure_new) != 0):

        v_new = np.concatenate( (v_new, store_v_new), axis = 0 )
        E_new = np.concatenate( (E_new, store_E_new) )

    return v_new, E_new



def check_variance_N1 (x):
    '''
    Check the variance of the eigenstates of the subsystem.
    Since we consider only nearest-neighbor hoppings,
    it is basically the population at the edge of the subsystem.
    '''

    (J0, J1, v) = x
    vari = J0*J0*v[:, 0]*v[:, 0] + J1*J1*v[:, -1]*v[:, -1]

    return vari


def obtain_eigen_subsystem(first_site, last_site, h, J, cutoff_variance):
    '''
    Calculate the true eigenvalues (energies) and eigenstates of a system
    which are localized in the given subsystem.
    '''

    hred_r = h[ first_site:last_site ]
    J_r = J[ first_site+1:last_site ]
    E, V = eigh_tridiagonal(-hred_r, J_r)
    V = V.T
    J0 = J[ first_site ]
    J1 = J[ last_site ]
    vari = check_variance_N1 ( [J0, J1, V ] )
    which = vari < cutoff_variance

    E_subsystem = E[which]
    v_subsystem = V[which]

    return E_subsystem, v_subsystem



def add_new_data_E_PR_Population_Number( popu, E, PR, number, first, last, E_new, v_new ):
    '''
    Bookeepping the obtained energies, PR, and population.
    '''

    popu[ first : last ] = popu[ first : last ] + np.sum( v_new**2, axis = 0 )
    E[ number : number + len(E_new)] = E_new
    PR[ number : number + len(E_new)] = 1/np.sum( v_new**4, axis = 1 )

    number = number + len(E_new)

    return popu, E, PR, number


def eliminate_repetitions(how_many, previous_eigen, previous_E, v_new, E_new,
                        M, previous_interval, last, cutoff_overlap, cutoff_E, i):

    '''
    Delete unneeded eigenstates (no overlap between the subsystem
    in which they are localized and the actual subsystem)
    '''

    previous = np.sum(how_many[:i])
    who = np.arange( previous, previous + how_many[i] )
    v_new, E_new = compare_vec( previous_eigen[who], previous_E[who], v_new,
            E_new, M, previous_interval[i], last, cutoff_overlap, cutoff_E )

    return v_new, E_new


def update_bookkepping(v_new, E_new, last, previous_eigen = [], previous_E = [], how_many = np.zeros(1, dtype = int), previous_interval = np.zeros(1, dtype = int) ):
    '''
    Bookeep the relevant eigenstates (how many, their eigenvalues and interval)
    '''
    if(len(previous_eigen) == 0):
        previous_eigen = v_new
    else:
        previous_eigen = np.concatenate( (previous_eigen, v_new), axis = 0)

    previous_E = np.concatenate( (previous_E, E_new) )
    how_many = np.concatenate( (how_many, [len(E_new)] ) )
    previous_interval = np.concatenate ( (previous_interval, [last]) )

    return previous_eigen, previous_E, how_many, previous_interval




def DaC_eigen_N1( parameters_system, parameters_technical):
    '''
    Main function to obtain localized eigenstates of the Anderson model.\n

    * **parameters_system**: Object of the class **System_Parameters_Anderson**,
    containing physical parameters to describe the system.\n
    * **parameters_technical**: Object of the class **Technical_Parameters_Anderson_Eigen**,
    containing several cutoffs and parameters needed in the DaC algorithm.\n
    It returns the set of obtained energies, PR and population in each site
    (to check that no overpopulation or missing population)
    '''
    h = parameters_system.potential
    J = parameters_system.hopping_dist
    L = parameters_system.system
    M = parameters_technical.subsystem
    shift = parameters_technical.shift
    jump = int(shift*M)


    cutoff_variance = parameters_technical.cutoff_variance
    cutoff_overlap = parameters_technical.cutoff_overlap
    cutoff_E = parameters_technical.cutoff_E

    E_local = np.zeros( L )
    PR_local = np.zeros( L )
    popu_local = np.zeros( L )

    first = 0
    last = M

    E_old, v_old = obtain_eigen_subsystem(first, last, h, J, cutoff_variance)

    number = 0

    popu_local, E_local, PR_local, number = add_new_data_E_PR_Population_Number(
                    popu_local, E_local, PR_local, number, first, last, E_old, v_old)


    previous_eigen, previous_E, how_many, previous_interval = update_bookkepping( v_old, E_old, last)

    first = jump

    while ( first <= L - M ):

        last = first + M

        E_new, v_new = obtain_eigen_subsystem(first, last, h, J, cutoff_variance)

        if( len(E_new) != 0 ):

            for i in range(0, len(previous_interval) ):
                v_new, E_new = eliminate_repetitions(how_many, previous_eigen,
                                    previous_E, v_new, E_new, M, previous_interval,
                                    last, cutoff_overlap, cutoff_E, i)

                if(len(E_new) == 0):
                    break

        popu_local, E_local, PR_local, number = add_new_data_E_PR_Population_Number(
                        popu_local, E_local, PR_local, number, first, last, E_new, v_new)


        if(first == L - M):
            break

        previous_eigen, previous_E, how_many, previous_interval = update_bookkepping(
                                v_new, E_new, last, previous_eigen, previous_E,
                                how_many, previous_interval )


        first = min( first + jump, L-M)

        #Eliminate previous eigenstates, if no overlap with next subsystem
        x = np.argmax( previous_interval > first )

        previous_eigen = previous_eigen[np.sum(how_many[:x]) : ]
        previous_E = previous_E[np.sum(how_many[:x]) : ]

        previous_interval = previous_interval[x:]
        how_many = how_many[x:]


    return E_local[:number], PR_local[:number], popu_local









def dynamics_site_first(x):
    '''
    Calculate the time evolution of several initial states (localized in one site),
    for all the times of interest.
    '''
    (delta, h, Jxx, J0, J1, max_va, T, epsilon, error_propagation_ratio) = x

    l0 = len(h)

    error_ampl = epsilon/( error_propagation_ratio*l0 )

    E, V = eigh_tridiagonal(-h, np.zeros(len(h)-1) + Jxx)

    V = V.T

    vari = check_variance_N1 ( [J0, J1, V ] )

    which = vari < max_va

    number = np.sum(which)

    new_v = V[which]
    new_E = E[which]

    which_complete = np.sum ( new_v[:, delta:]**2, axis = 0 ) > (1 - error_ampl)**2

    if(which_complete[0]):

        if( np.sum(which_complete) != len(which_complete) ):
            how_many = np.arange(len(which_complete))[np.logical_not(which_complete)][0]

        else:
            how_many = np.sum(which_complete)

        PR_T = np.zeros( (how_many, len(T)) )

        for j in range(0, how_many):
            PR_T[j] = 1/np.sum ( np.abs( np.dot( (new_v[:, delta+j ][:, np.newaxis] * new_v).T, np.exp( -np.outer(new_E, T)*1j ) ) )**4, axis = 0 )


    else:
        PR_T = []
        how_many = 0

    return PR_T, how_many



def DaC_dyn_N1( parameters_system, time, parameters_technical ):
    '''
    Main function to calculate the dynamics in the Anderson model.\n
    Note that we only consider constant hopping
    (otherwise, delocalized states exists).\n
    * **parameters_system**: Object of the class **System_Parameters_Anderson**,
    containing physical parameters to describe the system.\n
    * **time**: Array with the times of interest.
    * **parameters_technical**: Object of the class **Technical_Parameters_Anderson_Dyn**,
    containing several cutoffs and parameters needed in the DaC algorithm.\n
    It returns a 2D numpy array with the values of the PR for the different initial
    states (rows), at each time of interest (columns), together with the
    corresponding initial site .
    '''
    L = parameters_system.system
    h = parameters_system.potential
    Jxx = parameters_system.hopping_strength

    T = time

    M = parameters_technical.subsystem
    epsilon = parameters_technical.precision
    variance = parameters_technical.cutoff_variance
    min_jump = parameters_technical.min_jump
    error_propagation_ratio = parameters_technical.error_propagation_ratio

    l0 = int(M*0.5)

    PR = []
    real_site = []
    site_now = 0

    while( site_now < L ):

        if( site_now <= l0 ):
            h_local = h[ :2*l0 ]
            J0 = 0
            J1 = Jxx
            first_dyn = site_now

        elif( L - site_now <= l0):
            h_local = h[ -2*l0 : ]
            J0 = Jxx
            J1 = 0
            first_dyn = 2*l0-(L - site_now)

        else:
            h_local = h[ site_now -l0 : site_now + l0 ]
            J0 = Jxx
            J1 = Jxx
            first_dyn = l0

        PR_local, how_many = dynamics_site_first([first_dyn, h_local, Jxx, J0, J1, variance, T, epsilon, error_propagation_ratio])

        if(how_many != 0):
            if(len(PR) == 0):
                PR = PR_local
                real_site = np.arange(how_many) + site_now

            else:
                PR = np.concatenate( (PR, PR_local) )
                real_site = np.concatenate( (real_site, np.arange(how_many) + site_now) )

        site_now = site_now + max(how_many, min_jump)

    real_site = real_site.astype(int)

    return PR, real_site
