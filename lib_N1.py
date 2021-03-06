
import numpy as np

from scipy.linalg import eigh_tridiagonal


def energies_ED( potential, hopping ):

    E, V = eigh_tridiagonal(-potential, hopping)

    return E


def PR_ED( potential, hopping, time_interest ):

    E, V = eigh_tridiagonal(-potential, hopping)

    V = V.T

    PR_T_ED = np.zeros( (len(E), len(time_interest)) )

    for j in range(0, len(E)):
        PR_T_ED[j] = 1/np.sum ( np.abs( np.dot( (V[:, j ][:, np.newaxis] * V).T, np.exp( -np.outer(E, time_interest)*1j ) ) )**4, axis = 0 )

    return PR_T_ED




def compare_vec( v_old, E_old, v_new, E_new, M, last_old, last_new, tol_overlap, Delta_E ):

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

    (J0, J1, v) = x
    vari = J0*J0*v[:, 0]*v[:, 0] + J1*J1*v[:, -1]*v[:, -1]

    return vari



def DaC_eigen_N1( potential, hopping, system, subsystem, shift = 0, variance = 1e-32, cutoff_overlap = 1e-7, cutoff_E = 1e-7):

    h = potential
    J = hopping
    L = system
    M = subsystem

    J = np.concatenate( ([0], np.concatenate((J, [0]))) )

    if( shift == 0):
        jump = int(0.5*M)
    else:
        jump = int(shift*M)


    E_local = np.zeros( L )
    PR_local = np.zeros( L )
    popu_local = np.zeros( L )

    first = 0
    last = M

    hred_r = h[ first:last ]
    J_r = J[ first+1:last ]
    E, V = eigh_tridiagonal(-hred_r, J_r)
    V = V.T
    J0 = J[ 0 ]
    J1 = J[ M ]
    vari = check_variance_N1 ( [J0, J1, V ] )
    which = vari < variance

    E_old = E[which]
    v_old = V[which]

    popu_local[ first : last ] = popu_local[ first : last ] + np.sum( v_old**2, axis = 0 )
    E_local[ : len(E_old)] = E_old
    PR_local[ : len(E_old)] = 1/np.sum( v_old**4, axis = 1 )

    number = len(E_old)


    previous_eigen = v_old
    previous_E = E_old
    how_many = np.zeros(1, dtype = int) + number
    previous_interval = np.zeros(1, dtype = int) + last

    first = jump

    while ( first <= L - M ):

        last = first + M
        hred_r = h[ first:last ]

        J_r = J[ first+1:last ]
        E, V = eigh_tridiagonal(-hred_r, J_r)

        V = V.T
        J0 = J[ first ]
        J1 = J[ last ]
        vari = check_variance_N1 ( [J0, J1, V ] )
        which = vari < variance

        E_new = E[which]
        v_new = V[which]

        if( len(E_new) != 0 ):

            for i in range(0, len(previous_interval) ):
                previous = np.sum(how_many[:i])
                who = np.arange( previous, previous + how_many[i] )
                v_new, E_new = compare_vec( previous_eigen[who], previous_E[who], v_new, E_new, M, previous_interval[i], last, cutoff_overlap, cutoff_E )

                if(len(E_new) == 0):
                    break

        popu_local[ first : last ] = popu_local[ first : last ] + np.sum( v_new**2, axis = 0 )
        E_local[ number : number + len(E_new)] = E_new
        PR_local[ number : number + len(E_new)] = 1/np.sum( v_new**4, axis = 1 )

        number = number + len(E_new)

        if(first == L - M):
            break

        previous_eigen = np.concatenate( (previous_eigen, v_new), axis = 0)
        previous_E = np.concatenate( (previous_E, E_new) )
        how_many = np.concatenate( (how_many, [len(E_new)] ) )
        previous_interval = np.concatenate ( (previous_interval, [last]) )

        first = min( first + jump, L-M)

        #Eliminate previous eigenstates, if no overlap with next subsystem
        x = np.argmax( previous_interval > first )

        previous_eigen = previous_eigen[np.sum(how_many[:x]) : ]
        previous_E = previous_E[np.sum(how_many[:x]) : ]

        previous_interval = previous_interval[x:]
        how_many = how_many[x:]




    return E_local[:number], PR_local[:number], popu_local









def dynamics_site_first(x):

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


def DaC_dyn_N1( potential, Jxx, subsystem, system, precision, time, variance = 1e-32, min_jump = 1, error_propagation_ratio = 10 ):

    h = potential
    l0 = subsystem
    L = system
    epsilon = precision
    T = time

    if(l0 %2 != 0):
        print("ERROR\nSubsystem size must be even")
        return 0,[]

    l0 = int(l0*0.5)

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

        site_now = site_now + max(how_many, min_jump )

    real_site = real_site.astype(int)

    return PR, real_site
