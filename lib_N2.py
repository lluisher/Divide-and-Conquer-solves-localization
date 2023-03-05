'''
Module with the functions to implement the DaC for
the Two Interacting Particles (TIP) problem.
'''


import numpy as np

from scipy.linalg import eig_banded

from itertools import combinations

from numba import jit

from classes_DaC import Observables_TIP_class


def create_connections(M, M2):
    '''
    Determine which nodes are connected by the hopping, connects site (i,j)
    to sites (i+-1, j) and (i, j+-1)
    '''
    Degree = np.ones( M2, dtype = int )

    Edges = []

    comb = np.asarray( list( combinations(np.arange(M), 2) ) )
    magnitude = comb[:, 0]*M + comb[:, 1]

    for i in range(0, M2):

        dif = np.fabs( magnitude[i] - magnitude )

        need_add = np.arange(M2) [ np.logical_or(dif == 1, dif == M) ]

        Edges.append( need_add )

        Degree[i] = len(Edges[-1])

    return Edges, Degree


def order_RCM(M):
    '''
    Implement the Reverse Cuthill-McHill (RCM) algorithm, to reduce the bandwidth
    of the matrix.
    '''
    M2 = int( M*(M-1)*0.5)

    Vertex = np.arange(M2, dtype = int)

    Edges, Degree = create_connections(M, M2)

    R = np.zeros(M2, dtype = int)

    how = np.zeros(M2, dtype = int)

    how[0] = M2 - 1

    Q = np.array( [1], dtype = int )

    missing = np.ones(M2, dtype = bool)

    missing[ [0, 1] ] = False

    counter = 1

    while( counter != M2 ):

        R[counter] = Q[0]

        how[ Q[0] ] = M2 - 1 - counter

        extra = Edges[ Q[0] ] [missing[ Edges[ Q[0] ] ] ]

        extra = np.flip ( extra [ np.argsort( Degree[extra] ) ] )

        Q = np.append( Q,  extra )

        missing [ Edges[ Q[0] ] ] = False

        Q = Q[1:]

        counter = counter + 1

    permutation = np.flip(R)

    Vertex_new = permutation

    Edges_new = []

    for i in range(0, len(Edges)):

        who = Edges[Vertex_new[i]]

        real = how[who]

        Edges_new.append (real)


    new_band_width = 0
    entries = np.zeros( M2, dtype = int)

    for i in range(0, M2):

        new_band_width = max( new_band_width, np.amax( np.fabs(Edges_new[i] - i ) ) )



    band_width = int(new_band_width)

    hopping = np.zeros( (band_width, M2) )

    #If Edges - i is k, put 1 on the k-th row, i-th column

    for j in range(0, M2):
        entries = Edges_new[j] - j

        entries = entries[entries > 0]

        hopping[ entries - 1, j ] = 1


    return Vertex_new, np.argsort(Vertex_new), hopping, band_width



def find_auxi(Lred):
    '''
    Calculate which entries represent 1 particle in the left edge of the
    subsystem (auxi_left), the right end (auxi_right),
    and where the 2 particles are next to each other (where_together)
    '''
    L2red = int(Lred*(Lred-1)*0.5)

    auxi_left = np.arange(0, Lred - 1)

    where_together = np.zeros(Lred-1, dtype = int)

    where_together[1:] = np.cumsum( np.arange(Lred-1, 1, -1) )

    auxi_right = np.zeros(Lred-1, dtype = int)

    auxi_right[:-1] = where_together[1:] - 1

    auxi_right[-1] = where_together[-1]



    return auxi_left, auxi_right, where_together



def check_variance_N2 (x):
    '''
    Check variance of a possible eigen.
    '''

    (J1, J2, v, auxi_left, auxi_right) = x

    vari = J2*J2* ( np.sum( v[:, auxi_right]**2, axis = 1 ) ) + J1*J1*( np.sum( v[:, auxi_left]**2, axis = 1 ) )

    return vari


def generate_how_compare(start_left, start_right, end_left, end_right):
    '''
    Used when comparing eigenfunctions from different subset,
    give us the common entries. Assume that the subsystem have different sizes.
    '''
    M = end_left - start_left + 1

    l2 = int( M*(M-1)*0.5 )

    M_tilde = end_left - start_right + 1

    offset = int( ( M*(M-1) - M_tilde*(M_tilde-1) )*0.5 )

    which_old = np.arange( offset, l2, dtype = int )

    which_new = np.zeros( l2 - offset, dtype = int)

    counter = end_left - start_right

    which_new[0:counter] = np.arange(0, counter)

    jump = end_right - end_left + 1

    for i in range(start_right + 1, end_left):

        which_new[ counter : counter + end_left - i ] = which_new[counter - 1] + jump + np.arange( 0, end_left - i )
        counter = counter + end_left - i

    return which_old, which_new


def compare(E_left, v_left, E_center, v_center, which_old, which_new, tol_overlap, Delta_E ):
    '''
    Compare if eigenfunctions equal or not
    '''
    dif_E = np.fabs(np.tensordot(E_left, np.ones(len(E_center)), axes = 0) - np.tensordot(np.ones(len(E_left)), E_center, axes = 0)) < Delta_E

    sure_old = np.sum( dif_E, axis = 1) == 0
    sure_new = np.sum( dif_E, axis = 0) == 0

    dif_E = 0

    store_v_new = v_center[sure_new]
    store_E_new = E_center[sure_new]

    maybe_old = np.logical_not(sure_old)
    maybe_new = np.logical_not(sure_new)

    v_old = v_left[maybe_old]
    E_old = E_left[maybe_old]

    #Change name
    E_center = E_center[maybe_new]
    v_center = v_center[maybe_new]

    if(len(E_center) != 0 and len(E_old) != 0 ):

        real = np.sum( np.fabs (np.dot( v_center[:, which_new], np.transpose( v_old[:, which_old] ) ) ) > tol_overlap, axis = 1 ) == 0

        v_center = v_center[ real ]
        E_center = E_center[ real ]

    if(np.sum(sure_new) != 0):

        v_center = np.concatenate( (v_center, store_v_new), axis = 0 )
        E_center = np.concatenate( (E_center, store_E_new) )

    return E_center, v_center







def banded_H(h, Jxx, Jz, m, permutation, hopping, width ):
    '''
    Create banded matrix.
    '''

    L = int( m*(m-1)*0.5)

    H = np.zeros( (width + 1, L) )

    previous = 0

    for i in range(0, m-1):

        H[0][previous + np.arange(m - i - 1)] = -h[i] - h[i+1:]

        H[0][previous] = H[0][previous] + Jz

        previous = previous + m - i - 1

    H[0] = H[0][permutation]

    H[1:, :] = Jxx*hopping

    return H


def E_v_subset(x):
    '''
    Give back the eigenvalues (E) and eigenfunction itself (v).
    '''

    (hred_r, Jxx, J_hops, Jz, sub_size, permutation, inv_permutation, hopping, width, auxi_left, auxi_right, where_together, variance) = x

    H_correct = banded_H(hred_r, Jxx, Jz, sub_size, permutation, hopping, width )

    epsired_r, vred_r = eig_banded( H_correct, lower = True, overwrite_a_band = True, check_finite = False )

    M2 = len(epsired_r)

    vred1_r = np.transpose( vred_r[inv_permutation] )

    J0 = J_hops[ 0 ]
    J1 = J_hops[ -1 ]

    vari = check_variance_N2 ( [J0, J1, vred1_r, auxi_left, auxi_right ] )

    which = vari < variance

    number = np.sum(which)

    E = epsired_r[which]

    v = vred1_r[which]

    if(number == 0):
        return np.zeros(0), np.zeros((0, M2))

    return E, v



def DaC_eigen_N2( parameters_system, parameters_technical):

    '''
    Main function to obtain localized eigenstates of the TIP problem.\n

    * **parameters_system**: Object of the class **System_Parameters_TIP**,
    containing physical parameters to describe the system.\n
    * **parameters_technical**: Object of the class **Technical_Parameters_Eigen**,
    containing several cutoffs and parameters needed in the DaC algorithm.\n
    It returns the set of obtained energies, the Observables of interest
    (as an object of the class **Observables_TIP_class**),
    the population in each consecutive site (i,i+1) and in which interval
    each of the obtained eigenstate is localized.
    '''



    h = parameters_system.potential
    L = parameters_system.system
    Jxx = parameters_system.hopping
    Jz = parameters_system.interaction
    M_sub = parameters_technical.subsystem
    variance = parameters_technical.cutoff_variance
    tol_overlap = parameters_technical.cutoff_overlap
    Delta_E = parameters_technical.cutoff_E
    min_jump = int(parameters_technical.shift * M_sub)

    popu = np.zeros(L - 1)

    dim_sub = int(M_sub*(M_sub - 1)*0.5)

    #First subsystem

    Vertex_new, inv_Vertex_new, hopping, band_width = order_RCM( M_sub )

    auxi_left, auxi_right, where_together = find_auxi( M_sub )

    #Call functions to determine observables, doing twice prob together
    parameters = fun_parameters(M_sub)

    E_local = np.zeros(0)
    begin_local = np.zeros(0, dtype = int)
    #Create object, with no data in it
    Observables = Observables_TIP_class()

    first_site_r = 0

    last_site_r = first_site_r + M_sub - 1

    hred_r = h[ first_site_r:last_site_r + 1 ]

    J_r = np.array([0, Jxx])

    E_new, v_new = E_v_subset( [hred_r, Jxx, J_r, Jz, M_sub, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, variance] )

    popu[ first_site_r : last_site_r ] = popu[ first_site_r : last_site_r ] + np.sum( v_new[:, where_together]**2, axis = 0 )

    Observables = Observables + fun_give_back_observables(v_new, parameters, len(E_new))

    E_local = np.concatenate( (E_local, E_new) )
    begin_local = np.concatenate( (begin_local, np.zeros(len(E_new), dtype = int) + first_site_r )  )

    #Store
    previous_eigen = v_new
    previous_E = E_new
    start_previous_interval = np.zeros(1, dtype = int) + first_site_r
    end_previous_interval = np.zeros(1, dtype = int) + last_site_r

    how_many = np.zeros(1, dtype = int) + len(E_new)

    first_site_r = min ( first_site_r + min_jump, L - M_sub )


    while( first_site_r <= L - M_sub):

        last_site_r = first_site_r + M_sub - 1

        hred_r = h[ first_site_r:last_site_r + 1 ]

        J_r = np.array([Jxx, Jxx])

        if ( last_site_r == L - 1 ):
            J_r[1] = 0

        E_new, v_new = E_v_subset( [hred_r, Jxx, J_r, Jz, M_sub, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, variance] )

        #Get rid of possible repetitions
        if(len(E_new) != 0):
            for j in range(0, len(start_previous_interval) ):
                previous = np.sum(how_many[:j])
                who = np.arange( previous, previous + how_many[j] )
                if(how_many[j]):
                    which_old, which_new = generate_how_compare( start_previous_interval[j], first_site_r, end_previous_interval[j], last_site_r )
                    E_new, v_new = compare( previous_E[who], previous_eigen[who], E_new, v_new, which_old, which_new, tol_overlap, Delta_E)

                if(len(E_new) == 0):
                    break

        if(len(E_new) != 0):
            popu[ first_site_r : last_site_r ] = popu[ first_site_r : last_site_r ] + np.sum( v_new[:, where_together]**2, axis = 0 )

            #Add new entries to the Object Observables
            Observables =  Observables + fun_give_back_observables(v_new, parameters, len(E_new))

            E_local = np.concatenate( (E_local, E_new) )
            begin_local = np.concatenate( (begin_local, np.zeros(len(E_new), dtype = int) + first_site_r )  )

        if(first_site_r == L - M_sub):
            break
        #Store previous eigen
        previous_eigen = np.concatenate( (previous_eigen, v_new), axis = 0)
        previous_E = np.concatenate( (previous_E, E_new) )
        how_many = np.concatenate( (how_many, [len(E_new)] ) )
        start_previous_interval = np.concatenate ( (start_previous_interval, [first_site_r]) )
        end_previous_interval = np.concatenate ( (end_previous_interval, [last_site_r]) )


        first_site_r = min ( first_site_r + min_jump, L - M_sub )


        #Eliminate previous eigen
        x = np.argmax( end_previous_interval - 1 > first_site_r )           #Eliminate also if only 1 site overlap
        previous_eigen = previous_eigen[np.sum(how_many[:x]) : ]
        previous_E = previous_E[np.sum(how_many[:x]) : ]

        start_previous_interval = start_previous_interval[x:]
        end_previous_interval = end_previous_interval[x:]
        how_many = how_many[x:]

    return E_local, Observables, popu, begin_local



def time_evolution(A, new_E, T):
    '''
    Calculates the PR of the initial states (rows of A)
    as a function of time (T).\n
    Memory consuming but faster.
    '''

    evol_matrix = np.e**( -1j * np.tensordot(new_E, T, axes = 0) )
    return np.dot(A, evol_matrix)




@jit(nopython=True)
def time_evolution_loop(A, new_E, T):
    '''
    Calculates the PR of the initial states (rows of A)
    as a function of time (T).\n
    It uses numba, to speed up the for loop.\n
    We do not do a matrix multiplication to save memory.
    '''

    N_rows = np.shape(A)[0]
    N_internal = len(new_E)
    N_columns = len(T)

    final_r = np.zeros( (N_rows, N_columns) )
    final_c = np.zeros( (N_rows, N_columns) )

    for i in range(N_rows):
        for j in range(N_columns):

            r_aux = 0
            c_aux = 0

            for k in range(N_internal):

                r_aux = r_aux + A[i,k]*np.cos(new_E[k]*T[j])
                c_aux = c_aux - A[i,k]*np.sin(new_E[k]*T[j])

            final_r[i,j] = r_aux
            final_c[i,j] = c_aux

    return final_r, final_c



def dynamics_site_first(x):
    '''
    Calculate the dynamics of the states in a given subsystem (interval).
    '''

    (delta, end, h, Jxx, Jz, J1, J2, permutation, inv_permutation, hopping, width, auxi_left, auxi_right, where_together, max_va, T, epsilon, all_sites, error_propagation, reduce_memory) = x

    l0 = len(h)

    error_ampl = epsilon/(error_propagation*l0*3)

    H_correct = banded_H(h, Jxx, Jz, l0, permutation, hopping, width )

    epsired_r, vred_r = eig_banded( H_correct, lower = True, overwrite_a_band = True, check_finite = False )

    vred1_r = np.transpose( vred_r[inv_permutation] )

    vari = check_variance_N2 ( [J1, J2, vred1_r, auxi_left, auxi_right ] )

    which = vari < max_va

    number = np.sum(which)

    new_E = epsired_r[which]

    new_v = vred1_r[which]

    which_complete = np.sum ( new_v[ :, where_together[delta:end] ]**2, axis = 0 ) > (1 - error_ampl)**2

    if(which_complete[0]):

        if( np.sum(which_complete) != len(which_complete) ):
            how_many = np.arange(len(which_complete))[np.logical_not(which_complete)][0]

        else:
            how_many = np.sum(which_complete)

        maybe_big = new_v[ :, where_together[delta:delta+how_many] ]**2 > (error_ampl**2)/number

        new_v = new_v[ np.sum(maybe_big, axis = 1) != 0 ]
        new_E = new_E[ np.sum(maybe_big, axis = 1) != 0 ]
        number = np.sum(len(new_E))

        maybe_big = maybe_big[ np.sum(maybe_big, axis = 1) != 0 ]

        #phase = np.exp( -np.outer(new_E, T)*1j )
        PR_T = np.zeros( (how_many, len(T)) )

        for j in range(0, how_many):

            maybe = maybe_big[:, j]

            local_store = new_v[maybe, where_together[delta+j] ][:, np.newaxis] * new_v[maybe]

            relevant_sites = np.sum( local_store**2, axis = 0 ) > error_ampl**2

            local_store = local_store[:, relevant_sites]

            relevant_eigen = np.sum( local_store**2  > (error_ampl**2)/(np.sum(maybe)), axis = 1 ) > 0

            maybe[maybe] = relevant_eigen

            local_store = local_store[relevant_eigen].T

            if(reduce_memory):
                final_r, final_c = time_evolution_loop(local_store, new_E[maybe], T )
                final = final_r + final_c*1j
                final_r = 0
                final_c = 0
                local_store = 0
                PR_T[j] = cal_PR_density(final, all_sites, relevant_sites)
            else:
                final = time_evolution(local_store, new_E[maybe], T )
                local_store = 0
                PR_T[j] = cal_PR_density(final, all_sites, relevant_sites)

    else:
        PR_T = []
        how_many = 0

    return PR_T, how_many





def cal_PR_density(psi, all_sites, relevant_sites):
    '''
    Calculate the PR in real space.
    '''
    start = all_sites[relevant_sites, 0]
    end = all_sites[relevant_sites, 1]

    L = end.max() - start.min() + 1

    end = end - start.min()
    start = start - start.min()

    density = np.zeros( ( L, len(psi[0]) ) )

    for i in range(0, L):
        how = np.zeros( np.sum(relevant_sites) , dtype = bool)
        how [ np.logical_or( start == i, end == i) ] = True
        density[i] = np.sum( np.abs(psi[how])**2, axis = 0 )

    density = density / 2

    PR = 1/np.sum(density**2, axis = 0)

    return PR



def DaC_N2_dyn( parameters_system, time, parameters_technical):
    '''
    Main function to calculate the dynamics in the TIP problem.\n

    * **parameters_system**: Object of the class **System_Parameters_TIP**,
    containing physical parameters to describe the system.\n
    * **time**: Array with the times of interest.
    * **parameters_technical**: Object of the class **Technical_Parameters_Dyn**,
    containing several cutoffs and parameters needed in the DaC algorithm.\n
    It returns a 2D numpy array with the values of the PR for the different initial
    states (rows), at each time of interest (columns), together with the
    corresponding initial site .
    '''

    L = parameters_system.system
    h = parameters_system.potential
    Jxx = parameters_system.hopping
    Jz = parameters_system.interaction

    T = time

    l0 = parameters_technical.subsystem
    max_va = parameters_technical.cutoff_variance
    epsilon = parameters_technical.precision
    error_propagation = parameters_technical.error_propagation
    min_jump = parameters_technical.min_jump
    epsilon = parameters_technical.precision
    reduce_memory = parameters_technical.reduce_memory

    PR = []
    real_site = []
    site_now = 0

    auxi_left, auxi_right, where_together = find_auxi(l0)

    Vertex_new, inv_Vertex_new, hopping, band_width = order_RCM(l0)

    all_sites = np.asarray( list( combinations(np.arange(l0), 2) ) )


    while( site_now < L-1 ):

        if( site_now <= 0.5*l0 ):
            h_local = h[ :l0 ]
            J1 = 0
            J2 = Jxx
            first_dyn = site_now
            last_dyn = l0           #Not included

        elif( L-1 - site_now < 0.5*l0):
            h_local = h[ -l0 : ]
            J1 = Jxx
            J2 = 0
            first_dyn = l0-(L-1 - site_now) - 1
            last_dyn = l0           #Not included

        else:
            h_local = h[ site_now -int(0.5*l0) : site_now + int(0.5*l0) ]
            J1 = Jxx
            J2 = Jxx
            first_dyn = int(l0*0.5)
            last_dyn = min(first_dyn + L-1 - site_now, l0)     #Not included


        PR_local, how_many = dynamics_site_first( [first_dyn, last_dyn, h_local, Jxx, Jz, J1, J2, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, max_va, T, epsilon, all_sites, error_propagation, reduce_memory] )


        if(how_many != 0):
            if(len(PR) == 0):
                PR = PR_local
                real_site = np.arange(how_many) + site_now

            else:
                PR = np.concatenate( (PR, PR_local) )

                real_site = np.concatenate( (real_site, np.arange(how_many) + site_now) )

        site_now = site_now + max( how_many, min_jump )

    if(len(real_site) != 0):
        real_site = real_site.astype(int)

    return PR, real_site


def fun_parameters(M):
    '''
    All the parameters needed to calculate the observables.\n
    First is for the mean distance, second for CoM,
    third and forth for PR density, fifth for probability together
    '''

    M2 = int( M*(M-1)*0.5 )

    aux_parameters = []

    #Distance
    which_dist = np.zeros( np.sum(np.arange(M-1, 0, -1)) , dtype = int)

    counter = 0

    for i in range(M-1, 0, -1):

        which_dist[counter:counter+i] = np.arange(1,i+1)
        counter = counter + i

    aux_parameters.append( which_dist )

    #CoM

    CoM = np.zeros( M2 )

    counter = 0

    for i in range(0, M-1):

        CoM[ counter:counter + M - i -1 ] = i + np.arange(i + 1, M)
        counter = counter + M - i -1

    CoM = CoM/2

    aux_parameters.append( CoM )


    #PR den

    start = np.zeros( M, dtype = int )
    start[1:] = np.cumsum( np.arange(M-1, 0, -1) )

    how_density = np.zeros ( (M, M-1), dtype =  int )

    for i in range(0, M-1):

        how_density[i][:start[i+1] - start[i]] = np.arange(start[i], start[i+1] )

        if(i > 1):

            how_density[i][ start[i+1] - start[i] : ] = np.cumsum (np.append(i-1, np.arange(M - 2, M - i - 1 , -1) ) )

    how_density[-1] = np.cumsum( np.append (M-2, np.arange(M-2, 0, -1) ) )



    aux_parameters.append(how_density)
    aux_parameters.append( np.array(M) )

    #Prob together

    where_together = np.zeros(M-1, dtype = int)

    where_together[1:] = np.cumsum( np.arange(M-1, 1, -1) )


    aux_parameters.append(where_together)


    return aux_parameters


def fun_mean_dist(psi, which_dist):
    '''
    Calculate the mean distance of state, stored as rows of matrix psi.
    '''

    d = np.sum( (psi**2)*which_dist, axis = 1)

    return d


def fun_fluc_CoM(psi, which_CoM):
    '''
    Calculate the fluctuation of the CoM
    '''

    mean = np.sum( (psi**2)*which_CoM, axis = 1 )

    D_CoM = np.sum( (psi**2)*(which_CoM**2), axis = 1 ) - mean**2

    return D_CoM


def fun_cal_PR_density(psi, how_density, L):
    '''
    Calculate the PR from normalized density, vectors stored as rows of matrix psi.
    '''

    density = np.zeros( ( len(psi[:, 0]) ,L) )

    for i in range(0, L):
        density[:, i] = np.sum( psi[:, how_density[i]]**2, axis = 1 )

    weight = np.sum(density[0])

    density = density/weight

    PR = 1/np.sum(density**2, axis = 1)

    return PR


def fun_cal_PR_Fock(psi):
    '''
    Calculate the PR in Fock space, vectors stored as rows of matrix psi
    '''

    PR = 1/np.sum(psi**4, axis = 1)

    return PR


def fun_prob_together(psi, where_together):
    '''
    Population on consecutive sites (i,i+1).
    '''

    P = np.sum ( psi[:, where_together]**2, axis = 1 )

    return P



def fun_give_back_observables(psi, parameters, N):
    '''
    Calculate the observables of interest and store them in object of class
    **Observables_TIP_class**.
    '''

    Observables = Observables_TIP_class(fun_mean_dist( psi, parameters[0] ),
                        fun_fluc_CoM ( psi, parameters[1] ),
                        fun_cal_PR_density( psi, parameters[2], parameters[3] ),
                        fun_cal_PR_Fock(psi),
                        fun_prob_together(psi, parameters[4]))

    return Observables



def PR_ED_N2( potential, Jxx, time_interest, Jz, sites, reduce_memory ):
    '''
    Calculate the PR using ED, only for small systems.
    '''

    L = len(potential)

    Vertex_new, inv_Vertex_new, hopping, band_width = order_RCM( L )

    auxi_left, auxi_right, where_together = find_auxi( L )

    J_r = np.array([0, 0])

    E_new, v_new = E_v_subset( [potential, Jxx, J_r, Jz, L, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, 1] )

    all_sites = np.asarray( list( combinations(np.arange(L), 2) ) )

    PR_T = np.zeros( (len(sites), len(time_interest)) )

    for j in range(0, len(sites)):

        local_store = np.einsum("i, ik -> ki", v_new[:, where_together[sites[j]] ], v_new )

        if(reduce_memory):
            final_r, final_c = time_evolution_loop(local_store, E_new, time_interest )
            final = final_r + final_c*1j
            PR_T[j] = cal_PR_density(final, all_sites, np.ones(len(all_sites), dtype = bool) )
        else:
            final = time_evolution(local_store, E_new, time_interest )
            PR_T[j] = cal_PR_density(final, all_sites, np.ones(len(all_sites), dtype = bool) )


    return PR_T
