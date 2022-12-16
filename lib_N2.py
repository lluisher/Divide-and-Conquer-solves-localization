
#MORE THAN 3 SUBSETS ASSUMED

#IMPORTANT, SUBSYSTEM SIZE SHOULD BE EVEN.

#Same subsystem size except maybe the last.


import numpy as np

from scipy.linalg import eig_banded

from itertools import combinations

from numba import jit


def create_connections(M, M2):

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



#Give back edges in matrix form
#Call it only once, no optimized. For M 300, it takes around 15''

def order_RCM(M):

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




#
#
#Calculate which entries represent 1 particle in the extreme left (auxi left), extreme right (auxi right), where the 2 particles are next to each other (where together)
#
#

def find_auxi(Lred):

    L2red = int(Lred*(Lred-1)*0.5)

    auxi_left = np.arange(0, Lred - 1)

    where_together = np.zeros(Lred-1, dtype = int)

    where_together[1:] = np.cumsum( np.arange(Lred-1, 1, -1) )

    auxi_right = np.zeros(Lred-1, dtype = int)

    auxi_right[:-1] = where_together[1:] - 1

    auxi_right[-1] = where_together[-1]



    return auxi_left, auxi_right, where_together





#
#
#Check variance of a possible eigen.
#
#

def check_variance_N2 (x):

    (J1, J2, v, auxi_left, auxi_right) = x

    vari = J2*J2* ( np.sum( v[:, auxi_right]**2, axis = 1 ) ) + J1*J1*( np.sum( v[:, auxi_left]**2, axis = 1 ) )

    return vari



#
#
#Used when comparing eigenfunctions from different subset, give us the common entries. Assume that the subsystem have different sizes
#
#


def generate_how_compare(start_left, start_right, end_left, end_right):

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


#
#
#Compare if eigenfunctions equal or not
#
#

def compare(E_left, v_left, E_center, v_center, which_old, which_new, tol_overlap, Delta_E ):

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






#Create banded matrix. Combine previous plus the diagonal. m subsystem size here

def banded_H(h, Jxx, Jz, m, permutation, hopping, width ):

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





#
#
#Give back number of eigenfunctions, energy, and eigenfunction itself (v)
#
#

def E_v_subset(x):

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



#
#
#DaC N2.
#
#


def DaC_eigen_N2( potential, Jxx, Jz, system, subsystem, variance = 1e-32, cutoff_overlap = 1e-7, cutoff_E = 1e-7, min_jump = 0 ):

    h = potential
    L = system
    M_sub = subsystem
    tol_overlap = cutoff_overlap
    Delta_E = cutoff_E
    if(min_jump <= 0):
        min_jump = int(0.5*M_sub)

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
    Observables = Observables_class()

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



###########################
###########################DYNAMICS
###########################



@jit(nopython=True)
def time_evolution_matrix(A, new_E, T):

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

    (delta, end, h, Jxx, Jz, J1, J2, permutation, inv_permutation, hopping, width, auxi_left, auxi_right, where_together, max_va, T, epsilon, all_sites, error_propagation_ratio) = x

    l0 = len(h)

    error_ampl = epsilon/(error_propagation_ratio*l0*3)

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

            final_r, final_c = time_evolution_matrix(local_store, new_E[maybe], T )

            final = final_r + final_c*1j

            final_r = 0
            final_c = 0
            local_store = 0

            PR_T[j] = cal_PR_density(final, all_sites, relevant_sites)



    else:
        PR_T = []
        how_many = 0

    return PR_T, how_many





def cal_PR_density(psi, all_sites, relevant_sites):

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





#
#
#Calculation of the time evolution of the PR
#
#

def DaC_N2_dyn( potential, Jxx, Jz, subsystem, precision, time, system, variance = 1e-32, min_jump = 0, error_propagation_ratio = 10 ):


    h = potential
    l0 = subsystem
    max_va = variance
    epsilon = precision
    T = time
    L = system

    if(min_jump == 0):
        min_jump = 0.5*l0

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


        PR_local, how_many = dynamics_site_first( [first_dyn, last_dyn, h_local, Jxx, Jz, J1, J2, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, max_va, T, epsilon, all_sites, error_propagation_ratio] )


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






###########################
###########################OBSERVABLES
###########################





#All the parameters needed to calculate the observables
#First is for the mean distance, second for CoM, third and forth for PR den, fifth for probability together


def fun_parameters(M):

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






#Calculate the mean distance of state, stored as rows of matrix psi
def fun_mean_dist(psi, which_dist):

    d = np.sum( (psi**2)*which_dist, axis = 1)

    return d



#Calculate the fluctuation of the CoM
def fun_fluc_CoM(psi, which_CoM):

    mean = np.sum( (psi**2)*which_CoM, axis = 1 )

    D_CoM = np.sum( (psi**2)*(which_CoM**2), axis = 1 ) - mean**2

    return D_CoM


#Calculate the PR from normalized density, vectors stored as rows of matrix psi

def fun_cal_PR_density(psi, how_density, L):

    density = np.zeros( ( len(psi[:, 0]) ,L) )

    for i in range(0, L):
        density[:, i] = np.sum( psi[:, how_density[i]]**2, axis = 1 )

    weight = np.sum(density[0])

    density = density/weight

    PR = 1/np.sum(density**2, axis = 1)

    return PR


#Calculate the PR in Fock space, vectors stored as rows of matrix psi
def fun_cal_PR_Fock(psi):

    PR = 1/np.sum(psi**4, axis = 1)

    return PR




#Population on sites (i,i+1)

def fun_prob_together(psi, where_together):

    P = np.sum ( psi[:, where_together]**2, axis = 1 )

    return P


#Define a class, with observables as attributes

class Observables_class():
    '''Class with the different quantities of interest'''
    def __init__(self, meanDist=np.zeros(0), flucCoM=np.zeros(0), PRDensity=np.zeros(0), PRFock=np.zeros(0), probTogether=np.zeros(0)):
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

        return Observables_class( new_dist, new_CoM, new_PR_Density, new_PR_Fock, new_prob )



def fun_give_back_observables(psi, parameters, N):

    Observables = Observables_class(fun_mean_dist( psi, parameters[0] ), fun_fluc_CoM ( psi, parameters[1] ),
                    fun_cal_PR_density( psi, parameters[2], parameters[3] ), fun_cal_PR_Fock(psi),
                    fun_prob_together(psi, parameters[4]))

    return Observables





#######CHECK WITH ED



def PR_ED_N2( potential, Jxx, time_interest, Jz ):

    L = len(potential)

    Vertex_new, inv_Vertex_new, hopping, band_width = order_RCM( L )

    auxi_left, auxi_right, where_together = find_auxi( L )

    J_r = np.array([0, 0])

    E_new, v_new = E_v_subset( [potential, Jxx, J_r, Jz, L, Vertex_new, inv_Vertex_new, hopping, band_width, auxi_left, auxi_right, where_together, 1] )

    all_sites = np.asarray( list( combinations(np.arange(L), 2) ) )

    PR_T = np.zeros( (L-1, len(time_interest)) )

    for j in range(0, L-1):

        local_store = np.einsum("i, ik -> ki", v_new[:, where_together[j] ], v_new )

        final_r, final_c = time_evolution_matrix(local_store, E_new, time_interest )

        final = final_r + final_c*1j

        PR_T[j] = cal_PR_density(final, all_sites, np.ones(len(all_sites), dtype = bool) )

    return PR_T
