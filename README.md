# Divide-and-Conquer-solves-localization
Code, based in a Divide-and-Conquer (DaC) scheme, to find the localized eigenstates and the dynamical evolution of states out-of-equilibrium in arbitrarily large systems with few (1 or 2) particles.

##Aim: 
Study physical properties of arbitrarily large systems, reducing the effect of boundary conditions and dealing with apperance of, and effects due to, rare regions (regions where the fluctuations of the random potential is far away from the mean fluctuations). The limitation of the algorithm is given by the localization length of the eigenstates, not the system size. The localization length of the eigenstates depends on the considered potential (microscopic details).


##Model implemented: 
Disordered tight-binding model in 1D:

<img src="https://latex.codecogs.com/svg.image?H&space;=&space;\sum_{i=1}^L&space;h_i&space;n_i&space;&plus;&space;\sum_{i=1}^{L-1}&space;J_{i,i&plus;1}&space;(a_i^\dag&space;a_{i&plus;1}&space;&plus;&space;a_{i&plus;1}^\dag&space;a_{i})&space;&plus;&space;J_z&space;\sum_{i=1}^{L-1}&space;n_i&space;n_{i&plus;1}" title="https://latex.codecogs.com/svg.image?H = \sum_{i=1}^L h_i n_i + \sum_{i=1}^{L-1} J_{i,i+1} (a_i^\dag a_{i+1} + a_{i+1}^\dag a_{i}) + J_z \sum_{i=1}^{L-1} n_i n_{i+1}" />

Other models can be studied, with the corresponding changes of the matrix to diagonalize and the calculation of the variance of the eigenstates of the subsystems.


##Observables of interest:


The observables obtained are:

1. The Participation Ratio, for the case of a single particle:

<img src="https://latex.codecogs.com/svg.image?\text{PR}(\psi)_{N=1}&space;=&space;\frac{1}{\sum_{i=1}^L&space;|\psi(i)|^4}" title="https://latex.codecogs.com/svg.image?\text{PR}(\psi)_{N=1} = \frac{1}{\sum_{i=1}^L |\psi(i)|^4}" />


2. The mean distance between the 2 particles:
<img src="https://latex.codecogs.com/svg.image?\text{D}(\psi)_{N=2}&space;=&space;\sum_{i<j}&space;(j-i)&space;|\psi(i,j)|^2" title="https://latex.codecogs.com/svg.image?\text{D}(\psi)_{N=2} = \sum_{i<j} (j-i) |\psi(i,j)|^2" />


3. The fluctuations of the Center-of-Mass (CoM), for the case of two particles, where CoM is defined as:

<img src="https://latex.codecogs.com/svg.image?\text{CoM}(\psi)_{N=2}&space;=&space;\sum_{i<j}&space;\frac{i&plus;j}{2}&space;|\psi(i,j)|^2" title="https://latex.codecogs.com/svg.image?\text{CoM}(\psi)_{N=2} = \sum_{i<j} \frac{i+j}{2} |\psi(i,j)|^2" />


4. The Participation Ratio from the density in real space, for the 2 particle sector:

<img src="https://latex.codecogs.com/svg.image?\text{PR}_d(\psi)_{N=2}&space;=&space;\sum_i&space;n_i^2,&space;\quad&space;n_i&space;=&space;\frac{1}{2}&space;\sum_{i<j}&space;\psi(i,j)" title="https://latex.codecogs.com/svg.image?\text{PR}_d(\psi)_{N=2} = \sum_i n_i^2, \quad n_i = \frac{1}{2} \sum_{i<j} \psi(i,j)" />

5. The Participation Ratio in Fock space, for the 2 particle sector:

<img src="https://latex.codecogs.com/svg.image?\text{PR}_F(\psi)_{N=2}&space;=&space;\frac{1}{\sum_{i<j}&space;|\psi(i,j)|^4&space;}" title="https://latex.codecogs.com/svg.image?\text{PR}_F(\psi)_{N=2} = \frac{1}{\sum_{i<j} |\psi(i,j)|^4 }" />


6. The probability to find the 2 particles in consecutive sites:

<img src="https://latex.codecogs.com/svg.image?\text{P}_T(\psi)_{N=2}&space;=&space;\sum_{i}&space;|\psi(i,i&plus;1)|^2" title="https://latex.codecogs.com/svg.image?\text{P}_T(\psi)_{N=2} = \sum_{i} |\psi(i,i+1)|^2" />




##Functions:
There are 4 fundamental functions, 2 dealing with the Anderson model (in lib\_N1.py) and 2 functions dealing with the N=2 particles scenario (in lib\_N2.py).

1. DaC\_eigen\_N1( system, subsystem, potential, hopping ; variance = 1e-32, cutoff\_overlap = 1e-7, cutoff\_E = 1e-7 )

Calculate the eigenstate of the Anderson model of a system with *system* sites, with values of the on-site potential given in *potential* and the hopping terms given in *hopping* (entry i of *hopping* determines the hopping between site i an i+1), for a given subsystem of size *subsystem*.

It returns 2 different arrays, one with the obtained eigenvalues and the other the values of the Participation Ratio of the obtained eigenvectors.


2. DaC\_dyn\_N1 ( system, subsystem, potential, Jxx, precision, time; variance = 1e-32, min\_jump = 1, error\_propagation\_ratio = 10 )

Calculate the time evolution of the observable Participation Ratio (PR) for several times of interest, given in the numpy array *time*, where the initial state is a wavefunction with the particle localized in one of the sites of the system. The system has *system* sites, with values of the on-site potential given in *potential* and the hopping coefficient, constant in each site, is given in *Jxx*. An upper bound of the error in the PR is given by the value of *precision*.


It returns a 2D array, with the values of the Participation Ratio for the several times of interest, where the 0-axis refers to the initial states (maximum *system*) and the 1-axis refers to the instance of the time. It also returns a 1D array, specifying which are the initial states of the wavefunctions solved


3. DaC\_eigen\_N2 ( system, subsystem, potential, Jxx, Jz; variance = 1e-32, cutoff\_overlap = 1e-7, cutoff\_E = 1e-7, min\_jump = 0 )

Calculate the eigenstates of the two interacting particle problem, in a system with *system* sites, which can be fit in a subsytem of size *subsystem*. The values of the on-site potential are given in *potential*, the hopping coefficient is *Jxx* and the interaction strength is *Jz*.

It returns 4 arrays: 

   - A 1D-array, with the energy (eigenvalues) of the eigenstates (eigenvectors). 
   - A 2D-array, in the 0-axis are 5 different observables and the 1-axis refers to each of the obtained eigenstates. The observables are, in order, the mean distance, the fluctuations of the Center-of-Mass, the Participation Ratio from density, the Participation Ratio in Fock space and the probability to find the two particles in consecutive sites.
   - The filling factor in the population of the sites (i,i+1), for all i, from the obtained eigenvectors from the DaC method
   - A 1D-array, indicating in which sites the obtained eigenvectors start


4. DaC\_dyn\_N2 ( system, subsystem, potential, Jxx, Jz, time, precision; variance = 1e-32, min\_jump = 1, error\_propagation\_ratio = 10 )

Calculate the time evolution of the observable Participation Ratio (PR) for several times of interest, given in the numpy array *time*, where the initial state is a wavefunction with the two particles localized in consecutive sites of the system. The system has *system* sites, with values of the on-site potential given in *potential*, the hopping coefficient, constant in each site, is given in *Jxx* and the interaction strength is *J_z*. An upper bound of the error in the PR is given by the value of *precision*.




##How to use it: 

Examples of how to use the implemented functions are given in the files test\_functions\_N1.py and test\_functions\_N2.py.



##Future work:

- There is paper under work, where the physics of the few-particles physics and the more technical aspects of the code will be explained and justified in detail.

- Make a python package.