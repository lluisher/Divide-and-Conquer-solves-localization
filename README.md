# Divide-and-Conquer-solves-localization
Code, based in a Divide-and-Conquer (DaC) scheme, to find the localized eigenstates and the dynamical evolution of states out-of-equilibrium in arbitrarily large systems with few (1 or 2) particles.
For more details, please see the [paper in Arxiv](http://arxiv.org/abs/2211.13089).
The documentation for the functions can be found at [documentation](https://github.com/lluisher/Divide-and-Conquer-solves-localization/tree/main/html/DaC/index.html).



## Aim:
Study physical properties of arbitrarily large systems, reducing the effect of boundary conditions and dealing with appearance of rare regions, where the fluctuations of the random potential is far away from the mean fluctuations, and their effect. The limitation of the algorithm is given by the localization length of the eigenstates, not the system size. The localization length of the eigenstates depends on the considered potential (microscopic details).


## How to use it and example of results:

Examples of how to use the implemented functions are given in the files example\_DaC\_N1.py and example\_DaC\_N2.py. The accuracy is tested in files test\_DaC\_N1.py and test\_DaC\_N2.py.

The following plot is an example of detail level we can reach thanks to the large statistics, from the full set of eigenvalues of a matrix of dimension 10⁹ x 10⁹

![4D_box_W5d0](https://user-images.githubusercontent.com/102743817/210231439-a5b0e7cc-9a4e-4912-a352-0e02f31b1387.png)


## Model implemented:
Disordered tight-binding model in 1D:

<img src="https://render.githubusercontent.com/render/math?math={H=\sum_{i=1}^L h_i n_i%2B\sum_{i=1}^{L-1}J_{i,i%2B1}(a_i^{\dagger} a_{i%2B 1}%2Ba_{i%2B 1}^\dagger a_{i})%2BJ_z\sum_{i=1}^{L-1}n_in_{i %2B 1}}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}H=\sum_{i=1}^L h_i n_i%2B\sum_{i=1}^{L-1}J_{i,i%2B1}(a_i^{\dagger} a_{i%2B 1}%2Ba_{i%2B 1}^\dagger a_{i})%2BJ_z\sum_{i=1}^{L-1}n_in_{i %2B 1}}#gh-dark-mode-only">

Other models can be studied, with the corresponding changes of the matrix to diagonalize and the calculation of the variance of the eigenstates of the subsystems.


## Observables of interest:


The observables obtained are:

1. The Participation Ratio, for the case of a single particle:

<img src="https://render.githubusercontent.com/render/math?math={\text{PR}(\psi)_{N=1} = \frac{1}{\sum_{i=1}^L |\psi(i)|^4}}#gh-light-mode-only" >
<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{PR}(\psi)_{N=1} = \frac{1}{\sum_{i=1}^L |\psi(i)|^4}}#gh-dark-mode-only" >


2. The mean distance between the 2 particles:
<img src="https://render.githubusercontent.com/render/math?math={\text{D}(\psi)_{N=2} = \sum_{i<j} (j-i) |\psi(i,j)|^2}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{D}(\psi)_{N=2} = \sum_{i<j} (j-i) |\psi(i,j)|^2}#gh-dark-mode-only">



3. The fluctuations of the Center-of-Mass (CoM), for the case of two particles, where CoM is defined as:

<img src="https://render.githubusercontent.com/render/math?math={\text{CoM}(\psi)_{N=2} = \sum_{i<j} \frac{i %2B j}{2} |\psi(i,j)|^2}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{CoM}(\psi)_{N=2} = \sum_{i<j} \frac{i %2B j}{2} |\psi(i,j)|^2}#gh-dark-mode-only">



4. The Participation Ratio from the density in real space, for the 2 particle sector:

<img src="https://render.githubusercontent.com/render/math?math={\text{PR}_d(\psi)_{N=2} = \sum_i n_i^2, \quad n_i = \frac{1}{2} \sum_{i<j} \psi(i,j)}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{PR}_d(\psi)_{N=2} = \sum_i n_i^2, \quad n_i = \frac{1}{2} \sum_{i<j} \psi(i,j)}#gh-dark-mode-only">


5. The Participation Ratio in Fock space, for the 2 particle sector:

<img src="https://render.githubusercontent.com/render/math?math={\text{PR}_F(\psi)_{N=2} = \frac{1}{\sum_{i<j} |\psi(i,j)|^4 }}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{PR}_F(\psi)_{N=2} = \frac{1}{\sum_{i<j} |\psi(i,j)|^4 }}#gh-dark-mode-only">



6. The probability to find the 2 particles in consecutive sites:

<img src="https://render.githubusercontent.com/render/math?math={\text{P}_T(\psi)_{N=2} = \sum_{i} |\psi(i,i%2B 1)|^2}#gh-light-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={\color{white}\text{P}_T(\psi)_{N=2} = \sum_{i} |\psi(i,i%2B 1)|^2}#gh-dark-mode-only">



## Functions:
There are 4 main functions, 2 dealing with the Anderson model (in lib\_N1.py) and 2 functions dealing with the N=2 particles scenario (in lib\_N2.py).

1. [DaC\_eigen\_N1(...)](https://github.com/lluisher/Divide-and-Conquer-solves-localization/tree/main/html/DaC/lib_N1.html)

Calculate the eigenstates of the Anderson model.


2. [DaC\_dyn\_N1 (...)](https://github.com/lluisher/Divide-and-Conquer-solves-localization/tree/main/html/DaC/lib_N1.html)

Calculate the time evolution in the Anderson model, focusing on the the Participation Ratio (PR) for several times of interest


3. [DaC\_eigen\_N2 (...)](https://github.com/lluisher/Divide-and-Conquer-solves-localization/tree/main/html/DaC/lib_N2.html)

Calculate the eigenstates of the two interacting particle problem.

4. [DaC\_dyn\_N2 (...)](https://github.com/lluisher/Divide-and-Conquer-solves-localization/tree/main/html/DaC/lib_N2.html)

Calculate the time evolution in the Two-Interacting Particle problem, focusing on the Participation Ratio (PR) for several times of interest.

## Future work:

- Make a python package.
