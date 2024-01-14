[![DOI](http://joss.theoj.org/papers/10.21105/joss.01081/status.svg)](https://doi.org/10.21105/joss.01081)

# IDTxl

The **I**nformation **D**ynamics **T**oolkit **xl** (IDTxl) is a comprehensive software
package for efficient inference of networks and their node dynamics from
multivariate time series data using information theory. IDTxl provides
functionality to estimate the following measures:

1) For network inference:
    - multivariate transfer entropy (TE)/Granger causality (GC)
    - multivariate mutual information (MI)
    - bivariate TE/GC
    - bivariate MI
2) For analysis of node dynamics:
    - active information storage (AIS)
    - partial information decomposition (PID)

IDTxl implements estimators for discrete and continuous data with parallel
computing engines for both GPU and CPU platforms. Written for Python3.4.3+.

To **get started** have a look at the [wiki](https://github.com/pwollstadt/IDTxl/wiki) and the [documentation](http://pwollstadt.github.io/IDTxl/). For further discussions, join [IDTxl's google group](https://groups.google.com/forum/#!forum/idtxl).

## How to cite
P. Wollstadt, J. T. Lizier, R. Vicente, C. Finn, M. Martinez-Zarzuela, P. Mediano, L. Novelli, M. Wibral (2018). _IDTxl: The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks._ Journal of Open Source Software, 4(34), 1081. [https://doi.org/10.21105/joss.01081](https://doi.org/10.21105/joss.01081).

## How to cite
Wollstadt, Lizier, Vicente, Finn, Martinez Zarzeula, Lindner, Martinez Mediano, Novelli, Wibral, 2017. "IDTxl - The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks", GitHub Repository: https://github.com/pwollstadt/IDTxl.

## Contributors

- [Patricia Wollstadt](http://patriciawollstadt.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany; Honda Research Institute Europe GmbH, Offenbach am Main, Germany
- [Michael Wibral](http://www.uni-goettingen.de/de/datengetriebene+analyse+biologischer+netzwerke+%28wibral%29/603144.html), Campus Institute for Dynamics of Biological Networks, Georg August University, Göttingen, Germany
- [David Alexander Ehrlich](https://www.ds.mpg.de/person/106938), Campus Institute for Dynamics of Biological Networks, Georg August University, Göttingen, Germany; Max Planck Institute for Dynamics and Self-Organization, Goettingen, Germany
- [Joseph T. Lizier](http://lizier.me/joseph/), Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Raul Vicente](http://neuro.cs.ut.ee/people/), Computational Neuroscience Lab, Institute of Computer Science, University of Tartu, Tartu, Estonia
- [Abdullah Makkeh](https://abzinger.github.io/), Campus Institute for Dynamics of Biological Networks, Georg August University, Göttingen, Germany
- Conor Finn, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- Mario Martinez-Zarzuela, Department of Signal Theory and Communications and Telematics Engineering, University of Valladolid, Valladolid, Spain
- Leonardo Novelli, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Pedro Mediano](https://www.doc.ic.ac.uk/~pam213/), Computational Neurodynamics Group, Imperial College London, London, United Kingdom
- Michael Lindner, Campus Institute for Dynamics of Biological Networks, Georg August University, Göttingen, Germany
- Aaron J. Gutknecht, Campus Institute for Dynamics of Biological Networks, Georg August University, Göttingen, Germany

**How to contribute?** We are happy about any feedback on IDTxl. If you would like to contribute, please open an issue or send a pull request with your feature or improvement. Also have a look at the [developer's section](https://github.com/pwollstadt/IDTxl/wiki#developers-section) in the Wiki for details.


## Acknowledgements

This project has been supported by funding through:

- Universities Australia - Deutscher Akademischer Austauschdienst (German Academic Exchange Service) UA-DAAD Australia-Germany Joint Research Co-operation grant "Measuring neural information synthesis and its impairment", Wibral, Lizier, Priesemann, Wollstadt, Finn, 2016-17
- Australian Research Council Discovery Early Career Researcher Award (DECRA) "Relating function of complex networks to structure using information theory", Lizier, 2016-19
- Deutsche Forschungsgemeinschaft (DFG) Grant CRC 1193 C04, Wibral

## Key References
+ Multivariate transfer entropy: Lizier & Rubinov, 2012, Preprint, Technical Report 25/2012,
Max Planck Institute for Mathematics in the Sciences. Available from:
http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf
+ Hierarchical statistical testing for multivariate transfer entropy estimation: [Novelli et al., 2019, Network Neurosci 3(3)](https://www.mitpressjournals.org/doi/full/10.1162/netn_a_00092)
+ Kraskov estimator: [Kraskov et al., 2004, Phys Rev E 69, 066138](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138)
+ Nonuniform embedding: [Faes et al., 2011, Phys Rev E 83, 051112](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.83.051112)
+ Faes' compensated transfer entropy: [Faes et al., 2013, Entropy 15, 198-219](https://www.mdpi.com/1099-4300/15/1/198)
+ PID:
  + [Williams & Beer, 2010, arXiv:1004.2515 [cs.IT]](http://arxiv.org/abs/1004.2515)
  + [Makkeh et al., 2021, Phys Rev E 103, 032149](https://doi.org/10.1103/PhysRevE.103.032149)
  + [Gutknecht et al., 2021, Proc. R. Soc. A: Math. Phys. Eng, 477(2251), 20210110.](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2021.0110)
+ PID estimators:
  + [Bertschinger et al., 2014, Entropy, 16(4)](https://www.mdpi.com/1099-4300/16/4/2161)
  + [Makkeh et al., 2017, Entropy, 19(10)](https://www.mdpi.com/1099-4300/19/10/530)
  + [Makkeh et al., 2018, Entropy, 20(271)](https://www.mdpi.com/1099-4300/20/4/271)
  + [Makkeh et al., 2018, Phys. Rev. E 103, 032149](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.103.032149)
+ History-dependence estimator for neural spiking data: [Rudelt et al., 2021, PLOS Computational Biology, 17(6)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008927)
+ Significant subgraph mining: [Gutknecht et al., 2021, bioRxiv](https://doi.org/10.1101/2021.11.03.467050)
