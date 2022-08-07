# TRNPy
## Python Implementation of the Paper: Topology Representing Networks

This is the implementation of the paper: [Topology Representing Networks](https://www.sciencedirect.com/science/article/abs/pii/0893608094901090) : Thomas Martinetz, Klaus Schulten. Neural Networks, Vol 7, No. 3, pp. 507-522, 1994.

The authors propose two approaches for building the topology preserving map.
1. Simultaneously distributing the pointers over the manifold (vector quantization procedure) using neural gas algorithm[^1] and creating/updating connections using competitive Hebb rule[^2].
2. First distribute the pointers over the manifold and then create the connections between adjacent neural units with overlapping masked Voronoi polyhedra.

However, the authors have compiled the algorithm only for the first approach in the paper, I've implemented the same approach in this repository.

### Utilizing the Code

Topology Representing Network adaptation algorithm has been implemented in `trn.py`. Inputs and outputs from the adaptation function has been documented in-place.

### Simulations

I have simulated the adaptation of a simple 2-Dimensional Square shaped manifold. It can be found under `simulations/simple_square.py`. Simply follow the following steps to run the code for simulation.

1. Install the requirements through PyPI
: `pip install -r requirements.txt`
2. Run the simulation file
: `python3 simulations/simple_square.py`

- [ ] Simulations that authors have used, explicitly in `Figure 6` & `Figure 8` of the paper are yet to be implemented.

### Results

The simulation for the aforementioned 2-D square shaped manifold was carried out using `N = 200` (with other parameters being the same as what was suggested for the simulations by authors in section 5 of the paper) was carried out and the results were graphed.
Adaption iterated for 40000 steps.

**Manifold**

![Manifold](/.outputs/manifold.png)

**Manifold with Pointers on Initial Distribution**

![Manifold](/.outputs/manifold_prior.png)

**Manifold with Adapted Pointers**

![Manifold](/.outputs/manifold_after.png)

****

## Contact

For any inquiries: [chandeepadissanayake@gmail.com](mailto:chandeepadissanayake@gmail.com)

[^1]: "Neural-Gas" Network for Vector Quantization and its Application to Time Series Prediction ([Paper](https://www.ks.uiuc.edu/Publications/Papers/PDF/MART93B/MART93B.pdf))

[^2]: Competitive hebbian learning: Algorithm and demonstrations ([Paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800243))
