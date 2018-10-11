# Minimizing highly nonconvex functions using Evolutionary Game Theory
This repository provides the code corresponding to my Master's Thesis "Evolutionary Games for Global Function Minimization".

Two notable ressources are the papers ["Spatially Inhomogeneous Evolutionary Games"](https://arxiv.org/abs/1805.04027) by Luigi Ambrosio, Massimo Fornasier, Marco Morandotti and Giuseppe Savaré, which developed the underlying theory on which my thesis built on, as well as ["An analytical framework for a consensus-based global optimization method"]()https://arxiv.org/abs/1602.00220) from José A. Carrillo, Young-Pil Choi, Claudia Titzeck and Oliver Tse, which provided a well-performing comparison and inspiration.

## Usage
In order to provide multiple examples which were also used in the thesis, I provide the scripts contained in `bin/` which start the algorithm with specific parameters.
For example:
```bash
./bin/small_demo
```

<!-- ## Testing
[pytest](https://docs.pytest.org/en/latest/) is a nice tool! Use it with:
```pytest```
I mostly use it here to verify that my vectorized functions are doing what they should do. Describing `J` entry-wise is way easier but does not meet our performance requirements, so that I also have a `J_vectorized` that uses only native numpy.
 -->