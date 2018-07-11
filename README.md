# Master's Thesis Mathematics - Minimizing highly nonconvex functions using evolutionary game theory
The underlying theory is developedby Luigi Ambrosio, Massimo Fornasier, Marco Morandotti and Giuseppe Savaré in their paper ["Spatially Inhomogeneous Evolutionary Games"](https://arxiv.org/abs/1805.04027).

In this project we create an algorithm to minimize highly non-convex functions, by designing a game `J` such that the particles are incentivized to converge to a global minimum.

This is still a work in progress.

## Usage
Parameters are currently mostly changed inside the main script `egt/vectorized_script.py`. Run in with the current configuration using:
```python -m egt.vectorized_script```

## Testing
[pytest](https://docs.pytest.org/en/latest/) is a nice tool! Use it with:
```pytest```
I mostly use it here to verify that my vectorized functions are doing what they should do. Describing `J` entry-wise is way easier but does not meet our performance requirements, so that I also have a `J_vectorized` that uses only native numpy.
