# Minimizing Earth Movers Distance
Currently using SciPy approx_fprime to move distributions 'closer' to each other.

Run `python diff.py`.



## TODOs
- Clean up basic scipy minimizer code
- Test with larger number distributions
- Try a black-box differentiation scheme via Torch/TF
- 

## Setup
```
pipenv shell
pipenv install Pipfile.lock
```

### Dependencies
- Python 3.8
- SciPy
- cvxopt
- POT Python Optimal Transport Library

