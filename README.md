# Minimizing Earth Movers Distance
Currently using SciPy approx_fprime to move distributions 'closer' to each other.

Run `python diff_test.py`.



## TODOs
- Clean up basic scipy minimizer code
- Test with larger number distributions
- Try a black-box differentiation scheme via Torch/TF
- 

## Setup
```
pipenv install
pipenv shell
```

### Dependencies
- Python 3.8
- SciPy
- cvxopt
- POT Python Optimal Transport Library

