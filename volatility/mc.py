from typing import Union, Optional, Iterable, Callable
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import math
import numpy as np
from scipy import stats         # type: ignore

FloatArray = NDArray[float_]
Floats = Union[float, FloatArray]

@dataclass
class MCResult:
    """Results of Monte-Carlo simulation.
    Attributes:
        x: Simulation average (what we want to find).
        error: Margin of error, i.e. the confidence interval for the sought-for
            value is `[x-error, x+error]`.
        conf_prob: Confidence probability of the confidence interval.
        success: True if the desired accuracy has been achieved in simulation.
        iterations: Number of random realizations simulated.
        control_coef: Control variate coefficient (if it was used).
    """
    x: float
    error: float
    conf_prob: float
    success: bool
    iterations: int
    control_coef: Optional[float] = None
        
def monte_carlo(simulator: Callable[[int], NDArray[float_]],
                f: Callable[[NDArray[float_]], NDArray[float_]],
                abs_err: float = 1e-3,
                rel_err: float = 1e-3,
                conf_prob: float = 0.95,
                batch_size: int = 10000,
                max_iter: int = 10000000,
                control_f: Optional[Callable[[NDArray[float_]], NDArray[float_]]] = None,
                control_estimation_iter: int = 5000) -> MCResult:
        """The Monte-Carlo method for random processes.
        This function computes the expected value `E(f(X))`, where `f` is the
        provided function and `X` is a random process which simulated by calling
        `simulator`.
        Simulation is performed in batches of random paths to allow speedup by
        vectorization. One batch is obtained in one call of `simulator`. Simulation
        is stopped when the maximum allowed number of path has been exceeded or the
        method has converged in the sense that
            `error < abs_err + x*rel_err`
        where `x` is the current estimated mean value, `error` is the margin of
        error with given confidence probability, i.e. `error = z*s/sqrt(n)`, where
        `z` is the critical value, `s` is the standard error, `n` is the number of
        paths simulated so far.
        It is also possible to provide a control variate, so that the desired value
        will be estimated as
            `E(f(X) - theta*control_f(X))`
        (this helps to reduce the variance). The optimal coefficient `theta` is
        estimated by running a separate Monte-Carlo method with a small number of
        iterations. The random variable corresponding to `control_f` must have zero
        expectation.
        Args:
            simulator: A function which produces random paths. It must accept a
                single argument `n` which is the number of realizations to simulate
                (will be called with `n=batch_size` or `n=control_estimation_iter`)
                and return an array of shape `(n, d)` where `d` is the number of
                sampling points in one path.
            f: Function to apply to the simulated realizations. It must accept an
                a batch of simulated paths (an array of shape `(bath_size, d)`)
                and return an array of size `n`.`.
            abs_err: Desired absolute error.
            rel_err: Desired relative error.
            conf_prob: Desired confidence probability.
            batch_size: Number of random realizations returned in one call to
                `simulator`.
            max_iter: Maximum allowed number of simulated realizations. The desired
                errors may may be not reached if more than `max_iter` paths are
                required.
          control_f: A control variate. Must satisfy the same requirements as `f`.
          control_estimation_iter: Number of random realizations for estimating
            `theta`.
        Returns:
          An MCResult structure with simulation result.
        """
        
        x: float = 0 # current mean
        sigma: float = 0 # current standard error
        n: int = 0 # amount of batches
        theta: Optional[float] = None # control variate coefficient 
        z : float = stats.norm.ppf((1 + conf_prob) / 2)
        x_sq: float = 0 # cuurent mean of squares
            
        
        if control_f is not None:
            S = simulator(control_estimation_iter)
            covariance = np.cov(f(S), control_f(S))
            
            theta = covariance[0, 1] / covariance[1, 1]
            
        
        while (n == 0  or (
               z * sigma / np.sqrt(batch_size * n) > abs_err + abs(x) * rel_err and max_iter > n * batch_size)):
            
            S = simulator(batch_size)

            y = f(S) - theta * control_f if control_f is not None else f(S)
            
            x  = (x * n + np.mean(y)) / (n + 1)
            
            x_sq = (x_sq * n + np.mean(y**2)) / (n + 1)
            sigma = np.sqrt(x_sq - x**2)
            
            n += 1
        return MCResult(x=x,
                        error=z*sigma/np.sqrt(batch_size*n),
                        conf_prob=conf_prob,success=(max_iter > n * batch_size),
                        iterations=n*batch_size,
                        control_coef=theta) 