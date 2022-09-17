from typing import Union, Optional, Callable
from dataclasses import dataclass
import numpy as np
import scipy.stats as st    # type: ignore
from scipy import optimize  # type: ignore
import numpy.typing as npt


FloatArray = npt.NDArray[np.float_]
Floats = Union[float, FloatArray]


@dataclass
class BlackScholes:
    """The Black-Scholes model.

    The base asset is stock which under the pricing measure follows the SDE
      `d(S_t) = r*S_t*dt + sigma*S_t*d(W_t)`
    where `r` is the interest rate, `sigma>0` is the volatility.

    Attributes:
      s : Initial price, i.e. S_0.
      sigma : Volatility.
      r : Risk-free interest rate.

    Methods:
      call_price: Computes the price of a call option.
      call_delta: Computes delta of a call option.
      call_theta: Computes theta of a call option.
      call_vega: Computes vega of a call option.
      call_gamma: Computes gamma of a call option.
      call_volga: Computes volga of a call option.
      call_iv: Computes the implied volatility of a call option.
      simulate: Simulates paths.
      monte_carlo: Computes expectation by the Monte-Carlo method.
    """
    
    s: float
    sigma: float
    r: float = 0

    def _d1(self, t: Floats, k: Floats) -> Floats:
        """Computes `d_1` from the Black-Scholes formula."""
        return (np.log(self.s / k ) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))
    
    def _d2(self, t: Floats, k: Floats) -> Floats:
        """Computes `d_2` from the Black-Scholes formula."""
        return self._d1(t, k) - self.sigma * np.sqrt(t)

    def call_price(self, t: Floats, k: Floats) -> Floats:
        """Computes the price of a call option.

        Args:
          t: Expiration time (float or ndarray).
          k: Strike (float or ndarray).

        Returns:
          If `t` and `k` are scalars, returns the price of a call option as a
          scalar value. If `t` and/or `k` are arrays, applies NumPy
          broadcasting rules and returns an array of prices.

        Notes:
          For computation on a grid, use `call_price(*vol_grid(t, k))`, where
          `t` and `k` are 1-D arrays with grid coordinates.
        """
        return self.s * st.norm.cdf(self._d1(t, k)) - np.exp(-self.r * t) * k * st.norm.cdf(self._d2(t, k))

    def call_delta(self, t: Floats, k: Floats) -> Floats:
        """Computes delta of a call options.

        See `call_price` for description of arguments and return value.
        """
        return st.norm.cdf(self._d1(t, k))

    def call_theta(self, t: Floats, k: Floats) -> Floats:
        """Computes theta of a call options.

        See `call_price` for description of arguments and return value.
        """
        raise NotImplementedError

    def call_vega(self, t: Floats, k: Floats) -> Floats:
        """Computes vega of a call options.

        See `call_price` for description of arguments and return value.
        """
        raise NotImplementedError

    def call_gamma(self, t: Floats, k: Floats) -> Floats:
        """Computes gamma of a call options.

        See `call_price` for description of arguments and return value.
        """
        raise NotImplementedError

    def call_volga(self, t: Floats, k: Floats) -> Floats:
        """Computes volga of a call options.

        Volga is the second derivative of price with respect to volatility,
        `d^2 price / d sigma^2`.

        See `call_price` for description of arguments and return value.
        """

        raise NotImplementedError



    def simulate(self, t: float, nsteps: int, npaths: int) -> FloatArray:
        """Simulates paths of the price process.

        Args:
          t: Time horizon.
          nsteps: Number of simulation points minus 1, i.e. paths are sampled
            at `t_i = i*dt`, where `i = 0, ..., nsteps`, `dt = t/nsteps`.
          npaths: Number of paths to simulate.

        Returns:
          An array `s` of shape `(nsteps+1, npaths)`, where `s[i, j]` is the
          value of `j`-th path at point `t_i`.
        """
        
        dt = t / nsteps
        
        z = (self.r - self.sigma**2 * 0.5) * dt + np.random.standard_normal(size=(nsteps, npaths)) * self.sigma * np.sqrt(dt)
        
        br_motion = np.concatenate([np.zeros((1, npaths)), np.cumsum(z, axis=0)])
        
        return self.s * np.exp(br_motion)

  
