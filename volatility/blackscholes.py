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
        return (-self.s*st.norm.pdf(self._d1(t, k)) *
                self.sigma/(2*np.sqrt(t)) -
                self.r*np.exp(-self.r*t)*k*st.norm.cdf(self._d2(t, k)))

    def call_vega(self, t: Floats, k: Floats) -> Floats:
        """Computes vega of a call options.s

        See `call_price` for description of arguments and return value.
        """
        return self.s * st.norm.pdf(self._d1(t, k)) * np.sqrt(t)

    def call_gamma(self, t: Floats, k: Floats) -> Floats:
        """Computes gamma of a call options.

        See `call_price` for description of arguments and return value.
        """
        return st.norm.pdf(self._d1(t, k)) / (self.s*self.sigma*np.sqrt(t))

    def call_volga(self, t: Floats, k: Floats) -> Floats:
        """Computes volga of a call options.

        Volga is the second derivative of price with respect to volatility,
        `d^2 price / d sigma^2`.

        See `call_price` for description of arguments and return value.
        """

        return (self.call_vega(t, k) *
                self._d1(t, k)*self._d2(t, k) / self.sigma)

    def call_iv(
        self,
        c: Floats,
        t: Floats,
        k: Floats,
        iv_approx_bounds: Optional[tuple[float, float]] = None
    ) -> Floats:
        """Computes the implied volatility of a call option.
        This function wraps over the module's `call_iv` function, using the
        initial price and risk-free interest rate from the class attributes,
        and ignoring the class attribute `sigma` (volatility).
        See `call_iv` for the description of parameters and return value.
        """
        return call_iv(self.s, self.r, c, t, k, iv_approx_bounds)

    def simulate(self, t: Floats, nsteps: int, npaths: int) -> FloatArray:
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


"""
The following functions are used for computation of implied volatility.
* call_price implements the Black-Scholes formula.
* call_iv returns the implied volatility.
* _call_iv_approx gives an initial estimate of IV.
* _call_iv_f is the objective function for root finding
    (Black-Scholes price - market price).
* _call_iv_fprime is the derivative of the objective function.
"""


def call_price(
    s: Floats,
    sigma: Floats,
    t: Floats,
    k: Floats,
    r: Floats = 0
) -> Floats:
    """Computes the Black--Scholes price of a call option.
    Args:
        s: Underlying asset price.
        sigma: Volatility.
        t: Expiration time.
        k: Strike.
        r: Interest rate.
    Returns:
        Call option price. If the arguments are arrays, applies NumPy
        broadcasting rules and returns the array of prices.
    Notes:
        This functions does the same computation as BlackScholes.call_price,
        but it allows to vectorize `s`, `sigma` and `r`.
    """
    d1 = (np.log(s/k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    return s * st.norm.cdf(d1) - np.exp(-r * t) * k * st.norm.cdf(d2)

   


# For compatibility
_call_price = call_price


def _call_iv_approx(
    s: Floats,
    r: Floats,
    c: Floats,
    t: Floats,
    k: Optional[Floats] = None
) -> Floats:
    """Computes an approximation of the implied volatility of a call option.
    This is an auxiliary function to find an initial estimate of the implied
    volatility to be used in Newton's method in function `call_iv`.
    See `call_iv` for description of arguments and return value.
    Notes:
        If `k` is `None`, the Brenner-Subrahmanyam formula is used. If ` k` is
        not `None`, the Corrado-Miller formula is used.
    """
    
    if k is not None:
        x = k * np.exp(-r * t)
        a = np.maximum(0, (c - (s - x)/2)**2 - (s - x)**2 / np.pi)
        
        return np.sqrt(2 * np.pi / t) / (s + x) * (c - (s - x)/2 + np.sqrt(a))
    else:
        return 2.5 * c / (s * np.sqrt(t))
    
    
def _call_iv_f(
    sigma: Floats,
    s: Floats,
    r: Floats,
    c: Floats,
    t: Floats,
    k: Floats
) -> Floats:
    return call_price(s, sigma, t, k, r) - c


def _call_iv_fprime(
    sigma: Floats,
    s: Floats,
    r: Floats,
    c: Floats,
    t: Floats,
    k: Floats
) -> Floats:
    d1 = (np.log(s/k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    
    return s * np.sqrt(t) * st.norm.pdf(d1)


def call_iv(
    s: float,
    r: float,
    c: Floats,
    t: Floats,
    k: Floats,
    iv_approx_bounds: Optional[tuple[float, float]] = None
) -> Floats:
    """Computes the implied volatility of a call option.
    This function uses Newton's root finding method to invert the Black-Scholes
    formula with respect to `sigma`.
    Args:
        s: Underlying asset price.
        r: Risk-free interest rate.
        c: Market call option price.
        t: Expiration time.
        k: Strike (optional).
        iv_approx_bounds: A tuple `(min, max)` or `None` such that the initial
            guess of the implied volatility will be truncated if it is outside
            the interval `[min, max]`. This is useful for extreme strikes or
            maturities, when the approximate formula gives unrealistic results.
            If None, no truncation will be applied.
    Returns:
        The implied volatility if Newton's method converged successfully;
        otherwise returns NaN. If the arguments are arrays, which must be of
        the same shape, an array of the same shape is returned, where each
        element is the implied volatility or NaN.
    Notes:
        For computation on a grid of option prices, use
        `call_iv(s, r, c,  *vol_grid(t, k))`, where `t` and `k` are 1-D arrays
        with grid coordinates, and `c` is a 2-D of option prices.
    """
    iva = _call_iv_approx(s, r, c, t, k)
    
    if iv_approx_bounds is None:
        iva_0 = iva
    else:
        iva_0 = np.minimum(np.maximum(iva, iv_approx_bounds[0]), iv_approx_bounds[1])
        
    result = optimize.newton(func = _call_iv_f, 
                             args = (s, r, c, t, k),
                             x0 = iva_0,
                             fprime = _call_iv_fprime,
                             full_output = True)
    
    return np.where(result.converged, result.root, np.NaN)

    
        
   
        
        
        
        
        
        
        
        
        
        

  
