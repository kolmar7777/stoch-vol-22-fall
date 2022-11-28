from typing import Union, Optional, Callable
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import scipy.stats as st         # type: ignore
import scipy.optimize as opt     # type: ignore
import scipy.integrate as intg   # type: ignore
import scipy.special as spec     # type: ignore
import scipy.misc as spmisc      # type: ignore
from . import blackscholes


FloatArray = npt.NDArray[np.float_]
Floats = Union[float, FloatArray]

@dataclass
class Heston:
    """The Heston model.

    The base asset is stock which under the pricing measure follows the SDEs
      `d(S_t) = r*S_t*dt + sqrt(V_t)*d(W^1_t),
       d(V_t) = kappa*(theta - V_t)*dt + sigma*sqrt(V_t)*d(W^2_t)`
    where `r` is the interest rate, `V_t` is the variance process, `W^1_t` and
    `W^2_t` are standard Brownian motions with correlation coefficient `rho`,
    and `kappa>0, theta>0, sigma>0, -1 < rho < 1` are the model parameters.

    Attributes:
      s: Initial price, i.e. S_0.
      v: Initial variance, i.e. v_0.
      kappa, theta, sigma, rho: Model parameters.
      r: Interest rate.

    Methods:
      is_positive: Checks Feller's condition.
      call_price: Computes call option price.
      iv: Computes implied volatility produced by the model.
      calibrate: Calibrates parameters of the model.
      simulate_euler: Simulates paths by Euler's scheme.
      simulate_em: Simulates paths by E+M scheme.
      simulate_qe: Simulates paths by Andersen's QE scheme.
      simulate_exact: Simulates paths by Broadie-Kaya's exact scheme.
      monte_carlo: Computes expectation by the Monte-Carlo method.
    """
    s: float
    v: float
    kappa: float
    theta: float
    sigma: float
    rho: float
    r: float = 0

    def is_positive(self) -> bool:
        """Checks the condition of strict positivity of variance process."""
        return 2*self.kappa*self.theta >= self.sigma**2

    def _cf(self, u: float, t: float) -> complex:
        """The characteristic function of the log-price: `E(i*u*log(S_t))`.

        This realization uses the "good" solution of the Riccati equation
        (see "The Little Heston Trap" by Albrecher et at. (2007)).
        """
        d = np.sqrt(
            (self.rho*self.sigma*u*1j - self.kappa)**2 +
            self.sigma**2*(u*1j + u**2))
        g = ((self.rho*self.sigma*u*1j - self.kappa + d) /
             (self.rho*self.sigma*u*1j - self.kappa - d))
        C = (self.r*u*t*1j + self.kappa*self.theta/self.sigma**2 *
             (t*(self.kappa - self.rho*self.sigma*u*1j - d) -
              2*np.log((1 - g*np.exp(-d*t))/(1-g))))
        D = ((self.kappa - self.rho*self.sigma*u*1j - d)/self.sigma**2 *
             ((1 - np.exp(-d*t)) / (1 - g*np.exp(-d*t))))
        return np.exp(C + D*self.v + u*np.log(self.s)*1j)

    def call_price(self, t: Floats, k: Floats,
                   method: str = "heston") -> Floats:
        """Computes the price of a call option.

        Args:
          t: Expiration time (float or ndarray).
          k: Strike (float or ndarray).
          method: 'heston' or 'lewis'.

        Returns:
          If `t` and `k` are scalars, returns the price of a call option as a
          scalar value. If `t` and/or `k` are arrays, applies NumPy
          broadcasting rules and returns an array of prices.

        Notes:
          Heston's method is the one from the original paper by S. Heston, but
          uses the stable representation of the characteristic function (see
          Albrecher et al. "The little Heston trap" (2007)). Lewis' method is
          the one from his book "Option Valuation under Stochastic Volatility"
          (2000), which is based on the fundamantal trnsform.

          This function will not work faster if arrays are passed to it, as
          internally it does not use NumPy's vectorization.
        """
        b = np.broadcast(t, k)
        c = np.empty(b.shape)  # Stores option prices

        if method == "heston":
            def integrand(u, t_, k_):
                return (np.exp(-1j*u*np.log(k_))/(1j*u) *
                        (self._cf(u-1j, t_) - k_*self._cf(u, t_))).real

            c.flat = [0.5*(self.s - np.exp(-self.r*t_)*k_) +
                      1/np.pi * np.exp(-self.r*t_) *
                      intg.quad(integrand, 0, np.Inf, args=(t_, k_))[0]
                      for (t_, k_) in b]
            
            if b.nd:  # Vector arguments were supplied
                return c
            
            else:
                return float(c)
        elif method == "lewis":
            pass
        else:
            raise ValueError(
                f"Method is '{method}', but it must be 'heston' or 'lewis'")

    def iv(self, t: Floats, k: Floats, method="heston") -> Floats:
        """Computes the Black-Scholes implied volatility produced by the model.

        This function first computes the price of a call option with expiration
        time `t` and strike `k`, and then inverts the Black-Scholes formula to
        find `sigma`.

        Args:
          t: Expiration time (float or ndarray).
          k: Strike (float or ndarray).
          method: Method to compute option prices ('heston' or 'lewis').

        Returns:
          If `t` and `k` are scalars, returns a scalar value. If `t` and/or `k`
          are arrays, applies NumPy broadcasting rules and returns an array. If
          the implied volatility cannot be computed (i.e. cannot solve the
          Black-Scholes formula for `sigma`), returns NaN in the scalar case or
          puts NaN in the corresponding cell of the array.
        """
        return blackscholes.call_iv(
            self.s, self.r, self.call_price(t, k, method), t, k)
