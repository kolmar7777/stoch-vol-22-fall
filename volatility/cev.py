from typing import Union
from numpy import float_
from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np
import scipy.stats as st  # type: ignore
from scipy import optimize, special  # type: ignore
from . import blackscholes


@dataclass
class CEV:
    """The CEV (Constant Elasticity of Variance) model.

    The base asset is stock which under the pricing measure follows the SDE
        `d(S_t) = r*S_t*dt + sigma*S_t^beta*d(W_t)`
    where `r` is the interest rate, `sigma>0` and `beta>=0` are parameters.

    Attributes:
        s: Initial price, i.e. S_0.
        sigma: Volatility.
        beta: Parameter which controls the skew.
        r: Risk-free interest rate.

    Methods:
        call_price: Computes the price of a call option.
        iv: Computes the implied volatility produced by the model.
        calibrate: Calibrates parameters of the model.
    """
    s: float
    sigma: float
    beta: float
    r: float = 0

    def _remove_drift(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]],
    ) -> tuple[Union[float, NDArray[float_]], Union[float, NDArray[float_]]]:
        """Returns adjusted time to expiration and strike which reduce
        computations to the driftless model.

        Notes:
            If `t`, `k` are arrays, usual broadcasting rules are applied.
        """
        if np.isclose(self.r, 0, atol=1e-15):
            return t, k

        expon = 2 * self.r * (self.beta - 1)
        t_adj = (np.exp(expon * t) - 1) / expon
        k_adj = np.exp(-self.r * t) * k
        return t_adj, k_adj

    def call_price(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]]
    ) -> Union[float, NDArray[float_]]:
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

        if np.isclose(self.beta, 1, atol=1e-15):
            return blackscholes.call_price(
                self.s,
                self.sigma,
                t,
                k,
                self.r,
            )

        t_adj, k_adj = self._remove_drift(t, k)
        nu = 0.5 / (self.beta - 1)
        xi = (
            (self.s**(2 * (1 - self.beta)))
            / ((self.sigma * (1 - self.beta))**2 * t_adj)
        )
        y = (
            (k_adj**(2 * (1 - self.beta)))
            / ((self.sigma * (1 - self.beta)) ** 2 * t_adj)
        )

        if self.beta < 1:
            return (
                self.s * st.ncx2(2 * (1 - nu), xi).sf(y)
                - k_adj*st.ncx2(-2*nu, y).cdf(xi)
            )
        return (
            self.s*st.ncx2(2*nu, y).sf(xi)
            - k_adj*st.ncx2(2*(1+nu), xi).cdf(y)
        )


    def iv(
        self,
        t: Union[float, NDArray[float_]],
        k: Union[float, NDArray[float_]],
        use_approx: bool = False,
    ) -> Union[float, NDArray[float_]]:
        """Computes the Black-Scholes implied volatility produced by the model.

        This function first computes the price of a call option with expiration
        time `t` and strike `k`, and then inverts the Black-Scholes formula to
        find `sigma`.

        Args:
            t: Expiration time (float or ndarray).
            k: Strike (float or ndarray).
            use_approx: If True, uses Hagan and Woodward's approximate formula
                to compute option prices. If False, uses the exact formula.

        Returns:
            If `t` and `k` are scalars, returns a scalar value. If `t` and/or
            `k` are arrays, applies NumPy broadcasting rules and returns an
            array. If the implied volatility cannot be computed (i.e. cannot
            solve the Black-Scholes formula for `sigma`), returns NaN in the
            scalar case or puts NaN in the corresponding cell of the array.
        """
        if use_approx:
            #TODO: implement Hagan and Woodward formula
            raise NotImplementedError

        return blackscholes.call_iv(
            self.s,
            self.r,
            self.call_price(t, k),
            t,
            k,
        )

    @staticmethod
    def _calibrate_objective(
        x: tuple[float, float],
        s: float,
        r: float,
        t: NDArray[float_],
        k: NDArray[float_],
        iv: NDArray[float_],
        use_approx: bool
    ) -> float:
        """The objective function for parameter calibration.

        Computes the sum of squares of differences between the model and market
        implied volatilities, for each expiration time and strike.

        Args:
            x: Tuple of calibrated parameters `(sigma, beta)`.
            s: Initial price.
            r: Interest rate.
            t: Array of expiration times.
            k: Array of strikes.
            iv: Array of market implied volatilities for each combination of
            expiration time and strike, i.e. with shape `(len(t), len(k))`. May
            contain NaNs, which are ignored when computing the sum of squares.
            use_approx: If True, uses Hagan and Woodward's approximate formula.
        """
        C = CEV(s=s, sigma=x[0], beta=x[1], r=r)
        return sum((C.iv(t_, k_, use_approx) - iv_)**2
                   for t_, k_, iv_ in zip(t, k, iv))

    @classmethod
    def calibrate(
        cls,
        t: Union[float, NDArray[float_]],
        k: NDArray[float_],
        iv: NDArray[float_],
        s: float,
        r: float = 0,
        use_approx: bool = False,
        min_method: str = "L-BFGS-B",
        beta0=0.8,
        return_minimize_result: bool = False
    ):
        """Calibrates the parameters of the CEV model.

        This function finds the parameters `sigma` and `beta` of the model
        which minimize the sum of squares of the differences between market
        and model implied volatilities. Returns an instance of the class with
        the calibrated parameters.

        Args:
            t : Expiration time (scalar or array).
            k: Array of strikes.
            iv: Array of market implied volatilities.
            s: Initial price.
            r: Interest rate.
            use_approx: If True, uses Hagan and Woodward's approximate formula.
            min_method: Minimization method to be used, as accepted by
                `scipy.optimize.minimize`. The method must be able to handle
                bounds.
            beta0: Initial guess of parameter `beta` (see notes below).
            return_minimize_result: If True, return also the minimization
                result of `scipy.optimize.minimize`.

        Returns:
            If `return_minimize_result` is True, returns a tuple `(cls, res)`,
            where `cls` is an instance of the class with the calibrated
            parameters and `res` is the optimization result returned by
            `scipy.optimize.minimize`. Otherwise returns only `cls`.

        Notes:
            It is advised not to set `beta0=1`, since for `beta` close to 1
            (currently, for `beta` in `[1-1e-12, 1+1e-12]`) the model
            internally switches to the Black-Scholes formula for computation of
            option prices, and there is risk that the minimization method will
            get stuck at `beta=1`.
        """
        k_ = k.flatten()
        iv_ = iv.flatten()
        if isinstance(t, float):
            t_ = np.broadcast_to(t, np.shape(k_))
        else:
            t_ = t.flatten()
        sigma0 = iv_[np.abs(k_-s).argmin()]

        res = optimize.minimize(
            fun=CEV._calibrate_objective,
            x0=(sigma0, beta0),
            bounds=[(0, np.Inf), (0, np.Inf)],
            args=(s, r, t_, k_, iv_, use_approx),
            method=min_method
        )
        ret = cls(s=s, sigma=res.x[0], beta=res.x[1], r=r)
        if return_minimize_result:
            return ret, res
        else:
            return ret
