from __future__ import annotations

from typing import Callable, Any

import numpy as np
from scipy.linalg import solve


def gradient_weno5(y: np.ndarray, dx: np.float64 = 1.0, axis: int = -1) -> np.ndarray:
    """
    Compute the spatial derivative of y following the WENO discretization scheme, robust to discontinuity,
    along a specified axis.

    Parameters
    ----------
    y : np.ndarray
        The array to be derived
    dx : np.float64, optional, default: 1.0
        The space step, the space domain should be evenly spaced with dx
    axis : int, optional, default: -1
        The axis along which to compute the derivative

    Returns
    -------
    np.ndarray
        The order 1 spatial derivative of y at each point along the specified axis
    """
    eps: np.float64 = np.double(1e-200)

    # Move the specified axis to the last position for easier manipulation
    y = np.moveaxis(y, axis, -1)

    # Initialize arrays for alpha0, alpha1, alpha2
    alpha0 = ((13. / 12.) * (y[..., :-4] - 2. * y[..., 1:-3] + y[..., 2:-2]) ** 2
              + (1. / 4.) * (y[..., :-4] - 4. * y[..., 1:-3] + 3. * y[..., 2:-2]) ** 2)
    alpha1 = ((13. / 12.) * (y[..., 2:-2] - 2. * y[..., 3:-1] + y[..., 4:]) ** 2
              + (1. / 4.) * (y[..., 2:-2] - y[..., 4:]) ** 2)
    alpha2 = ((13. / 12.) * (y[..., 1:-3] - 2. * y[..., 2:-2] + y[..., 3:-1]) ** 2
              + (1. / 4.) * (y[..., 1:-3] - 4. * y[..., 2:-2] + 3. * y[..., 3:-1]) ** 2)
    # alpha0 = ((13. / 12.) * (y[..., 2:-2] - 2. * y[..., 3:-1] + y[..., 4:]) ** 2
    #           + (1. / 4.) * (3. * y[..., 2:-2] - 4. * y[..., 3:-1] + y[..., 4:]) ** 2)
    # alpha1 = ((13. / 12.) * (y[..., 1:-3] - 2. * y[..., 2:-2] + y[..., 3:-1]) ** 2
    #           + (1. / 4.) * (y[..., 1:-3] - y[..., 3:-1]) ** 2)
    # alpha2 = ((13. / 12.) * (y[..., :-4] - 2. * y[..., 1:-3] + y[..., 2:-2]) ** 2
    #           + (1. / 4.) * (y[..., :-4] - 4. * y[..., 1:-3] + 3. * y[..., 2:-2]) ** 2)
    eps = 1e-6 * np.min([abs(alpha0), abs(alpha1), abs(alpha2)]) + 1e-150
    # eps = 1e200
    # print(eps, np.argwhere(np.isnan(alpha0)), np.argwhere(np.isnan(alpha1)), np.argwhere(np.isnan(alpha2)))
    # Compute alpha0, alpha1, alpha2 along the last axis

    alpha0 = 0.1 / ((eps + alpha0) * (eps + alpha0))

    alpha1 = 0.6 / ((eps + alpha1) * (eps + alpha1))

    alpha2 = 0.3 / ((eps + alpha2) * (eps + alpha2))

    # print(alpha0, alpha1, alpha2)

    # Compute fp
    fp = (alpha0 * ((2. / 6.) * y[..., :-4] - (7. / 6.) * y[..., 1:-3] + (11. / 6.) * y[..., 2:-2])
          + alpha1 * (-(1. / 6.) * y[..., 1:-3] + (5. / 6.) * y[..., 2:-2] + (2. / 6.) * y[..., 3:-1])
          + alpha2 * ((2. / 6.) * y[..., 2:-2] + (5. / 6.) * y[..., 3:-1] - (1. / 6.) * y[..., 4:])) / (
                 alpha0 + alpha1 + alpha2)
    # fp = (alpha0 * ((-1. / 6.) * y[..., :-4] + (5. / 6.) * y[..., 1:-3] + (2. / 6.) * y[..., 2:-2])
    #       + alpha1 * ((2. / 6.) * y[..., 1:-3] + (5. / 6.) * y[..., 2:-2] - (1. / 6.) * y[..., 3:-1])
    #       + alpha2 * ((11. / 6.) * y[..., 2:-2] - (7. / 6.) * y[..., 3:-1] + (2. / 6.) * y[..., 4:])) / (
    #                  alpha0 + alpha1 + alpha2)
    # Initialize result array
    res = np.zeros_like(y)

    # Compute the derivative along the last axis
    res[..., 3:-2] = (fp[..., 1:] - fp[..., :-1]) / dx
    # res[..., :3] = (y[..., 1:4] - y[..., :3]) / dx
    res[..., 1:3] = (res[..., 2:4] - res[..., :2]) * 2 / dx
    res[..., 0] = (y[..., 1] - y[..., 0]) / dx
    res[..., -2:] = (y[..., -2:] - y[..., -3:-1]) / dx

    # Move the axis back to its original position
    res = np.moveaxis(res, -1, axis)
    # return np.gradient(y, axis=axis) / dx
    # return fp
    return res


def gradient_g(y: np.ndarray[np.float64], dx: np.float64 = np.double(1), axis: int = -1) -> np.ndarray[np.float64]:
    y = np.moveaxis(y, axis, -1)
    res = np.zeros_like(y)
    res[..., 1:] = (y[..., 1:] - y[..., :-1]) / dx
    res[..., 0] = (y[..., 1] - y[..., 0]) / dx
    res = np.moveaxis(res, -1, axis)
    return res


def gradient_d(y: np.ndarray[np.float64], dx: np.float64 = np.double(1), axis: int = -1) -> np.ndarray[np.float64]:
    y = np.moveaxis(y, axis, -1)
    res = np.zeros_like(y)
    res[..., :-1] = (y[..., 1:] - y[..., :-1]) / dx
    res[..., -1] = (y[..., -1] - y[..., -2]) / dx
    res = np.moveaxis(res, -1, axis)
    return res


def gradient(y: np.ndarray[np.float64], dx: np.float64 = np.double(1), axis: int = -1,
             radial_speed: np.ndarray[np.float64] = None) -> np.ndarray[np.float64]:
    y = np.moveaxis(y, axis, -1)
    dg = gradient_g(y, dx, axis)
    dd = gradient_d(y, dx, axis)
    if radial_speed is None:
        res = (dg + dd) / 2
    else:
        res = dd
        if np.any(radial_speed < 0):
            idx_negs: np.ndarray[int] = np.argwhere(radial_speed < 0)[:, 0]
            res[..., idx_negs] = dg[..., idx_negs]
    res = np.moveaxis(res, -1, axis)
    return res


def gradient2(y: np.ndarray[np.float64], dx: np.float64 = np.double(1), axis: int = -1) -> np.ndarray[np.float64]:
    y = np.moveaxis(y, axis, -1)
    res = np.zeros_like(y)
    res[..., 1:-1] = (y[..., 2:] + y[..., :-2] - 2. * y[..., 1:-1]) / (dx * dx)
    # res[..., 0] = (y[..., 0] - 2 * y[..., 1] + y[..., 2]) / (dx * dx)
    # res[..., -1] = (y[..., -3] - 2 * y[..., -2] + y[..., -1]) / (dx * dx)
    # res[wleft * wright == 0] = 0.
    res = np.moveaxis(res, -1, axis)
    return res


def linear_interpolation(x: np.ndarray[np.float64] | np.float64, y: np.ndarray[np.float64] | np.float64, ):
    """
    Linear interpolation

    Parameters
    ----------
    x : np.float64 | np.ndarray[np.float64]
        The variable
    y : np.float64 | np.ndarray[np.float64]
        The datapoints asssociated with the variable x to be interpolated. Both x and y can be multisimensionals vectors

    Returns
    -------
    A function that returns the interpolated value
    """

    def result(x_new: np.float64) -> np.float64 | np.ndarray:
        # i = np.searchsorted(x, x_new)
        # i = np.clip(i, 0, len(x) - 2)
        i = np.argmin(np.abs(x_new - x))
        x1 = x[i]
        x2 = x[i + 1]
        alpha = (x_new - x1) / (x2 - x1)
        y_new = (1 - alpha) * y[i] + alpha * y[i + 1]
        return y_new

    return result


# Intégration d'équations différentielles
class SolutionED:
    def __init__(self, x, y: list | np.ndarray, itérations: int = -1):
        self.t: np.ndarray = np.array(x)
        self.y: np.ndarray = np.array(y).T
        self.status: int = 0
        self.iteration: int = itérations


def rk4(
        f: Callable,
        x0: np.float64 | np.ndarray = None, y0: np.float64 | np.ndarray = None,
        xf: np.float64 = np.double(-1), dx_max: np.float64 = None,
        args: list | tuple = None, IT_s_max: int = 1000,
        f_dx: Callable = None,
) -> SolutionED:
    if args is None:
        args = []
    if dx_max is None:
        dx_max: np.float64 = (xf - x0) / IT_s_max
    Y = [y0]
    Xf: list = [x0]
    if f_dx is not None:
        dx: np.float64 = np.sign(dx_max) * min(abs(dx_max), abs(f_dx(x0, y0, *args)))
    else:
        dx: np.float64 = dx_max
    k1 = f(Xf[-1], Y[-1], *args)
    while Xf[-1] < xf:
        k2 = f(Xf[-1] + dx / 2, Y[-1] + (dx / 2) * k1, *args)
        k3 = f(Xf[-1] + dx / 2, Y[-1] + (dx / 2) * k2, *args)
        k4 = f(Xf[-1] + dx, Y[-1] + dx * k3, *args)
        y = Y[-1] + (dx / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if f_dx is not None :
            dx: np.float64 = np.sign(dx_max) * min(abs(dx_max), abs(f_dx(Xf[-1] + dx, y, *args)))
        Xf.append(Xf[-1] + dx)
        Y.append(y)

    résultat: SolutionED = SolutionED(np.array(Xf), Y)
    return résultat
