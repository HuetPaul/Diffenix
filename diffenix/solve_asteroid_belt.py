import numpy as np
from scipy.integrate import trapezoid, cumulative_trapezoid, quad, solve_ivp
from scipy.interpolate import Akima1DInterpolator, FloaterHormannInterpolator, PchipInterpolator
from scipy.optimize import root
from scipy.signal import savgol_filter
import phenigraph as g

from diffenix.Constantes import *
from diffenix.numericals_methods import linear_interpolation, rk4, gradient2


def linear_interp(t_new, t_old, values):
    if isinstance(t_new, list | np.ndarray):
        return np.array([linear_interp(t_new=t, t_old=t_old, values=values) for t in t_new])

    i = np.argmin(np.abs(t_new - np.array(t_old)))
    if i == 0 or (i < len(t_old) - 1 and t_new > t_old[i]):
        t1 = t_old[i]
        t2 = t_old[i + 1]
        alpha = (t_new - t1) / (t2 - t1)
        return (1 - alpha) * values[i] + alpha * values[i + 1]
    else:
        t1 = t_old[i - 1]
        t2 = t_old[i]
        alpha = (t_new - t1) / (t2 - t1)
        return (1 - alpha) * values[i - 1] + alpha * values[i]



class Compteur:
    """
    This object can be used to estimate the number of calls for a function if given in parameter
    and incremented in the function (mutable object)

    """

    def __init__(self):
        self.value: int = 0


def number_density(a, amax: np.float64, abig: np.float64, amed: np.float64, amin: np.float64,
                   qh: np.float64, qm: np.float64, ql: np.float64,
                   Mtot: np.float64 = np.double(1.), rho: np.float64 = np.double(1.)) -> np.float64 | np.ndarray:
    """

    Return the number density of asteroid of size a

    Parameters
    ----------
    a : np.float64 | np.ndarray
        The diameter(s) of asteroids
    amax : np.float64
        The maximum diameter
    abig : np.float64
        A reference diameter
    amed : np.float64

    amin : np.float64
        The minimum diameter
    qh : np.float64
        The slope of distribution for the highest radius
    qm : np.float64
        The slope of distribution for medium seized asteroids
    ql : np.float64
        The slope of distribution for the smallest asteroids
    Mtot : np.float64, optional, default=1
        The total mass of the belt.
        The result is directly proportional to Mtot.
        number_density(Mtot)=Mtot*number_density(1.).
        It can be used to estimate the number density
        in a belt = sigma*number_density(1.) with sigma the surface density
    rho : np.float64
        The mean density of asteroids

    Returns
    -------
    np.float64 | np.ndarray
        The number density of asteroids for diameter(s) a

    """

    # Normalization of the distribution. Nmax is the value of the number density for a=amax calculated such as
    # the integration of the number density multiplied by the mass of asteroids for a diameter a giving the total mass
    Nmax: np.float64 = (6. * Mtot
                        / (Pi * rho
                           * (amax * amax * amax * amax / (qh + 4.)
                              + ((abig / amax) ** qh)
                              * ((abig * abig * abig * abig * (1. / (qm + 4.) - 1 / (qh + 4.)))
                                 + ((amed / abig) ** qm)
                                 * (amed * amed * amed * amed * (1. / (ql + 4.) - 1. / (qm + 4.))
                                    - ((amin / amed) ** ql) * amin * amin * amin * amin / (ql + 4.)))
                              )))
    # Nmax: np.float64 = (6. * Mbelt
    #                    / (Pi * rho_refr
    #                       * (((abig / amax) ** qh) * ((amed / abig) ** qm) * ql * (1 - (amin / amed) ** (ql - 1))
    #                           + ((abig / amax) ** qh) * qm * (1 - (amed / abig) ** (qm - 1))
    #                          + qh * (1 - (abig / amax) ** (qh - 1)))))
    if type(a) is np.ndarray:
        n: np.ndarray = np.zeros_like(a)
        idx: np.ndarray = np.intersect1d(np.argwhere(a >= amin)[:, 0], np.argwhere(a < amed)[:, 0])
        n[idx] = Nmax * ((abig / amax) ** qh) * ((amed / abig) ** qm) * ((a[idx] / amed) ** ql)
        idx: np.ndarray = np.intersect1d(np.argwhere(a >= amed)[:, 0], np.argwhere(a < abig)[:, 0])
        n[idx] = Nmax * ((abig / amax) ** qh) * ((a[idx] / abig) ** qm)
        idx: np.ndarray = np.intersect1d(np.argwhere(a >= abig)[:, 0], np.argwhere(a <= amax)[:, 0])
        n[idx] = Nmax * ((a[idx] / amax) ** qh)
        return n
    elif a < amin or a > amax:
        return zero
    elif a < amed:
        return Nmax * ((abig / amax) ** qh) * ((amed / abig) ** qm) * ((a / amed) ** ql)
    elif a < abig:
        return Nmax * ((abig / amax) ** qh) * ((a / abig) ** qm)
    else:
        return Nmax * ((a / amax) ** qh)


mu_H2O: np.float64 = np.double(18. * 1e-3) * C_M
Tref: np.float64 = np.double(273.)  # K
hvap: np.float64 = np.double(2.78e6) * C_E / C_M
Bh2o: np.float64 = np.double(mu_H2O * hvap / Rgp)
Ah2o: np.float64 = np.double(1e5 * C_M / (C_t * C_L)) * np.exp(Bh2o / Tref)


def temp_belt(r: np.ndarray | np.float64, t: np.float64 | np.ndarray, consts: dict
              ) -> np.ndarray | np.float64:
    """

    Calculate and return the temperature at radius r and time t

    Parameters
    ----------
    r : np.ndarray | np.float64
        radius
    t : np.ndarray | np.float64
        the current time
    consts : dict
        the constantes
    Returns
    -------
    np.ndarray | np.float64

    """
    if "Ms" in consts.keys():
        if interpLbol is not None:
            temp0 = (278 * (((1. - consts["Abelt"]) * interpLbol(t, consts["Ms"]) / Lsun) ** (1 / 4)) * (
                    (a0 / au) ** (-1 / 2)))
        else:
            temp0 = (278 * (((1. - consts["Abelt"]) * consts["Ls"] / Lsun) ** (1 / 4)) * (
                    (a0 / au) ** (-1 / 2)))
    else:
        if interpLbol is not None:
            temp0 = (278 * (((1. - consts["Abelt"]) * interpLbol(t, consts["Ms"]) / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))
        else:
            temp0 = (278 * (((1. - consts["Abelt"]) * consts["Ls"] / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))
    if isinstance(t, float | np.float64 | int | np.int64) and t < 0:
        return zero
    elif t.shape == r.shape and np.any(t < zero):
        # temp[t < consts["t0"]] = 10
        temp0[t < consts["t0"]] = 10

    temp = temp0 * (abs(r) / a0) ** (- 1 / 2)
    return temp


def tau_diff(pp: np.ndarray, rr: np.ndarray, t: np.float64 | np.ndarray, consts: dict) -> np.ndarray:
    """
    Calculate the diffusion timescale at time t and depth p

    Parameters
    ----------

    p: np.ndarray
        The depth at which the gas is realised

    r: np.ndarray
        The distance from the central star at which the gas is realised

    t: np.ndarray
        The time at which the gas is realised

    consts
        All the necessary constants

    Returns
    -------
    np.ndarray
        The time needed for the gas to go at the surface

    """
    return (3. * pp * pp * np.sqrt(2. * Pi * mu_H2O
                                   / (Rgp * temp_belt(rr, t - pp * pp / consts["K"], consts)))
            / (4. * consts["Phi"] * consts["rp"]))


def sigmap_rho(t: np.float64 | np.ndarray,
               rr: np.ndarray, pp: np.ndarray,
               consts: dict, comp: Compteur = Compteur()) \
        -> np.ndarray:
    """
    Calculate and return the gas generation rate at depth p, radius r and time t.
    This doesn't take into acompte the limited amonth of material

    Parameters
    ----------
    t : np.ndarray | np.float64
        The current time
    r : np.ndarray | np.float64
        The distance to the central star (semi-major axis)
    p : np.ndarray | np.float64
        The depth into the asteroid/KBO
    consts : dict
        The constantes
    comp : Compteur, optional
        To estimate the number of iterations

    Returns
    -------
    np.ndarray | np.float64

    """

    S: np.float64 = 3. * (1. - consts["Phi"]) / consts["rp"]
    T: np.ndarray = temp_belt(rr, t - pp * pp / consts["K"], consts)
    # The pp * pp / consts["K"] factor model the thermal diffusion

    drho_sub = (S * Ah2o * np.exp(-Bh2o / T)
                * np.sqrt(mu_H2O / (2. * Pi * Rgp * T)))
    if comp.value % 1000 == 0:
        print("Iteration ", comp.value, ": t=", t.max() / Myr, " Myr")
        # print("Iteration ", comp.value, ": t=", t / Myr, " Myr")

    comp.value += 1
    return drho_sub


def minit(r: np.ndarray, pinit: np.float64, rho_ice: np.float64, a0: np.float64, delta_a: np.float64,
          Mbelt: np.float64, kwargsN) -> np.float64:

    p: np.ndarray = np.geomspace(kwargsN['amin'], kwargsN['amax'], 10000)
    rr: np.ndarray = r[:, np.newaxis] + np.zeros_like(p)
    pp: np.ndarray = (p[:, np.newaxis] + np.zeros_like(r)).T
    rhod_control = 4. * Pi * cumulative_trapezoid(p * p * rho_ice * np.ones_like(pp), p, axis=1, initial=zero)
    rhod_control[pp > pinit] *= 0
    sigma0: np.ndarray = (Mbelt * ((r / a0) ** (-3. / 2.))
                          / (4. * Pi * a0 * a0 * (np.sqrt(1. + delta_a / a0)
                                                  - np.sqrt(1. - delta_a / a0))))  # initial surface density
    ni: np.ndarray = ((np.array([number_density(2. * d_2, Mtot=np.double(1.), **kwargsN)
                                 for d_2 in p])[:, np.newaxis] + np.zeros(len(r))).T
                      * (sigma0[:, np.newaxis] + np.zeros(len(p))))  # number density
    sig_init: np.ndarray = trapezoid(ni * rhod_control, 2 * p)
    return trapezoid(2. * Pi * r * sig_init, r)


def sigma_dot(Mbelt: np.float64, r: np.ndarray,
              tps: np.ndarray,
              a0: np.float64, delta_a: np.float64,
              tinit: np.float64, tmax: np.float64,
              f_ice: np.float64, rho_refr: np.float64, rho_ice: np.float64,
              Abelt: np.float64, K: np.float64,
              Its: int, Phi: np.float64,
              dmin: np.float64, dmax: np.float64, Its_p: int,
              rp: np.float64,
              Itt: int, kwargsN: dict, **kwargs
              ) -> np.ndarray:
    """
    Estimate the surface density mass generation rate for asteroids of diameters d at the distance r of the central star
    and of diameter d in a belt of mass Mbelt during the time times_loc

    Parameters
    ----------
    Mbelt : np.float64
        The total belt's mass
    r : np.ndarray
        The distance to the central star
    p : np.ndarray
        The asteroids/KBO diameters
    tps : np.ndarray
        The time
    a0 : np.float64
        The belt's semi-major axis
    delta_a : np.float64
        The (half) size pf the belt
    tinit : np.float64
        The initial time
    tmax : np.float64
        The maximum time
    f_ice : np.float64
        The initial ice to refractory mass ratio
    rho_refr : np.float64
        The asteroids/KBO density
    rho_ice : np.float64
        The asteroids/KBO ice density
    Abelt : np.float64
        The asteroids/KBO albedo
    K : np.float64
        The thermal diffusion coefficient
    Its : np.float64
        The number of spatial steps
    Phi : np.float64
        The porosity (dimensionless number)
    dmin : np.float64
        The minimum diameters
    dmax : np.float64
        The maximum diameters
    Its_p : np.float64
        The number of diameters steps
    rp : np.float64
        The pore's radius
    Itt : np.float64
        The number of time steps
    kwargsN : dict
        The directory with all informations to build the belt's size number distribution
    kwargs : dict

    Returns
    -------
    np.ndarray

    """
    if r is None:
        r = np.linspace(a0 - delta_a, a0 + delta_a, Its)

    pmin: np.float64 = max(dmin / 2., np.sqrt(K * 0.1 * Myr))
    # pmin: np.float64 = max(dmin / 2., np.sqrt(K * 0.1 * yr))
    p = np.geomspace(pmin / 2., dmax / 2., Its_p)
    print("pmin= ", pmin / C_L, "m")

    m_init: np.float64 = minit(r=r, pinit=pmin, rho_ice=rho_ice,
                               a0=a0, delta_a=delta_a, Mbelt=Mbelt, kwargsN=kwargsN)

    if tps is None:
        tps = np.geomspace(max(tinit, yr), tmax, Itt)
    else:
        tinit = tps.min()
        tmax = tps.max()

    consts: dict = {"C_L": C_L, "C_M": C_M, "C_t": C_t,
                    "Ms": Ms, "Abelt": Abelt, "K": K,
                    "a0": a0, "delta_a": delta_a, "t0": t0,
                    "rho_refr": rho_refr,
                    "Its": Its, "Its_p": Its_p,
                    "f_ice": f_ice,
                    "Phi": Phi, "rp": rp}

    # Estimation of the gas sublimation rate realised for every distance of the central star r and depth p
    # The factor tau_diff(p, r, t, consts) model the diffusion of the gas through the pore of refractory materials
    # The thermal diffusion is modeled in sigmap_rho

    rr: np.ndarray = r[::5, np.newaxis] + np.zeros_like(p)
    pp: np.ndarray = (p[:, np.newaxis] + np.zeros_like(r[::5])).T
    comp: Compteur = Compteur()

    if kwargsN is None:  # Default size distribution
        kwargsN = dict(amax=np.double(1.e6) * C_L,
                       abig=np.double(120e3) * C_L,
                       amed=np.double(20.e3) * C_L, amin=C_L,
                       qh=np.double(-4.5), qm=np.double(-1.2), ql=np.double(-3.6),
                       rho=rho_refr)

    def integrate(ti: np.float64, tf: np.float64, yinit: np.ndarray, dtinit: np.float64 = None,
                  kwargsN=kwargsN) -> tuple[np.ndarray]:
        if dtinit is None:
            # dtinit = np.double(1e-10) * (tf - ti)
            dtinit = np.double(1e-12) * (tf - ti)

        def system(t, y):
            yp = sigmap_rho(t=t - tau_diff(pp=pp, rr=rr, t=t, consts=consts) - pp * pp / consts["K"],
                            rr=rr, pp=pp, consts=consts, comp=comp).flatten()
            # yp = sigmap_rho(t=t, rr=rr, pp=pp, consts=consts, comp=comp).flatten()
            # Taking into acompte the limited amounth of ice
            yp[y > rho_ice] *= zero
            # yp[y > 0.9 * rho_ice] *= (y[y > 0.9 * rho_ice] / Myr) * np.exp(- np.maximum(200, rho_ice / np.maximum(abs(y[y > 0.9 * rho_ice] - rho_ice), 1e-200)))
            # yp[pp.flatten() < pmin] *= zero
            return yp
        #
        sol_ivp = solve_ivp(system, t_span=[ti, tf],
                            y0=yinit,
                            max_step=(tmax - ti) / Itt, method="RK45", args=[],
                            rtol=np.double(1e-4), atol=atol, first_step=dtinit)
        print(sol_ivp.message)

        sol_ivp.y = np.minimum(sol_ivp.y, rho_ice)
        res_final = sol_ivp.y[:, -1]
        times_loc = sol_ivp.t

        rhodot = np.gradient(sol_ivp.y.T.reshape((len(sol_ivp.t), pp.shape[0], pp.shape[1])), sol_ivp.t, axis=0)


        yplot = sol_ivp.y.T[-1].reshape(pp.shape).T
        yplot[np.isnan(yplot)] = zero
        yplot[abs(yplot) == np.inf] = zero
        if np.any(yplot > 0):
            yplot = np.maximum(yplot, 0.1 * min(yplot[yplot > 0]))
        else:
            yplot = np.maximum(yplot, rho_ice * 1e-50)
        if yplot.min() < yplot.max():
            print("test max : ", yplot.max() / rho_ice)
            gtest = g.image(yplot / rho_ice, r[::5] / C_L, p, colorscale="log", show=False, cmap="inferno")
            gtest.config_ax(yscale="log")
            gtest.title = r"$\rho_{lost}$"
            gtest.show()
        else:
            print(yplot.max() / rho_ice)

        rhodot_d: np.ndarray = 4. * Pi * cumulative_trapezoid(p * p * rhodot, p, axis=2, initial=zero)

        rho_d: np.ndarray = 4. * Pi * cumulative_trapezoid(p * p * (sol_ivp.y.T.reshape((len(sol_ivp.t), pp.shape[0], pp.shape[1]))),
                                                           p, axis=2, initial=zero)
        if kwargsN is None:  # Default size distribution
            kwargsN = dict(amax=np.double(1.e6) * C_L,
                           abig=np.double(120e3) * C_L,
                           amed=np.double(20.e3) * C_L, amin=C_L,
                           qh=np.double(-4.5), qm=np.double(-1.2), ql=np.double(-3.6),
                           rho=rho_refr)

        ## Calcul of number of asteroids for all diameters and distance to central star
        sigma0: np.ndarray = (Mbelt * ((r[::5] / a0) ** (-3. / 2.))
                              / (4. * Pi * a0 * a0 * (np.sqrt(1. + delta_a / a0)
                                                      - np.sqrt(1. - delta_a / a0))))  # initial surface density
        ni: np.ndarray = ((np.array([number_density(2. * d_2, Mtot=np.double(1.), **kwargsN)
                                     for d_2 in p])[:, np.newaxis] + np.zeros(len(r[::5]))).T
                          * (sigma0[:, np.newaxis] + np.zeros(len(p))))  #number density

        # Multiplying the gas production rate for one asteroid by the number of asteroids and integrated along d axis
        res: np.ndarray = np.zeros((len(times_loc), len(r[::5])), dtype="double")
        sig_lost: np.ndarray = np.zeros((len(times_loc), len(r[::5])), dtype="double")
        for i in range(1, len(times_loc)):
            res[i] = trapezoid(ni * rhodot_d[i], 2. * p)
            sig_lost[i] = trapezoid(ni * rho_d[i], 2 * p)

        mlost = trapezoid(2. * Pi * r[::5] * sig_lost, r[::5])
        print("tf = ", times_loc[-1] / Myr, " Myr. Total mass lost : ", mlost[-1] / Mearth,
              "Fraction of total mass loss ", 100 * mlost[-1] / (Mbelt * f_ice), "%")
        # g.loglog(times_loc, [mlost, minit])
        # return res, times_loc, res_final
        return np.gradient(sig_lost, times_loc, axis=0), times_loc, res_final

    times_intermediates = np.linspace(tinit, tmax, 100)[1:]
    # times_intermediates = np.linspace(tinit, tmax, 4)[1:]
    res_final = np.zeros(len(r[::5]) * len(p))
    res = []
    times = []
    ti: np.float64 = tinit
    dtinit = np.double(1e-10) * (times_intermediates[1] - ti)
    for t in times_intermediates:
        res_loc, times_loc, res_final = integrate(ti=ti, tf=t, yinit=res_final, dtinit=dtinit)
        dtinit = times_loc[-1] - times_loc[-2]
        print(res_loc.shape, times_loc.shape, len(r))
        times.extend(list(times_loc))
        res.extend(list(res_loc))
        print(len(res), res[-1].shape)
        ti = times_loc[-1]
    print("Initial gas mass : ", m_init)
    return np.array([linear_interp(r, r[::5], sig) for sig in res]), np.array(times), m_init, p


def mbelt_coll(t: np.float64, smin: np.float64 = np.double(1.e-8) * C_L,
               smax: np.float64 = np.double(4.e6) * C_L,
               sb: np.float64 = np.double(316) * C_L, qp: np.float64 = np.double(3),
               qs: np.float64 = np.double(11 / 6), qg: np.float64 = np.double(1.67), e: np.float64 = np.double(0.075),
               i: np.float64 = None, r: np.float64 = None, dr: np.float64 = np.double(4. * au),
               As: np.float64 = np.double(5.) * C_E / C_M, bs: np.float64 = np.double(-0.1),
               bg: np.float64 = np.double(0.5), M0: np.float64 = KB_Mbelt, Ms: np.float64 = Ms,
               rho_ast: np.float64 = KB_rho_refr, method_root: str = "lm",
               st_est: np.float64 | np.ndarray = None) -> np.float64 | np.ndarray:
    """
    Estimate the masse of a collisional belt as function of time (formula 38 of Löhne et al. 2008)

    Parameters
    ----------

    t : np.float64
        The time
    smin : np.float64
        Minimum size of bodys

    smax : np.float64
        Maximum size of bodys

    sb : np.float64
        Medium size

    qp : np.float64
        Primordial slope as defined in Löhne et al. 2008

    qs : np.float64
        Intermediate slope

    qg : np.float64
        Slope for bodys in collisional regim

    e : np.float64
        Eccentricity

    i : np.float64
        Inclination

    r : np.float64
        Distance of the center of the belt to the central star

    dr : np.float64
        Radial extension of the belt

    As : np.float64
        Minimal disruption energy (see formula 1 of Löhne et al. 2008)

    bs : np.float64
        Slope of the massique disruption energy (see formula 1 of Löhne et al 2008)

    bg : np.float64
        Slope of the massique disruption energy (see formula 1 of Löhne et al. 2008)

    M0 : np.float64
        Initial belt's mass

    Ms : np.float64
        Central star's mass

    rho_ast : np.float64
        Density of asteroids/KBO

    method_root : str, optional, default="lm"
        The method for the root-finding routine used to find st the size such as tau(st)=t

    st_est : np.float64
        Estimation of st

    Returns
    -------
    np.float64 | np.ndarray
        mbelt

    References
    ----------
     Löhne et al. 2008

    """
    # qp: np.float64 = np.double(1.9)
    if r is None:
        r: np.float64 = a0
    if i is None:
        i: np.float64 = e / np.double(2.)

    # dr: np.float64 = r / np.double(4.)
    # As: np.float64 = np.double(5e2) * C_E / C_M
    QDb: np.float64 = As * ((sb / C_L) ** (3. * bs) + (sb / (1e3 * C_L)) ** (3. * bg))
    f_ei: np.float64 = np.sqrt((5 / 4) * e * e + i * i)
    # Qd_smax: np.float64 = QDb * ((smax / sb) ** (3. * bg))
    Qd_smax: np.float64 = As * ((smax / C_L) ** (3. * bs) + (smax / (1e3 * C_L)) ** (3. * bg))
    XC_smax: np.float64 = (2. * Qd_smax * r / (f_ei * f_ei * G * Ms)) ** (1 / 3.)
    G_smax: np.float64 = ((XC_smax ** (5. - 3. * qg) - 1.)
                          + (2. * (qg - 5. / 3.) / (qg - 4 / 3))
                          * (XC_smax ** (4. - 3. * qg) - 1.)
                          + ((qg - 5. / 3.) / (qg - 1.))
                          * (XC_smax ** (3. - 3. * qg) - 1.))

    def tau(s: np.float64 | np.ndarray):
        """
        The collisional timescale for bodys of size s

        Parameters
        ----------
        s: np.float64 | np.ndarray
            The size
        Returns
        -------
        np.float64 | np.ndarray

        """
        Qd_s: np.float64 = As * ((s / C_L) ** (3. * bs) + (s / (1e3 * C_L)) ** (3. * bg))
        XC_s: np.float64 = (2. * Qd_s * r / (f_ei * f_ei * G * Ms)) ** (1 / 3.)
        G_qp_s: np.float64 = ((XC_s ** (5. - 3. * qp) - (smax / s) ** (5. - 3. * qp))
                              + (2. * (qp - 5. / 3.) / (qp - 4 / 3))
                              * (XC_s ** (4. - 3. * qp) - (smax / s) ** (4. - 3. * qp))
                              + ((qp - 5. / 3.) / (qp - 1.))
                              * (XC_s ** (3. - 3. * qp) - (smax / s) ** (3. - 3. * qp)))
        return ((16 * Pi * rho_ast / (3. * M0))
                * ((s / smax) ** (3. * qp - 5))
                * (smax * (r ** (5. / 2.)) * dr / np.sqrt(G * Ms))
                * ((qp - 5. / 3.) / (2 - qp))
                * (1. - ((smin / smax) ** (6. - 3. * qp)))
                * i / (f_ei * G_qp_s))

    tau_max: np.float64 = ((16. * Pi * rho_ast / (3. * M0)) * smax
                           * ((r ** (5 / 2)) * dr / np.sqrt(G * Ms))
                           * (((qg - 5. / 3.) / (2. - qp)) / (1. - ((smin / smax) ** (6. - 3. * qp))))
                           * i / (f_ei * G_smax))  # Maximum collisional timescale

    # print("tau max=", tau_max, tau(smax))

    def rech_st(s: np.float64 | np.ndarray) -> np.float64 | np.ndarray:
        """
        The function is used to find the scale for which tau(s)=t

        Parameters
        ----------
        s: np.float64 | np.ndarray
            The size to test

        Returns
        -------
        np.float64 | np.ndarray
            A parameter null where tau(s)=t
        """
        return (t - tau(s)) / t

    tau_b: np.float64 = tau(sb)
    if st_est is None:  # Estimation of st such as tau(s_t)=t
        st_est: np.float64 | np.ndarray = sb * ((t / tau_b) ** (1 / (3. * qp - 5. + 3. * (qp - 1) * bg)))
    st: np.float64 | np.ndarray = root(rech_st, st_est, method=method_root, tol=1e-12).x
    st = st[0]

    if isinstance(t, float | np.float64) and t > tau_max:
        return zero
    if t < tau_b:
        sb: np.float64 = st

    C1: np.float64 = M0 / (1. - (smin / smax) ** (6. - 3. * qp))
    C2: np.float64 = (((sb / smax) ** (6. - 3. * qp))
                      * (1. - (2. - qp) / (2. - qg)))
    C3: np.float64 = (((sb / smax) ** (6. - 3. * qp))
                      * ((2. - qp) / (2. - qs) - (2. - qp) / (2. - qg)))
    C4: np.float64 = (((sb / smax) ** (3. * qs - 3. * qp))
                      * ((smin / smax) ** (6. - 3. * qs))
                      * (2. - qp) / (2. - qs))
    alpha_st: np.float64 = np.double(1. / (3. * qp - 5 + 3. * (qp - 1.) * bg))

    res = C1 * (1 - C2 * ((st / sb) ** (6. - 3. * qp)) + ((st / sb) ** (3. * qg - 3. * qp)) * (C3 - C4)) / (
            1 + t / tau_max)

    if isinstance(t, np.ndarray):
        res[t > tau_max] *= zero

    return res


class SolrhoBelt:
    def __init__(self,
                 sublimation_model: str,
                 Mbelt: np.float64,
                 Ms: np.float64,
                 a0: np.float64, delta_a: np.float64,
                 tinit: np.float64, tmax: np.float64,
                 rho_refr: np.float64, rho_ice: np.float64, f_ice: np.float64,
                 A_ast: np.float64, K: np.float64,
                 Its: int, Phi: np.float64,
                 dmin: np.float64, sb: np.float64, dmax: np.float64,
                 e: np.float64, i: np.float64, As: np.float64, bs: np.float64, bg: np.float64,
                 Its_p: int, rp: np.float64, Itt: int,
                 t0_diss: np.float64, t1_diss: np.float64,
                 radius: np.ndarray[np.float64] | None = None,
                 kwargs_N: dict = None, sig_dots: np.ndarray = None, tps: np.ndarray = None,
                 m_init: np.float64 = zero,
                 const_mdot: np.float64 = zero, **kwargs
                 ):
        """
        Initialise a SolrhoBelt object
        This object is used to get the gas surface density generation rate thru its method_root
        interp_sig_dot(t)

        Parameters
        ----------
        sublimation_model : str, {"none", "thermal_full", "constant_rate"}
            The model use to estimate the gas generation rate :
                - none : No gas is produced
                - thermal_full : the gas is produced by sublimation : all sublimation, thermal diffusion and molecular
                 diffusion are take into acompte
                - constant_rate : The gas is produced at a constant rate of const_mdot
        Mbelt : np.float64
            The total belt's mass
        Ms : np.float64
            The central star's mass
        a0 : np.float64
            The belt's semi major axis (located at the middle of the belt)
        delta_a : np.float64
            The belt's (half) size
        tinit : np.float64
            The initial time
        tmax : np.float64
            The final time
        rho_refr : np.float64
            The density of refractory materials
        rho_ice : np.float64
            The ice's density
        f_ice : np.float64
            The ice to refractory mass ratio
        A_ast : np.float64
            Asteroids/KBO albedo
        K : np.float64
            Thermal diffusion coefficient
        Its : np.float64
            Number of spatial iterations
        Phi : np.float64
            Porosity (dimensionless number between 0 to 1)
        dmin : np.float64
            The minimum radius of asteroids/KBO
        sb : np.float64
            The final transition size between collisional and primordial regim of asteroids/KBO
        dmax : np.float64
            The maximum radius of asteroids/KBO
        e : np.float64
            Eccentricity
        i : np.float64
            Inclinaison
        As : np.float64
            Minimal disruption energy (see formula 1 of Löhne et al 2008)
        bs : np.float64
            Slope of the massique disruption energy (see formula 1 of Löhne et al 2008)
        bg : np.float64
            Slope of the massique disruption energy (see formula 1 of Löhne et al 2008)
        Its_p : np.float64
            The number of steps in diameters space
        rp : np.float64
            The pore's radius to estimate the gas diffusion thru the solid
        Itt : np.float64
            Number of spatial iterations
        t0_diss : np.float64
            Belt's lifetime before dissipation
        t1_diss : np.float64
            Dissipation timescale
        radius : np.ndarray
            Table of radius
        kwargs_N : dict
            keywords arguments for the size distribution of asteroids/KBO
        sig_dots : np.ndarray
            table of sigma dot as function of time and radius (shape=(Ttt, Its)) if it has already been calculated
            This table is interpolated with the CubicSpline function to get interp_sig_dot(t)
        const_mdot : np.float64, optional, default=0.
            The mass generation rate for the `constant_rate` sublimation model

        """
        self.sublimation_model: str = sublimation_model
        self.const_mdot: np.float64 = const_mdot
        self.Mbelt: np.float64 = Mbelt
        self.Ms: np.float64 = Ms
        self.Abelt: np.float64 = A_ast
        self.K: np.float64 = K
        self.a0: np.float64 = a0
        self.delta_a: np.float64 = delta_a
        self.t0: np.float64 = tinit
        self.tmax: np.float64 = tmax
        self.rho_refr: np.float64 = rho_refr
        self.rho_ice: np.float64 = rho_ice
        self.Its: int = Its
        self.Its_p: int = Its_p
        self.Itt: int = Itt
        self.f_ice: np.float64 = f_ice  # ice to refractory mass ratios
        self.Phi: np.float64 = Phi  # Porosity
        self.rp: np.float64 = rp  # Pore's radius
        self.dmin: np.float64 = dmin
        self.sb: np.float64 = sb
        self.dmax: np.float64 = dmax
        self.e: np.float64 = e
        self.i: np.float64 = i
        self.As: np.float64 = As
        self.bs: np.float64 = bs
        self.bg: np.float64 = bg
        self.m_init: np.float64 = m_init
        if radius is None:
            self.r = np.linspace(a0 - delta_a, a0 + delta_a, Its)
        else:
            self.r = np.copy(radius)
            self.Its = len(self.r)
        # self.p = np.linspace(dmin / 2., dmax / 2., Its_p)
        self.p = np.geomspace(dmin / 2., dmax / 2., Its_p)
        if tps is None:
            self.tps: np.ndarray = tinit + np.geomspace(yr, tmax - tinit, Itt)
        else:
            self.tps: np.ndarray = tps

        if kwargs_N is None:
            self.kwargs_N = dict(amax=np.double(1.e6) * C_L,
                                 abig=np.double(120e3) * C_L,
                                 amed=np.double(20.e3) * C_L, amin=C_L,
                                 qh=np.double(-4.5), qm=np.double(-1.2), ql=np.double(-3.6),
                                 rho=self.rho_refr)
        else:
            self.kwargs_N = kwargs_N.copy()
        self.sig_dots = None

        if sig_dots is not None:
            self.sig_dots = sig_dots
            # print("test :", len(self.tps), sig_dots.shape)
            tps, idx = np.unique(self.tps, return_index=True)
            self.tps = np.copy(tps)
            if self.sig_dots.shape[0] == len(self.tps):
                self.sig_dots = (self.sig_dots[idx]).copy()
            else:
                self.sig_dots = (self.sig_dots.T[idx]).copy()

            def linear_interp(t_new):
                if isinstance(t_new, list | np.ndarray):
                    return np.array([linear_interp(t) for t in t_new])

                i = np.argmin(np.abs(t_new - np.array(self.tps)))
                if i == 0 or (i < len(self.tps) - 1 and t_new > self.tps[i]):
                    t1 = self.tps[i]
                    t2 = self.tps[i + 1]
                    alpha = (t_new - t1) / (t2 - t1)
                    return (1 - alpha) * self.sig_dots[i] + alpha * self.sig_dots[i + 1]
                else:
                    t1 = self.tps[i - 1]
                    t2 = self.tps[i]
                    alpha = (t_new - t1) / (t2 - t1)
                    return (1 - alpha) * self.sig_dots[i - 1] + alpha * self.sig_dots[i]

            self.interp_sig_dot = linear_interp
            # self.interp_sig_dot = linear_interpolation(self.tps, self.sig_dots)
            
            # def interp_sig_dot(t: np.float64 | np.ndarray) -> np.ndarray[np.float64]:
            #     return np.interp(t, self.tps, self.sig_dots)
            # self.interp_sig_dot = interp_sig_dot
            # self.interp_sig_dot = linear_interpolation(self.tps, sig_dots)
            # self.interp_sig_dot = PchipInterpolator(self.tps, sig_dots)
            # self.interp_sig_dot = CubicSpline(self.tps - 0.1 * Myr, sig_dots)
            # self.interp_sig_dot = FloaterHormannInterpolator(self.tps, sig_dots)
            # self.interp_log_sig_dot = CubicSpline(self.tps, np.maximum(np.log10(self.sig_dots), 1e-200))
            #
            # def interp_sig_dot(t: np.float64 | np.ndarray) -> np.ndarray[np.float64]:
            #     return 10 ** self.interp_log_sig_dot(t)
            #
            # self.interp_sig_dot = interp_sig_dot
        elif self.sublimation_model == "none":
            def null(t: np.float64 | np.ndarray) -> np.float64:
                return zero

            self.interp_sig_dot = null

        elif self.sublimation_model == "thermal_full":
            # self.sig_dots, self.tps, self.m_init, self.p = sigma_dot(r=self.r, tps=self.tps, kwargsN=self.kwargs_N, **self.const())
            # print("test shape", self.sig_dots.shape, self.r.shape, self.tps.shape)
            # M = trapezoid(trapezoid(2. * np.pi * self.r * self.sig_dots, self.r, axis=1), self.tps)
            # print("Test mass", M / Mearth)
            # # while M > 1.1 * self.Mbelt * self.f_ice:
            # #     self.Itt = int(self.Itt * 1.2)
            # #     print("Integration failure, restarting", self.Itt)
            # #     self.sig_dots, self.tps = sigma_dot(r=self.r, p=self.p, tps=self.tps, kwargsN=self.kwargs_N,
            # #                                         **self.const())
            # #     M = trapezoid(trapezoid(2. * np.pi * self.r * self.sig_dots, self.r), self.tps)
            # # self.sig_dots = np.maximum(savgol_filter(self.sig_dots, 7, 3, axis=1), zero)
            # # dissipation of the belt
            # # if np.any(self.tps - 0.1 * Myr > t0_diss):
            # #     self.sig_dots[self.tps - 0.1 * Myr > t0_diss] *= np.exp(
            # #         -(self.tps[self.tps - 0.1 * Myr > t0_diss] - t0_diss) / t1_diss)
            # tps_u, idx = np.unique(self.tps, return_index=True)
            # self.tps = tps_u.copy()
            # self.Itt = len(self.tps)
            # self.sig_dots = (self.sig_dots[idx]).copy()

            self.sig_dots =  sig_dot_full_diff(tps=self.tps, r=self.r, const=self.const(),
                                               kwargsN=self.kwargs_N, It_d=200)

            def linear_interp(t_new):
                if isinstance(t_new, list | np.ndarray):
                    return np.array([linear_interp(t) for t in t_new])

                i = np.argmin(np.abs(t_new - np.array(self.tps)))
                if i == 0 or (i < len(self.tps) - 1 and t_new > self.tps[i]):
                    t1 = self.tps[i]
                    t2 = self.tps[i + 1]
                    alpha = (t_new - t1) / (t2 - t1)
                    return (1 - alpha) * self.sig_dots[i] + alpha * self.sig_dots[i + 1]
                else:
                    t1 = self.tps[i - 1]
                    t2 = self.tps[i]
                    alpha = (t_new - t1) / (t2 - t1)
                    return (1 - alpha) * self.sig_dots[i - 1] + alpha * self.sig_dots[i]

            self.interp_sig_dot = linear_interp

            # self.interp_sig_dot = linear_interpolation(self.tps, self.sig_dots)
            # self.interp_sig_dot = PchipInterpolator(self.tps, self.sig_dots)
            # self.interp_sig_dot = CubicSpline(self.tps - 0.1 * Myr, self.sig_dots)
            # self.interp_log_sig_dot = CubicSpline(self.tps, np.maximum(np.log10(self.sig_dots), 1e-200))
            #
            # def interp_sig_dot(t: np.float64 | np.ndarray) -> np.ndarray[np.float64]:
            #     return 10 ** self.interp_log_sig_dot(t)
            # self.interp_sig_dot = interp_sig_dot
            # self.interp_sig_dot = FloaterHormannInterpolator(self.tps, self.sig_dots)
            # selot = CubicSpline(self.tps, self.sig_dots)
        elif self.sublimation_model == "collisions":

            rho = rho_refr
            # with f_ice = rho_ice / rho

            mbelt_: np.ndarray = np.array([(mbelt_coll(t, smax=self.dmax, smin=self.dmin,
                                                       sb=self.sb, qp=np.double((-self.kwargs_N["qh"] + 2.) / 3.),
                                                       qs=np.double((-self.kwargs_N["ql"] + 2.) / 3.),
                                                       qg=np.double((-self.kwargs_N["qm"] + 2.) / 3.), e=self.e,
                                                       i=self.i, r=self.a0, dr=self.delta_a, As=self.As,
                                                       M0=self.Mbelt, rho_ast=rho,
                                                       method_root="lm", bs=self.bs, bg=self.bg)) for t in self.tps])

            mdot_: np.ndarray = - self.f_ice * np.gradient(mbelt_, self.tps - 0.1 * Myr)
            # dissipation of the belt
            if np.any(self.tps - 0.1 * Myr > t0_diss):
                mdot_[self.tps - 0.1 * Myr > t0_diss] *= np.exp(
                    -(self.tps[self.tps - 0.1 * Myr > t0_diss] - t0_diss) / t1_diss)
            sigmad0: np.float64 = mdot_ / (
                        4. * Pi * a0 * a0 * (np.sqrt(1. + delta_a / a0) - np.sqrt(1. - delta_a / a0)))
            self.sig_dots: np.ndarray = np.full((len(self.tps), len(self.r)), sigmad0)
            self.sig_dots *= ((self.r / self.a0) ** (-3. / 2.))
            self.interp_sig_dot = CubicSpline(self.tps - 0.5 * Myr, self.sig_dots)

        elif self.sublimation_model == "thermal_diff":
            # The sublimation timescale and molecular diffusion timescale are neglected
            def n(diameter: np.float64):
                return number_density(diameter, **self.kwargs_N)

            rho = rho_refr

            def sig_dot_therm(t: np.float64):
                t0 = 1e3 * yr

                mdot_t = (2. * Pi * rho * self.f_ice * (self.K ** (3 / 2))
                          * quad(n, max(self.kwargs_N["amin"], 2. * np.sqrt(self.K * (t + t0))),
                                 self.kwargs_N["amax"])[0] * np.sqrt(t))
                if np.isnan(mdot_t):
                    mdot_t = zero
                elif mdot_t < 0:  # the integration with quad has not worked
                    diams = np.geomspace(max(self.kwargs_N["amin"], 2. * np.sqrt(self.K * (t + t0))),
                                         self.kwargs_N["amax"], 1000)
                    mdot_t = (2. * Pi * rho * f_ice * (self.K ** (3 / 2))
                              * trapezoid([n(d) for d in diams], diams) * np.sqrt(t))
                    if mdot_t < 0:  # the integration with quad has not worked
                        diams = np.geomspace(max(self.kwargs_N["amin"], 2. * np.sqrt(self.K * (t + t0))),
                                             self.kwargs_N["amax"], 10000)
                        mdot_t = (2. * Pi * rho * f_ice * (self.K ** (3 / 2))
                                  * trapezoid([n(d) for d in diams], diams) * np.sqrt(t))

                sigmad0: np.float64 = mdot_t / (
                            4. * Pi * a0 * a0 * (np.sqrt(1. + delta_a / a0) - np.sqrt(1. - delta_a / a0)))
                if t > t0_diss:
                    sigmad0 *= np.exp(-(t - t0_diss) / t1_diss)
                return sigmad0 * ((self.r / self.a0) ** (-3. / 2.))

            self.interp_sig_dot = sig_dot_therm

        elif self.sublimation_model == "constant_rate":
            # norm_gauss = quad(lambda x: np.exp(-x * x / 2.), 0, 4.)[0]
            # sigmar: np.float64 = np.double(self.delta_a / (2. * np.sqrt(2. * np.log(2))))
            # sigmap_0: np.float64 = m_dot / (4. * Pi * sigmar * self.a0 * norm_gauss)
            # sigmap: np.ndarray = sigmap_0 * np.exp(- (self.r - self.a0) * (self.r - self.a0) / (
            #         2. * sigmar * sigmar))
            # sigmap_0: np.float64 = m_dot / (4. * Pi * self.a0  * self.a0 * (
            #         np.sqrt(1 + self.delta_a / (2 * self.a0)) - np.sqrt(1 - self.delta_a / (2 * self.a0))))
            sigmap_0: np.float64 = const_mdot / (4. * Pi * self.a0  * self.a0 * (
                    np.sqrt(1 + self.delta_a / self.a0) - np.sqrt(1 - self.delta_a / self.a0)))
            sigmap: np.ndarray = sigmap_0 * ((self.r / self.a0) ** (-3/2))
            def cons_rate(t: np.float64 | np.ndarray) -> np.ndarray:
                if t < t0_diss:
                    return sigmap
                else:
                    return sigmap * np.exp(-max((t - t0_diss) / t1_diss, 150))

            self.interp_sig_dot = cons_rate

        else:
            raise UserWarning("The sublimation model ", self.sublimation_model, " is not implemented (yet)")

    def const(self, prefix: str = "") -> dict:
        return {prefix + "sublimation_model": self.sublimation_model,
                prefix + "tinit": self.t0, prefix + "tmax": self.tmax,
                prefix + "Mbelt": self.Mbelt,
                prefix + "Ms": self.Ms, prefix + "Abelt": self.Abelt, prefix + "K": self.K,
                prefix + "a0": self.a0, prefix + "delta_a": self.delta_a, prefix + "t0": self.t0,
                prefix + "rho_refr": self.rho_refr, prefix + "rho_ice": self.rho_ice,
                prefix + "Its": self.Its, prefix + "Its_p": self.Its_p,
                prefix + "f_ice": self.f_ice,
                prefix + "Phi": self.Phi, prefix + "rp": self.rp,
                prefix + "Itt": self.Itt,
                prefix + "dmin": self.dmin,
                prefix + "dmax": self.dmax,
                }

    def return_kwargsN(self, prefix: str = "") -> dict:
        res = {}
        for k in self.kwargs_N.keys():
            res[prefix + k] = self.kwargs_N[k]
        return res


def thermal_diff(d: np.float64, consts: dict, It_d: int = 300, t: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the thermal diffusion equation into a bodi of radius d as a function of time assuming the bodi to be
    at the distance a_0 from the central star. To get the temperatures at others location r, one must multiplie the
    result by (r / a0) ** (-1/2)

    Parameters
    ----------
    d : np.float64
        The size of the bodi
    consts : dict
        The dictionary of constant parameters
    It_d : int
        The number of spatials steps
    t : np.ndarray
        the times at which evaluate the solution

    Returns
    -------
    np.ndarray

    """
    r = np.linspace(d / (100 * It_d), d, It_d)
    dr = r[1] - r[0]

    def system(t: np.float64, T: np.ndarray, compteur: Compteur = Compteur()):
        T[-1] = (278 * (((1. - consts["Abelt"]) * interpLbol(t, consts["Ms"]) / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))
        # T[0] = 1
        T[0] = T[1]
        # dT = consts["K"] * (gradient2(T, dr) + np.gradient(T, r) / r) - T * interp_L(t, 1) / (4 * interp_L(t))
        dT = consts["K"] * (gradient2(T, dr) + np.gradient(T, r) / r)
        dT[0] = 0
        dT[-1] = 0
        if compteur.value > 0 and compteur.value % 10000 == 0:
            print("Iteration ", compteur.value, "t={:.4f} Myr".format(t / Myr))
        compteur.value += 1
        return dT

    Tinit = np.zeros(It_d) + 0.1
    # Tinit[0] = 1
    compt: Compteur = Compteur()
    sol_ivp = solve_ivp(system, t_span=[consts["tinit"], consts["tmax"]],
                        y0=Tinit, t_eval=t,
                        max_step=Myr, method="LSODA", args=[compt],
                        rtol=np.double(1e-6), atol=atol)
    if sol_ivp.status == -1:
        print(sol_ivp.message)
        if t is None:
            t = np.geomspace(consts["tinit"], consts["tmax"], 1000)
        temp_lim = (278 * (((1. - consts["Abelt"]) * interpLbol(t, consts["Ms"]) / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))
        return r, t, (temp_lim[:, np.newaxis] + np.zeros_like(r)).T
    sol_ivp.y[-1] = (278 * (((1. - consts["Abelt"]) * interpLbol(sol_ivp.t, consts["Ms"]) / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))
    sol_ivp.y[0] = sol_ivp.y[1]
    return r, sol_ivp.t, sol_ivp.y


def mdot_s_l(rd: np.ndarray, t: np.ndarray, temp: np.ndarray, consts: dict) -> np.ndarray:
    """
    Compute the gas mass generation for a given bodie

    Parameters
    ----------
    rd : np.ndarray
        The radial coordinates inside the asteroid / KBO
    t : np.ndarray
        The time
    temp : np.ndarray
        The array of temperature as function of rd and t calculated by `thermal_diff`
    consts: dict
        The dictionary of constants

    Returns
    -------
    np.ndarray
        the gas mass generation rate as function of time

    """

    # mask = np.cumsum(temp > level, axis=1)
    # mask[mask > 1] = 0
    # mask = np.array(mask, dtype=bool)
    #
    # rr = rd[:, np.newaxis] + np.zeros(len(t))
    # dr = rd[1] - rd[0]
    # mlost = 4 * np.pi * consts["rho_ice"] * ((rr + dr / 2) ** 3 - (rr - dr / 2) ** 3) / 3
    # mlost[~mask] = zero
    # mlost = mlost.sum(axis=0)
    # if np.sum(mlost > 0) > 1:
    #     return np.interp(t, t[mlost > 0], np.gradient(np.cumsum(mlost[mlost > 0]), t[mlost > 0]))
    # elif np.any(mlost > 0):
    #     res = np.zeros_like(t)
    #     i = np.argwhere(mlost > 0)[0, 0]
    #     if i < len(t) - 1:
    #         res[i] = mlost[i] / (t[i + 1] - t[i])
    #     else:
    #         res[i] = mlost[i] / (t[i] - t[i-1])
    #     return res
    # else:
    #     # print("mdot_s_l, dmax={:.4f} m, Tsub={:.4f} K, no ice sublimation. Tmax={:.4f} K".format(rd.max() / C_L, level, np.max(temp)))
    #     return np.zeros_like(t)

    S: np.float64 = 3. * (1. - consts["Phi"]) / consts["rp"]
    dr = rd[1] - rd[0]
    rr = rd[:, np.newaxis] + np.zeros(len(t))
    mdot = (4 * Pi / 3.) * (S * Ah2o * np.exp(-Bh2o / temp) * np.sqrt(mu_H2O / (2. * Pi * Rgp * temp))) * (
                (rr + dr) ** 3 - (rr - dr) ** 3)
    mlost_max = (4 * Pi / 3.) * consts["rho_ice"] * ((rr + dr) ** 3 - (rr - dr) ** 3)
    mdot[cumulative_trapezoid(mdot, t, initial=0., axis=1) >= mlost_max] *= zero
    mmax = (4 * Pi / 3.) * consts["rho_ice"] * (rd.max() ** 3 - rd.min() ** 3)
    if np.any(cumulative_trapezoid(mdot.sum(axis=0), t, initial=0.) >= mmax):
        i = np.argwhere((cumulative_trapezoid(mdot.sum(axis=0), t, initial=0.) >= mmax))[0, 0]
        mdot[:, max(0, i-1):] *= zero
    # print("test : ", t.max(), " K", trapezoid(mdot.sum(axis=0), t) / mmax)
    return mdot.sum(axis=0)
    # mlost = np.minimum(cumulative_trapezoid(mdot, t, initial=0., axis=1), mlost_max)
    # mmax =  (4 * Pi / 3.) * consts["rho_ice"] * (rd.max() ** 3 - rd.min() ** 3)
    # print("test : ",rd.max() / C_L, trapezoid(np.gradient(mlost.sum(axis=0), t), t) / mmax, (mlost < mlost_max).sum() / len(mlost_max.flatten()))
    # return np.gradient(mlost.sum(axis=0), t)


def mdot_s(s: np.float64, t: np.ndarray, r: np.ndarray, consts: dict, It_d=300) -> np.ndarray:
    """
    Compute the gas mass generation rate for a given bodie at differents radius

    Parameters
    ----------
    s: np.float64
        The size of the considered bodie
    t: np.ndarray
        The time at which evaluate the gas mass generation rate
    r: np.ndarray
        The distances from the central star at which evaluate the gas mass generation rate
    consts : dict
        The dictionary of constants
    It_d : int
        The number of iterations

    Returns
    -------
    np.ndarray
        the gas mass generation rate as function of time and radius

    """
    print("s={:.4f}".format(s / C_L), "m")

    rd, t, temp = thermal_diff(d=s / 2, consts=consts, It_d=It_d, t=t)
    # gr=g.image(temp, t, rd, cmap="inferno", colorscale="log", show=False)
    # gr.contours(levels, color="w")
    # gr.show()
    # return np.array([mdot_s_l(rd=rd, t=t, temp=temp, level=l, consts=consts) for l in levels])
    return np.array([mdot_s_l(rd=rd, t=t, temp=temp * (abs(R) / consts["a0"]) ** (-1 / 2),
                              consts=consts) for R in r])
    # mlost_max = (4 * Pi / 3.) * consts["rho_ice"] * ((s / 2) ** 3)
    # return np.array([np.ones_like(t) * mlost_max / (t.max() - t.min()) for R in r])


def sig_dot_full_diff(tps: np.ndarray, r: np.ndarray, const: dict, kwargsN: dict = None, It_d: int = 100) -> np.ndarray:
    """
    Compute the gas surface density generation rate for a belt discribed into the const dictionary

    Parameters
    ----------
    tps : np.ndarray
        The time at which evaluate the result
    r : np.ndarray
        The distance to the central star at which evaluate the solution
    const : dict
        The dictionary of constants
    kwargsN : dict
        The dictionary that defined the size distribution
    It_d : int
        The number of depth steps for thermal diffusion integration

    Returns
    -------
    np.ndarray

    """

    sizes = np.geomspace(const["dmin"], const["dmax"], const["Its_p"])

    mdots = np.array([mdot_s(s=s, t=tps, r=r, consts=const, It_d=It_d) for s in sizes])
    # The gas mass generation rates of each bodies's size for each radius for each time

    if kwargsN is None:  # Default size distribution
        kwargsN = dict(amax=np.double(1.e6) * C_L,
                       abig=np.double(120e3) * C_L,
                       amed=np.double(20.e3) * C_L, amin=C_L,
                       qh=np.double(-4.5), qm=np.double(-1.2), ql=np.double(-3.6),
                       rho=np.double(3e3 * C_M / (C_L * C_L * C_L)))

    ## Calcul of number of asteroids for all diameters and distance to central star
    sigma0: np.ndarray = (const["Mbelt"] * ((r / const["a0"]) ** (-3. / 2.))
                          / (4. * Pi * const["a0"] * const["a0"] * (np.sqrt(1. + const["delta_a"] / const["a0"])
                                                  - np.sqrt(1. - const["delta_a"] / const["a0"]))))  # initial surface density
    ni: np.ndarray = ((np.array([number_density(s, Mtot=np.double(1.), **kwargsN)
                                 for s in sizes])[:, np.newaxis] + np.zeros(len(r)))
                      * (sigma0[:, np.newaxis] + np.zeros(len(sizes))).T)  # number density
    ni: np.ndarray = ni[..., np.newaxis] + np.zeros(mdots.shape[-1])
    print("ni max={:.4f} min={:.4f}".format(ni.max(), ni.min()))
    print("mdots max={:.4f} mdots={:.4f}".format(mdots.max(), mdots.min()))
    return trapezoid(ni * mdots, sizes, axis=0).T
