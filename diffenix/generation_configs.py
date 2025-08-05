import os.path

import numpy as np
from scipy.optimize import root
from wx.py.editor import directory

from diffenix.Constantes import *

dic_ref: dict = {"racine": racine, "results": results,
                 "C_L": C_L, "C_M": C_M, "C_t": C_t,
                 "Ms": Msun, "Ts": np.double(5500.), "Rs": Rsun, "Ls": Lsun,
                 "a_in": .05 * au, "a_out": 10 * a0, "a0": 2.65 * au,
                 "T0": T0, "pls_temp": np.double(-1/2), "mH_init": zero,
                 "alpha": np.double(1e-3), "gamma": np.double(1.),
                 "AB_mdot_st_init": np.double(1e-10) * Mearth / Myr,
                 "AB_Mbelt": np.double(0.12) * Mearth,
                 "AB_Abelt": np.double(0.06),
                 "AB_a0": np.double(2.65) * au, "AB_delta_r": np.double(2.65 / 2.) * au,
                 "AB_K": np.double(1e-5 * C_L * C_L / C_t), "AB_Phi": np.double(0.6), "AB_rp": np.double(1e-6 * C_L),
                 "AB_f_ice": np.double(0.2), "AB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)) ,
                 "AB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "AB_sublimation_model": "thermal_full",
                 "AB_const_mdot": zero,
                 "AB_SD_amax": np.double(1.e6) * C_L, "AB_SD_abig": np.double(120e3) * C_L,
                 "AB_SD_amed": np.double(20.e3) * C_L, "AB_SD_amin": C_L,
                 "AB_SD_qh": np.double(-4.5), "AB_SD_qm": np.double(-1.2), "AB_SD_ql": np.double(-3.6),
                 "AB_sb":  np.double(316) * C_L, "AB_e": np.double(0.075), "AB_i": np.double(0.075 / 2.),
                 "AB_As": np.double(5.) * C_E / C_M , "AB_bs": np.double(-0.1), "AB_bg": np.double(0.5),
                 "AB_t0_diss": 150 * Myr, "AB_t1_diss": Myr,
                 "KB_mdot_st_init": zero,
                 "KB_Mbelt": np.double(0.01 * Mearth), "KB_Abelt": np.double(0.06),
                 "KB_a0": np.double(44) * au, "KB_delta_r": np.double(8) * au,
                 "KB_K": np.double(1e-10 * C_L * C_L / C_t), "KB_Phi": np.double(0.6), "KB_rp": np.double(1e-6 * C_L),
                 "KB_f_ice": np.double(0.2), "KB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                 "KB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "KB_sublimation_model": "none",
                 "KB_const_mdot": zero,
                 "KB_SD_amax": np.double(1.e6) * C_L, "KB_SD_abig": np.double(120e3) * C_L,
                 "KB_SD_amed": np.double(20.e3) * C_L, "KB_SD_amin": C_L,
                 "KB_SD_qh": np.double(-4.5), "KB_SD_qm": np.double(-1.2), "KB_SD_ql": np.double(-3.6),
                 "KB_sb":  np.double(316) * C_L, "KB_e": np.double(0.075), "KB_i": np.double(0.075 / 2.),
                 "KB_As": np.double(5.) * C_E / C_M, "KB_bs": np.double(-0.1), "KB_bg": np.double(0.5),
                 "KB_t0_diss": np.double(10.) * Gyr, "KB_t1_diss": np.double(10.) * Gyr,
                 "a_planets": np.array([au]), "m_planets": np.array([Mearth]), "f_accr": np.double(0.5),
                 "t0": np.double(5. * Myr),
                 "IT_s": 3000, "dip": 10, "rtol": np.double(1e-6), "atol": np.double(1e-250), "method_root": 'RK45',
                 }


def temp_ppd(r: np.ndarray[np.float64] | np.float64,
             consts: dict, mdot: np.float64 = zero
             ) -> np.float64 | np.ndarray[np.float64]:
    """
    The temperature of the protoplanetary disc
    Parameters
    ----------
    r : np.float64 | np.ndarray[np.float64]
        The distance to the central star
    consts : dict
        All the constants needed
    mdot : np.float, optional, default=0
        The accretion rate on the central star

    Returns
    -------
    np.float64 | np.ndarray[np.float64]
        The temperature of the protoplanetary disc at the dissipation time

    """
    C: np.float64 = 5  # Ratio between h the heigh at wich the disc became optically thick and Hz the typical scale-heigh
    # asssumed to be a constant
    # primordial disc protertys
    M0 = 0.01 * Msun # total disc mass
    r0 = 10 * au # reference radius
    rout = 200 * au # outer radius
    rin = 0. #inner radius
    kappa_r = C_L * C_L / C_M  # Rosseland opacity
    sigma0 = M0 / (2. * Pi * au * (r0 - rin + 2 * np.sqrt(au) * (np.sqrt(rout) - np.sqrt(r0))))
    tau = np.zeros_like(r)
    if isinstance(r, np.ndarray):
        tau[r < r0] = kappa_r * sigma0 * (au / r[r < r0])
        tau[r >= r0] = kappa_r * sigma0 * ((r[r >= r0] / au) ** (-3 / 2))
    elif r < r0:
        tau[r < r0] = kappa_r * sigma0 * (au / r)
    else:
        tau[r < r0] = kappa_r * sigma0 * ((r / au) ** (-3 / 2))

    return ((0.2 * interpRs(consts["t0"], consts["Ms"]) / r
            + (C / 7) * ((Rgp * consts["Rs"] / (G * consts["Ms"] * 2.33 * 1e-3 * C_M)) ** (4 / 7))
            * ((interpLbol(consts["t0"], consts["Ms"]) / (4. * Pi * sigma * interpRs(consts["t0"], consts["Ms"])
                                                          - interpRs(consts["t0"], consts["Ms"]))) ** (1 / 7))
            * (r / interpRs(consts["t0"], consts["Ms"])) ** (2 / 7))
            * (interpLbol(consts["t0"], consts["Ms"]) / (4. * Pi * sigma * r * r))
            + ((3 / 4) * (tau + 2 / 3))
            * 3 * G * consts["Ms"] * mdot * (1. - np.sqrt(interpRs(consts["t0"], consts["Ms"]) / r))
            / (8. * Pi * sigma * r * r * r)) ** (1 / 4)


def ice_line(consts: dict, mdot: np.float64 = zero) -> np.float64:
    """
    Return the lacation of the iceline when the protoplanetary disc dissipate

    Parameters
    ----------
    consts : dict
        All the constants needed
    mdot : np.float, optional, default=0
        The accretion rate on the central star

    Returns
    -------
    np.float64
        the ice-line location
    """
    def null(r: np.float64 | np.ndarray[np.float64]) -> np.ndarray[np.float64] | np.float64:
        return temp_ppd(r, consts=consts, mdot=mdot) - 170
    return root(null, x0 = au).x

directory: str = "./configs"

if not os.path.exists(directory):
    os.mkdir(directory)
mstars = np.linspace(wfus["M"].min(), wfus["M"].max(), 5) * Msun
mds = [0.1 * Mearth, 1e-3 * Mearth]
tinits = [Myr, np.double(5.) * Myr]
alphas = [1e-3, 1e-1]

for im in range(len(mstars)):
    for imd in range(len(mds)):
        for it in range(len(tinits)):
            consts_ms = dic_ref.copy()
            consts_ms["Ms"] = mstars[im]
            consts_ms["t0"] = tinits[it]
            a0: np.float64 = ice_line(consts_ms, 1e-10 * Msun / yr)
            a0s = [0.7 * a0, a0, 1.3 * a0]
            for ialpha in range(len(alphas)):
                for ia0 in range(len(a0s)):
                    consts_msi = consts_ms.copy()
                    consts_msi["AB_Mbelt"] = mds[imd]
                    consts_msi["alpha"] = alphas[ialpha]
                    consts_msi["a0"] = a0s[ia0]
                    consts_msi["a_planets"] = 0.7 * a0s[ia0] * np.double(1 - 0.5)
                    consts_msi["a_out"] = np.double(100.) * a0s[ia0]
                    consts_msi["AB_a0"] = a0s[ia0]
                    consts_msi["AB_delta_r"] = a0s[ia0] * np.double(0.5)
                    consts_msi["Simulation"] = "Simulation_" + str(im + 1) + str(imd + 1) + str(it + 1) + str(ialpha + 1) + str(ia0 + 1)
                    print(consts_msi["Simulation"])
                    np.savez_compressed(directory + "/" + consts_msi["Simulation"], **consts_msi)
