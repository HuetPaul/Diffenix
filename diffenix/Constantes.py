from __future__ import annotations

# This file contains all globals constants needed

import os
import numpy as np
from diffenix.chemains import racine, results, data_leiden, data_stars
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator, CubicSpline

ii = np.iinfo(int)  # Informations sur les results possibles de la

ii_max = ii.max  # np.iinfo(int).max the maximum integer possible

if os.path.exists(racine):
    if "interrupteur.npz" not in os.listdir(results):
        np.savez_compressed(results + "/interrupteur.npz")
    if not os.path.exists(results + "/informations.csv"):
        print("Genrating new informations.csv file")
        # d: dict = dict(simulation=np.empty(1), C_L=np.empty(1), C_M=np.empty(1), C_t=np.empty(1),
        #                Me=np.empty(1), gamma=np.empty(1), IT_s=np.empty(1), méthode=np.empty(1), Rad=np.empty(1))
        d: dict = dict(Simulation=np.empty(1), C_L=np.empty(1),
                       C_M=np.empty(1), C_t=np.empty(1),
                       Ms=np.empty(1), Ts=np.empty(1), Rs=np.empty(1), Ls=np.empty(1),
                       a_in=np.empty(1), a_out=np.empty(1), a0=np.empty(1),
                       T0=np.empty(1), pls_temp=np.empty(1), m_dot=np.empty(1),
                       alpha=np.empty(1), gamma=np.empty(1),
                       AB_Mbelt=np.empty(1), AB_Abelt=np.empty(1),
                       AB_a0=np.empty(1), AB_delta_r=np.empty(1),
                       AB_K=np.empty(1), AB_Phi=np.empty(1), AB_rp=np.empty(1),
                       AB_f_ice=np.empty(1), AB_rho_refr=np.empty(1),
                       AB_rho_ice=np.empty(1), AB_sublimation_model=np.empty(1),
                       AB_SD_amax=np.empty(1), AB_SD_abig=np.empty(1),
                       AB_SD_amed=np.empty(1), AB_SD_amin=np.empty(1),
                       AB_SD_qh=np.empty(1), AB_SD_qm=np.empty(1), AB_SD_ql=np.empty(1),
                       AB_sb=np.empty(1), AB_e=np.empty(1), AB_i=np.empty(1),
                       AB_As=np.empty(1), AB_bs=np.empty(1), AB_bg=np.empty(1), AB_t0_diss=np.empty(1),
                       AB_t1_diss=np.empty(1), KB_Mbelt=np.empty(1), KB_Abelt=np.empty(1),
                       KB_a0=np.empty(1), KB_delta_r=np.empty(1),
                       KB_K=np.empty(1), KB_Phi=np.empty(1), KB_rp=np.empty(1),
                       KB_f_ice=np.empty(1), KB_rho_refr=np.empty(1),
                       KB_rho_ice=np.empty(1), KB_sublimation_model=np.empty(1),
                       KB_SD_amax=np.empty(1), KB_SD_abig=np.empty(1),
                       KB_SD_amed=np.empty(1), KB_SD_amin=np.empty(1),
                       KB_SD_qh=np.empty(1), KB_SD_qm=np.empty(1), KB_SD_ql=np.empty(1),
                       KB_sb=np.empty(1), KB_e=np.empty(1), KB_i=np.empty(1),
                       KB_As=np.empty(1), KB_bs=np.empty(1), KB_bg=np.empty(1), KB_t0_diss=np.empty(1),
                       KB_t1_diss=np.empty(1), a_planets=np.empty(1), m_planets=np.empty(1), f_accr=np.empty(1),
                       t0=np.empty(1), IT_s=np.empty(1), dip=np.empty(1), rtol=np.empty(1), atol=np.empty(1),
                       method_root=np.empty(1))
        import pandas as pd

        tab: pd.DataFrame = pd.DataFrame(columns=d)
        tab.to_csv(results + "/informations.csv")

# Mathematical constantes
Pi: np.float64 = np.double(np.pi)  #: Pi double precision
inf: np.float64 = np.double(np.inf)  #: infinite double precision
zero: np.float64 = np.double(0.)  #: 0. double precision

# Conversion factor from MKSA to code unit, the only unit that cannot be changed is the temperature in K

C_L: np.float64 = np.double(1 / 149597870700)  #: 1/6371009 # m -> au: length
C_M: np.float64 = np.double(1 / 5.9722e24)  #: 1/5.9722*1e24 # kg -> Mearth : mass
C_t: np.float64 = np.double(1 / (3600 * 24 * 365.25 * 1e6))  #: s-> 1 Myr : time

C_F: np.float64 = np.double(C_M * C_L / (C_t * C_t))  #: N -> Mearth au**2 / Myr : force
C_rho: np.float64 = np.double(C_M / (C_L * C_L * C_L))  #: kg/m3 -> Mearth / au**2: density
C_E: np.float64 = np.double(C_M * C_L * C_L / (C_t * C_t))  #: j ->  Mearth au**2 / Myr**2 : energy
C_Lum: np.float64 = np.double(C_E / C_t)  #: W = j / s ->  Mearth au**2 / Myr**3: Luminosity

yr: np.float64 = np.double(3600 * 24 * 365.25) * C_t  #: One year in code unit
kyr: np.float64 = np.double(3600 * 24 * 365.25 * 1.e3) * C_t  #: One thousand year in code unit
Myr: np.float64 = np.double(3600 * 24 * 365.25 * 1.e6) * C_t  #: One million year in code unit
Gyr: np.float64 = np.double(3600 * 24 * 365.25 * 1.e9) * C_t  #: One billion year in code unit

# Fundamentals constantes
G: np.float64 = np.double(6.67430e-11 * C_F * C_L * C_L / (C_M * C_M)
                          )  #: 6.67430e-11 N m*2 kg**-2 : Gravitational constant
Rgp: np.float64 = np.double(8.31446261815324 * C_E)  #: 8.31446261815324 J K−1 mol−1 : perfect gas constant
c: np.float64 = np.double(299792458) * C_L / C_t  #: 299792458 m/s Speed of light
sigma: np.float64 = np.double(5.670374419 * 1e-8 * C_E / (C_t * C_L * C_L)
                              )  #: 5.670374419 * 1e-8 W m-2 K-4 : Stefan-Boltzmann's constant
kb: np.float64 = np.double(1.380649e-23 * C_E)  #: 1.380649e-23 J/K Boltzmann constant
Na: np.float64 = np.double(Rgp / kb)  #: Avogadro constant
h: np.float64 = np.double(6.62607015e-34 * C_E * C_t)  #: 6.62607015e-34 J/s Planck constant

# Earth
Rearth: np.float64 = np.double(6371009 * C_L)  # 6371009 m Earth's radius
Mearth: np.float64 = np.double(5.9722 * 1e24 * C_M)  # 5.9722 * 1e24 kg Earth's mass
au: np.float64 = np.double(149597870700 * C_L)  # 149597870700 m : Astronomic unity
# Sun
Msun: np.float64 = np.double((1.9884 * 1e30) * C_M)  # 1.9891*10**30 kg Sun's mass
Rsun: np.float64 = np.double(1.392684e9 / 2. * C_L)  # Present-day sun's radius
Lsun: np.float64 = np.double(3.826e26 * C_Lum)  # 3.826 10^26 W Sun's luminosity (present day)

# Default central star properties
Ms: np.float64 = Msun
Ts: np.float64 = np.double(5500.)  # 5500 K Surface temperature of the central star
Rs: np.float64 = Rsun  # Radius
Ls: np.float64 = np.double(Lsun)  # Luminosity

# Disc properties

# Radial extension
a_in: np.float64 = .05 * au  #: Inner radius
# a_in = 5 * au  #: Inner radius
# a0 = 1. * au
a0: np.float64 = 2.65 * au
# a_out = 50 * au  #: outer radius
a_out: np.float64 = 500 * au  #: outer radius
# a_in = .5 * au
# a0 = 26.5 * au
# a0 = 50 * au
# a_out = 1500 * au

# Temperature
T0: np.float64 = (278 * ((Ls / Lsun) ** (1 / 4)) * ((a0 / au) ** (-1 / 2)))  #: Reference temperature at a0

pls_temp: np.float64 = -0.5  #: Temperature distribution power slope
# m_dot: np.float64 = np.double(1e-3 * Mearth / (C_t * 3600 * 24 * 365.25 * 1e6)
# m_dot: np.float64 = np.double(1e-3 * Mearth / (C_t * 3600 * 24 * 365.25 * 1e6)
#                               )  #: Gas generation rate (if constant generation rate)
m_dot: np.float64 = np.double(1e-1 * Mearth / (C_t * 3600 * 24 * 365.25 * 1e6)
                              )  #: Gas generation rate (if constant generation rate)

# Gas properties
# alpha: np.float64 = np.double(5e-1)  # viscous parameter
# alpha: np.float64 = np.double(1e-1)  # viscous parameter
alpha: np.float64 = np.double(1e-3)  #: viscous parameter
gamma: np.float64 = np.double(1.)
# gamma: np.float64 = np.double(7. / 5.)

# Belts properties

# Asteroid belt : Water ice

# AB_sublimation_model: str = "thermal_full"
AB_sublimation_model: str = "constant_rate"
# AB_sublimation_model: str = "none"
AB_a0: np.float64 = np.double(2.65) * au  #: Central radius
AB_delta_r: np.float64 = np.double(.8) * au  #: Radial extention above and below central redius (half of belt's size)
AB_Mbelt: np.float64 = np.double(0.12 * Mearth)  #: Total initial mass
AB_Abelt: np.float64 = np.double(0.06)  # Asteroids' albedo
AB_rho_refr: np.float64 = np.double(2.5e3 * C_M / (C_L * C_L * C_L))  # Refractories density 2.5 kg / m3
AB_rho_ice: np.float64 = np.double(1.e3 * C_M / (C_L * C_L * C_L))  # ice density = 1 kg / m3
AB_f_ice: np.float64 = np.double(0.2)  # Initial ice to solid mass ratio for Asteroids
AB_K: np.float64 = np.double(1e-5 * C_L * C_L / C_t)  # thermal diffusion coefficent
AB_Phi: np.float64 = np.double(0.6)  # Porosity
AB_rp: np.float64 = np.double(1e-6 * C_L)  # Pore's radius
# Size distribution of asteroid belt
AB_SD_amax: np.float64 = np.double(1.e6) * C_L  # Maximum size of asteroids
AB_SD_abig: np.float64 = np.double(120e3) * C_L  # Intermediate size
AB_SD_amed: np.float64 = np.double(20.e3) * C_L  # Medium size
AB_SD_amin: np.float64 = C_L  # Minimal size
AB_SD_qh: np.float64 = np.double(-4.5)  # Power slope of the distribution between abig and amax
AB_SD_qm: np.float64 = np.double(-1.2)  # Power slope of the distribution between amed and abig
AB_SD_ql: np.float64 = np.double(-3.6)  # Power slope of the distribution between amin and amed
# Parameter for the collision model
AB_sb: np.float64 = np.double(316) * C_L  # Intermediate size
AB_e: np.float64 = np.double(0.075)  # Eccentricity
AB_i: np.float64 = AB_e / np.double(2.)  # Inclination
AB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
AB_bs: np.float64 = np.double(-0.1)
AB_bg: np.float64 = np.double(0.5)
# Lifetime
AB_t0_diss: np.float64 = 2000 * Gyr  # Belt's lifetime before dissipation
AB_t1_diss: np.float64 = 0.1 * Myr  # Dissipation timescale
# Kuiper belt : CO ice
KB_sublimation_model: str = "none"
# KB_sublimation_model: str = "constant_rate"
KB_a0: np.float64 = np.double(44) * au  #: Central radius
KB_delta_r: np.float64 = np.double(8) * au  #: Radial extention above and below central redius (half of belt's size)
KB_Mbelt: np.float64 = np.double(0.01 * Mearth)  #: Total initial mass
KB_Abelt: np.float64 = np.double(0.06)  # KBO's albedo
KB_rho_refr: np.float64 = np.double(2.5e3 * C_M / (C_L * C_L * C_L))  # Solids density 2.5 kg / m3
KB_rho_ice: np.float64 = np.double(1.e3 * C_M / (C_L * C_L * C_L))  # ice density = 1 kg / m3
KB_f_ice: np.float64 = np.double(0.2)  # Initial ice to solid mass ratio for Asteroids
KB_K: np.float64 = np.double(1e-10 * C_L * C_L / C_t)  # thermal diffusion coefficent
KB_Phi: np.float64 = np.double(0.6)  # Porosity
KB_rp: np.float64 = np.double(1e-6 * C_L)  # Pore's radius
# Size distribution of asteroid belt
KB_SD_amax: np.float64 = np.double(1.e6) * C_L  # Maximum size of KBO
KB_SD_abig: np.float64 = np.double(120e3) * C_L  # Intermediate size
KB_SD_amed: np.float64 = np.double(20.e3) * C_L  # Medium size
KB_SD_amin: np.float64 = C_L  # Minimal size
KB_SD_qh: np.float64 = np.double(-4.5)  # Power slope of the distribution between abig and amax
KB_SD_qm: np.float64 = np.double(-1.2)  # Power slope of the distribution between amed and abig
KB_SD_ql: np.float64 = np.double(-3.6)  # Power slope of the distribution between amin and amed
# Parameter for the collision model
KB_sb: np.float64 = np.double(316) * C_L  # Intermediate size
KB_e: np.float64 = np.double(0.075)  # Eccentricity
KB_i: np.float64 = KB_e / np.double(2.)  # Inclination
KB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
KB_bs: np.float64 = np.double(-0.1)
KB_bg: np.float64 = np.double(0.5)
# Lifetime
KB_t0_diss: np.float64 = 10 * Gyr  # Belt's lifetime before dissipation
KB_t1_diss: np.float64 = 10 * Gyr  # Dissipation timescale

# Planets
# a_planets: np.ndarray = np.array([1.52371 * au, au, 0.723336 * au, 0.387098 * au])
# m_planets: np.ndarray = np.array([0.107 * Mearth, Mearth, 0.815 * Mearth, 0.055 * Mearth])
# a_planets: np.ndarray = np.array([1.53 * au, au, 0.723336 * au, 0.387098 * au])
# m_planets: np.ndarray = np.array([0.107 * Mearth, Mearth, 0.815 * Mearth, 0.055 * Mearth])
a_planets: np.ndarray = np.array([])
m_planets: np.ndarray = np.array([])
# a_planets: np.ndarray = np.array([au])  #: Semi majors axis
# a_planets: np.ndarray = np.array([0.6 * AB_a0])  #: Semi majors axis
# m_planets: np.ndarray = np.array([Mearth])  #: Masses
# a_planets: np.ndarray = np.array([30.069 * au, 19.189 * au, 9.537 * au])
# m_planets: np.ndarray = np.array([17.147 * Mearth, 14.536 * Mearth, 95.152 * Mearth])
f_accr: np.float64 = np.double(0.5)

# Photodissociation
t0: np.float64 = np.double(5.) * Myr  #: Time at which the disc dissipates
if os.path.exists(data_stars):  # Loading luminosity profiles as function of mass en time
    wfus = np.load(data_stars)
    interpLbol = CloughTocher2DInterpolator(np.array([wfus["Age"] * 3600 * 24 * 365.25 * 1.e6 * C_t,
                                                      wfus["M"] * Msun]).T, wfus["Lbol"] * Lsun, fill_value=Lsun)
    interpRs = CloughTocher2DInterpolator(np.array([wfus["Age"] * 3600 * 24 * 365.25 * 1.e6 * C_t,
                                                    wfus["M"] * Msun]).T, wfus["R"] * Rsun, fill_value=Rsun)
    interpReuv_3692 = CloughTocher2DInterpolator(np.array([wfus["Age"] * 3600 * 24 * 365.25 * 1.e6 * C_t,
                                                           wfus["M"] * Msun]).T, wfus["Reuv_3692"])
    interpReuv_lya = CloughTocher2DInterpolator(np.array([wfus["Age"] * 3600 * 24 * 365.25 * 1.e6 * C_t,
                                                          wfus["M"] * Msun]).T, wfus["Reuv_lya"])
else:
    print("Warning : no file for the central star's luminosity has been provided. The luminosity will be assumed "
          "to be a constant")
    interpLbol = None
    interpRs = None
    interpReuv_3692 = None
    interpReuv_lya = None

# Loading photodissociation data
photo_diss: dict = {}
tabH2O = np.loadtxt(data_leiden + "/H2O.txt")
photo_diss["lambdas_nm"] = tabH2O[:, 0]
photo_diss["cs"] = tabH2O[:, 1] * 1e-4 * C_L * C_L
tabI = np.loadtxt(data_leiden + "/ISRF.dat")
photo_diss["I"] = np.interp(photo_diss["lambdas_nm"], tabI[:, 0], tabI[:, 1]) / (1e-4 * C_L * C_L * C_t)
photo_diss["I"][photo_diss["lambdas_nm"] < tabI[0, 0]] = 0.

# Integration parameter
IT_s: int = 1000  #: Spatials steps
dip: int = 10  #: half of the number of sink cells for a given planets to model the accretion
method: str = 'RK45'  #: 'LSODA','BDF','DOP853','RK45' : Temporal integration method_root
rtol: np.float64 = np.double(1e-5)  #: Relative tolerance for temporal integration
atol: np.float64 = np.double(1e-250)  #: Absolute tolerance for temporal integration (by default too small hence to


# relative tolerance will always be the limiting factor)


def constantes() -> dict[str, np.double | str]:
    """
    All the constants

    Returns
    -------
    dict
        A dictionary with all the constants :
            - racine : str, The absolute path to the directory where all the results are saved
            - results : str, The relative path to racine of the directory (inside racine) where the simulations are saved
            - C_L : np.float64,  1/6371009  m -> au: length conversion factor
            - C_M : np.float64, 1/(5.9722*1e24) kg -> Mearth : mass conversion factor
            - C_t : np.float64, 1 / (3600 * 24 * 365.25 * 1e6) : s-> 1 Myr : time conversion factor
            - Ts : np.float64, Surface temperature of the central star
            - Ms : np.float64, Mass of the central star
            - Rs : np.float64, Radius of the central star
            - Ls : np.float64, Luminosity of the central star
            - a_in : np.float64, Inner radius of the integration domain
            - a0 : np.float64,
            - a_out : np.float64, Outer radius of the integration domain
            - T0 : np.float64, Temperature at a0
            - pls_temp : np.float64, the radial power slope of the temperature
            - mdot : np.float64, gas mass generation rate (if constant generation rate)
            - AB_Mbelt : np.float64, Asteroid belt's size
            - AB_Abelt : np.float64, Asteroids albedo
            - AB_a0 : np.float64, Asteroid belt's central radius
            - AB_delta_r : np.float64, Half of asteroid belt size
            - AB_K : np.float64, Thermal diffusion coefficient in asteroids
            - AB_Phi : np.float64, Porosity
            - AB_rp : np.float64, Asteroid's pore radius
            - AB_f_ice : np.float64, Ice to total mass ratio
            - AB_rho_refr : np.float64, Refractory's materials density in asteroids
            - AB_rho_ice : np.float64 Ice density
            - AB_sublimation_model : np.float64 Sublimation model in asteroid belt
            - AB_SD_amax : np.double, Maximum size of asteroids
            - AB_SD_abig : np.double, Intermediate size
            - AB_SD_amed : np.double, Medium size for size distribution
            - AB_SD_amin : np.double, Minimal size
            - AB_SD_qh : np.double, Power slope of the distribution between abig and amax
            - AB_SD_qm : np.double, Power slope of the distribution between amed and abig
            - AB_SD_ql : np.double, Power slope of the distribution between amin and amed
            - AB_s_min: np.float64 = np.double(316) * C_L  # Minimum size for collision model
            - AB_sb: np.float64 = np.double(316) * C_L  # Intermediate size for collision model
            - AB_s_max: np.float64 = np.double(316) * C_L  # Maximum size for collision model
            - AB_e: np.float64 = np.double(0.075)  # Eccentricity
            - AB_i: np.float64 = AB_e / np.double(2.)  # Inclination
            - AB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
            - AB_bs: np.float64 = np.double(-0.1)
            - AB_bg: np.float64 = np.double(0.5)
            - AB_t0_diss: np.float64 = 10 * Gyr  # Belt's lifetime before dissipation
            - AB_t1_diss: np.float64 = 10 * Gyr  # Dissipation timescale
            - KB_Mbelt : np.float64, Kuiper belt's mass
            - KB_Abelt : np.float64, KBO's albedo
            - KB_a0 : np.float64, Kuiper belt central radius
            - KB_delta_r : np.float64, Half of Kuiper belt size
            - KB_K : np.float64, Thermal diffusion coefficient in KBO
            - KB_Phi : np.float64, KBO's porosity
            - KB_rp : np.float64, KBO's pore's radius
            - KB_f_ice : np.float64, Initial ice to total mass ratio in KBO
            - KB_rho_refr : np.float64, Refractory's materials density in KBO
            - KB_rho_ice : np.float64, Ice density
            - KB_sublimation_model : np.float64, Sublimation model in Kuiper belt
            - KB_SD_amax : np.double, Maximum size of KBO
            - KB_SD_abig : np.double, Intermediate size
            - KB_SD_amed : np.double, Medium size
            - KB_SD_amin : np.double, Minimal size
            - KB_SD_qh : np.double, Power slope of the distribution between abig and amax
            - KB_SD_qm : np.double, Power slope of the distribution between amed and abig
            - KB_SD_ql : np.double, Power slope of the distribution between amin and amed
            - KB_s_min: np.float64 = np.double(316) * C_L  # Minimum size for collision model
            - KB_sb: np.float64 = np.double(316) * C_L  # Intermediate size for collision model
            - KB_s_max: np.float64 = np.double(316) * C_L  # Maximum size for collision model
            - KB_e: np.float64 = np.double(0.075)  # Eccentricity
            - KB_i: np.float64 = KB_e / np.double(2.)  # Inclination
            - KB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
            - KB_bs: np.float64 = np.double(-0.1)
            - KB_bg: np.float64 = np.double(0.5)
            - KB_t0_diss: np.float64 = 10 * Gyr  # Belt's lifetime before dissipation
            - KB_t1_diss: np.float64 = 10 * Gyr  # Dissipation timescale
            - a_planet : list[np.double], planet's semi-major axis
            - m_planet : list[np.double], planet's masses
            - f_accr : np.float64, Hydrodynamic accretion efficiency
            - t0 : The dissipation time of the protoplanetary disc
            - IT_s : int, number of spatial steps
            - dip : int, half of the number of sink cells for a given planets to model the accretion
            - rtol : np.float64, Relative tolerance for temporal integration
            - atol : np.float64, Absolute tolerance for temporal integration (by default too small hence to
                relative tolerance will always be the limiting factor)
            - method_root : str, 'LSODA','BDF','DOP853','RK45' : Temporal integration method_root

    """
    dic: dict = {"racine": racine, "results": results,
                 "C_L": C_L, "C_M": C_M, "C_t": C_t,
                 "Ms": Ms, "Ts": Ts, "Rs": Rs, "Ls": Ls,
                 "a_in": a_in, "a_out": a_out, "a0": a0,
                 "T0": T0, "pls_temp": pls_temp, "m_dot": m_dot,
                 "alpha": alpha, "gamma": gamma,
                 "AB_Mbelt": AB_Mbelt, "AB_Abelt": AB_Abelt,
                 "AB_a0": AB_a0, "AB_delta_r": AB_delta_r,
                 "AB_K": AB_K, "AB_Phi": AB_Phi, "AB_rp": AB_rp,
                 "AB_f_ice": AB_f_ice, "AB_rho_refr": AB_rho_refr,
                 "AB_rho_ice": AB_rho_ice, "AB_sublimation_model": AB_sublimation_model,
                 "AB_SD_amax": AB_SD_amax, "AB_SD_abig": AB_SD_abig,
                 "AB_SD_amed": AB_SD_amed, "AB_SD_amin": AB_SD_amin,
                 "AB_SD_qh": AB_SD_qh, "AB_SD_qm": AB_SD_qm, "AB_SD_ql": AB_SD_ql,
                 "AB_sb": AB_sb, "AB_e": AB_e, "AB_i": AB_i,
                 "AB_As": AB_As, "AB_bs": AB_bs, "AB_bg": AB_bg, "AB_t0_diss": AB_t0_diss, "AB_t1_diss": AB_t1_diss,
                 "KB_Mbelt": KB_Mbelt, "KB_Abelt": KB_Abelt,
                 "KB_a0": KB_a0, "KB_delta_r": KB_delta_r,
                 "KB_K": KB_K, "KB_Phi": KB_Phi, "KB_rp": KB_rp,
                 "KB_f_ice": KB_f_ice, "KB_rho_refr": KB_rho_refr,
                 "KB_rho_ice": KB_rho_ice, "KB_sublimation_model": KB_sublimation_model,
                 "KB_SD_amax": KB_SD_amax, "KB_SD_abig": KB_SD_abig,
                 "KB_SD_amed": KB_SD_amed, "KB_SD_amin": KB_SD_amin,
                 "KB_SD_qh": KB_SD_qh, "KB_SD_qm": KB_SD_qm, "KB_SD_ql": KB_SD_ql,
                 "KB_sb": KB_sb, "KB_e": KB_e, "KB_i": KB_i,
                 "KB_As": KB_As, "KB_bs": KB_bs, "KB_bg": KB_bg, "KB_t0_diss": KB_t0_diss, "KB_t1_diss": KB_t1_diss,
                 "a_planets": a_planets, "m_planets": m_planets, "f_accr": f_accr,
                 "t0": t0,
                 "IT_s": IT_s, "dip": dip, "rtol": rtol, "atol": atol, "method_root": method,
                 }
    return dic


def update_constantes(dic: dict) -> None:
    """
    Update all constates from a dictionary

    Parameters
    ----------

    dic : dict, optional, default=None
        A dictionary with all the constants :
            - racine : str, The absolute path to the directory where all the results are saved
            - results : str, The relative path to racine of the directory (inside racine) where the simulations are saved
            - C_L : np.float64,  1/6371009  m -> au: length conversion factor
            - C_M : np.float64, 1/(5.9722*1e24) kg -> Mearth : mass conversion factor
            - C_t : np.float64, 1 / (3600 * 24 * 365.25 * 1e6) : s-> 1 Myr : time conversion factor
            - Ts : np.float64, Surface temperature of the central star
            - Ms : np.float64, Mass of the central star
            - Rs : np.float64, Radius of the central star
            - Ls : np.float64, Luminosity of the central star
            - a_in : np.float64, Inner radius of the integration domain
            - a0 : np.float64,
            - a_out : np.float64, Outer radius of the integration domain
            - T0 : np.float64, Temperature at a0
            - mdot : np.float64, gas mass generation rate (if constant generation rate)
            - AB_Mbelt : np.float64, Asteroid belt's size
            - AB_Abelt : np.float64, Asteroids albedo
            - AB_a0 : np.float64, Asteroid belt's central radius
            - AB_delta_r : np.float64, Half of asteroid belt size
            - AB_K : np.float64, Thermal diffusion coefficient in asteroids
            - AB_Phi : np.float64, Porosity
            - AB_rp : np.float64, Asteroid's pore radius
            - AB_f_ice : np.float64, Ice to total mass ratio
            - AB_rho_refr : np.float64, Refractory's materials density in asteroids
            - AB_rho_ice : np.float64 Ice density
            - AB_sublimation_model : np.float64 Sublimation model in asteroid belt
            - AB_SD_amax : np.double, Maximum size of asteroids
            - AB_SD_abig : np.double, Intermediate size
            - AB_SD_amed : np.double, Medium size for size distribution
            - AB_SD_amin : np.double, Minimal size
            - AB_SD_qh : np.double, Power slope of the distribution between abig and amax
            - AB_SD_qm : np.double, Power slope of the distribution between amed and abig
            - AB_SD_ql : np.double, Power slope of the distribution between amin and amed
            - AB_s_min: np.float64 = np.double(316) * C_L  # Minimum size for collision model
            - AB_sb: np.float64 = np.double(316) * C_L  # Intermediate size for collision model
            - AB_s_max: np.float64 = np.double(316) * C_L  # Maximum size for collision model
            - AB_e: np.float64 = np.double(0.075)  # Eccentricity
            - AB_i: np.float64 = AB_e / np.double(2.)  # Inclination
            - AB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
            - AB_bs: np.float64 = np.double(-0.1)
            - AB_bg: np.float64 = np.double(0.5)
            - AB_t0_diss: np.float64 = 10 * Gyr  # Belt's lifetime before dissipation
            - AB_t1_diss: np.float64 = 10 * Gyr  # Dissipation timescale
            - KB_Mbelt : np.float64, Kuiper belt's mass
            - KB_Abelt : np.float64, KBO's albedo
            - KB_a0 : np.float64, Kuiper belt central radius
            - KB_delta_r : np.float64, Half of Kuiper belt size
            - KB_K : np.float64, Thermal diffusion coefficient in KBO
            - KB_Phi : np.float64, KBO's porosity
            - KB_rp : np.float64, KBO's pore's radius
            - KB_f_ice : np.float64, Initial ice to total mass ratio in KBO
            - KB_rho_refr : np.float64, Refractory's materials density in KBO
            - KB_rho_ice : np.float64, Ice density
            - KB_sublimation_model : np.float64, Sublimation model in Kuiper belt
            - KB_SD_amax : np.double, Maximum size of KBO
            - KB_SD_abig : np.double, Intermediate size
            - KB_SD_amed : np.double, Medium size
            - KB_SD_amin : np.double, Minimal size
            - KB_SD_qh : np.double, Power slope of the distribution between abig and amax
            - KB_SD_qm : np.double, Power slope of the distribution between amed and abig
            - KB_SD_ql : np.double, Power slope of the distribution between amin and amed
            - KB_s_min: np.float64 = np.double(316) * C_L  # Minimum size for collision model
            - KB_sb: np.float64 = np.double(316) * C_L  # Intermediate size for collision model
            - KB_s_max: np.float64 = np.double(316) * C_L  # Maximum size for collision model
            - KB_e: np.float64 = np.double(0.075)  # Eccentricity
            - KB_i: np.float64 = KB_e / np.double(2.)  # Inclination
            - KB_As: np.float64 = np.double(5.) * C_E / C_M  # Massique energy to disrupt the asteroids
            - KB_bs: np.float64 = np.double(-0.1)
            - KB_bg: np.float64 = np.double(0.5)
            - KB_t0_diss: np.float64 = 10 * Gyr  # Belt's lifetime before dissipation
            - KB_t1_diss: np.float64 = 10 * Gyr  # Dissipation timescale
            - a_planet : list[np.double], planet's semi-major axis
            - m_planet : list[np.double], planet's masses
            - f_accr : np.float64, Hydrodynamic accretion efficiency
            - t0 : The dissipation time of the protoplanetary disc
            - IT_s : int, number of spatial steps
            - dip : int, half of the number of sink cells for a given planets to model the accretion
            - rtol : np.float64, Relative tolerance for temporal integration
            - atol : np.float64, Absolute tolerance for temporal integration (by default too small hence to
                relative tolerance will always be the limiting factor)
            - method_root : str, 'LSODA','BDF','DOP853','RK45' : Temporal integration method_root


    Returns
    -------
    None

    """

    global racine, results, C_L, C_M, C_t, C_F, C_rho, C_E, C_Lum, yr, kyr, Myr, Gyr, G, Rgp, c, sigma, \
        kb, Na, h, Rearth, Mearth, au, Msun, Lsun, Ms, Ts, Rs, Ls, a_in, a_out, a0, T0, pls_temp, m_dot, alpha, gamma, \
        AB_Mbelt, AB_delta_r, AB_K, AB_Phi, AB_rp, AB_f_ice, AB_rho_refr, AB_rho_ice, \
        AB_sublimation_model, AB_SD_amax, AB_SD_abig, AB_SD_amed, AB_SD_amin, AB_SD_qh, AB_SD_qm, \
        AB_sb, AB_e, AB_i, AB_As, AB_bs, AB_bg, AB_t0_diss, AB_t1_diss, KB_Mbelt, KB_Abelt, \
        AB_SD_ql, KB_Mbelt, KB_Abelt, KB_a0, KB_delta_r, KB_K, KB_Phi, KB_rp, KB_f_ice, KB_rho_refr, \
        KB_rho_ice, KB_sublimation_model, KB_SD_amax, KB_SD_abig, KB_SD_amed, KB_SD_amin, KB_SD_qh, \
        KB_SD_qm, KB_SD_ql, KB_sb, KB_e, KB_i, KB_As, KB_bs, KB_bg, KB_t0_diss, KB_t1_diss, \
        a_planets, m_planets, f_accr, t0, IT_s, dip, rtol, atol, method

    if not os.path.exists(dic["racine"]):
        raise UserWarning("The directory racine : " + str(dic["racine"]) + "dosn't exist")
    racine = str(dic["racine"])
    results = str(dic["results"])
    C_L = np.double(dic["C_L"])
    C_M = np.double(dic["C_M"])
    C_t = np.double(dic["C_t"])
    Ms = np.double(dic["Ms"])
    Ts = np.double(dic["Ts"])
    Rs = np.double(dic["Rs"])
    Ls = np.double(dic["Ls"])
    a_in = np.double(dic["a_in"])
    a_out = np.double(dic["a_out"])
    a0 = np.double(dic["a0"])
    T0 = np.double(dic["T0"])
    pls_temp = np.double(dic["pls_temp"])
    m_dot = np.double(dic["m_dot"])
    alpha = np.double(dic["alpha"])
    gamma = np.double(dic["gamma"])
    AB_Mbelt = np.double(dic["AB_Mbelt"])
    AB_delta_r = np.double(dic["AB_delta_r"])
    AB_K = np.double(dic["AB_K"])
    AB_Phi = np.double(dic["AB_Phi"])
    AB_rp = np.double(dic["AB_rp"])
    AB_f_ice = np.double(dic["AB_f_ice"])
    AB_rho_refr = np.double(dic["AB_rho_refr"])
    AB_rho_ice = np.double(dic["AB_rho_ice"])
    AB_sublimation_model = np.double(dic["AB_sublimation_model"])
    AB_SD_abig = np.double(dic["AB_SD_abig"])
    AB_SD_amed = np.double(dic["AB_SD_amed"])
    AB_SD_amin = np.double(dic["AB_SD_amin"])
    AB_SD_qh = np.double(dic["AB_SD_qh"])
    AB_SD_qm = np.double(dic["AB_SD_qm"])
    AB_SD_ql = np.double(dic["AB_SD_ql"])
    AB_sb = np.double(dic["AB_sb"])
    AB_e = np.double(dic["AB_e"])
    AB_i = np.double(dic["AB_i"])
    AB_As = np.double(dic["AB_As"])
    AB_bs = np.double(dic["AB_bs"])
    AB_bg = np.double(dic["AB_bg"])
    AB_t0_diss = np.double(dic["AB_t0_diss"])
    AB_t1_diss = np.double(dic["AB_t1_diss"])
    KB_Mbelt = np.double(dic["KB_Mbelt"])
    KB_delta_r = np.double(dic["KB_delta_r"])
    KB_K = np.double(dic["KB_K"])
    KB_Phi = np.double(dic["KB_Phi"])
    KB_rp = np.double(dic["KB_rp"])
    KB_f_ice = np.double(dic["KB_f_ice"])
    KB_rho_refr = np.double(dic["KB_rho_refr"])
    KB_rho_ice = np.double(dic["KB_rho_ice"])
    KB_sublimation_model = np.double(dic["KB_sublimation_model"])
    KB_SD_abig = np.double(dic["KB_SD_abig"])
    KB_SD_amed = np.double(dic["KB_SD_amed"])
    KB_SD_amin = np.double(dic["KB_SD_amin"])
    KB_SD_qh = np.double(dic["KB_SD_qh"])
    KB_SD_qm = np.double(dic["KB_SD_qm"])
    KB_SD_ql = np.double(dic["KB_SD_ql"])
    KB_sb = np.double(dic["KB_sb"])
    KB_e = np.double(dic["KB_e"])
    KB_i = np.double(dic["KB_i"])
    KB_As = np.double(dic["KB_As"])
    KB_bs = np.double(dic["KB_bs"])
    KB_bg = np.double(dic["KB_bg"])
    KB_t0_diss = np.double(dic["KB_t0_diss"])
    KB_t1_diss = np.double(dic["KB_t1_diss"])
    t0 = np.double(dic["t0"])
    a_planets = list(np.array(dic["a_planets"], dtype="double"))
    m_planets = list(np.array(dic["m_planets"], dtype="double"))
    f_accr = np.double(dic["f_accr"])
    IT_s = np.double(dic["IT_s"])
    dip = np.double(dic["dip"])
    rtol = np.double(dic["rtol"])
    atol = np.double(dic["atol"])
    method = np.double(dic["method_root"])

    if "interrupteur.npz" not in os.listdir(results):
        np.savez_compressed(results + "/interrupteur.npz")
    if "informations.csv" not in os.listdir(results):
        d: dict = dict(simulation=np.empty(1), C_L=np.empty(1), C_M=np.empty(1), C_t=np.empty(1),
                       Me=np.empty(1), Mc=np.empty(1), Rc=np.empty(1), a=np.empty(1),
                       gamma=np.empty(1), mu=np.empty(1), Pd=np.empty(1), Td=np.empty(1),
                       Mp=np.empty(1), IT_s=np.empty(1), méthode=np.empty(1), Rad=np.empty(1))
        tab: pd.DataFrame = pd.DataFrame(columns=d)
        tab.to_csv(results + "/informations.csv")
        print("test csv : ", os.listdir(results))

    # Secondary calculs
    C_F = np.double(C_M * C_L / (C_t * C_t))  #: N -> Mearth au**2 / Myr : force
    C_rho = np.double(C_M / (C_L * C_L * C_L))  #: kg/m3 -> Mearth / au**2: density
    C_E = np.double(C_M * C_L * C_L / (C_t * C_t))  #: j ->  Mearth au**2 / Myr**2 : energy
    C_Lum = np.double(C_E / C_t)  #: W = j / s ->  Mearth au**2 / Myr**3: Luminosity

    yr = np.double(3600 * 24 * 365.25) * C_t  #: One year in code unit
    kyr = np.double(3600 * 24 * 365.25 * 1.e3) * C_t  #: One million year in code unit
    Myr = np.double(3600 * 24 * 365.25 * 1.e6) * C_t  #: One million year in code unit
    Gyr = np.double(3600 * 24 * 365.25 * 1.e9) * C_t  #: One million year in code unit

    # Fundamentals constantes
    G = np.double(6.67430e-11 * C_F * C_L * C_L / (C_M * C_M)
                  )  #: 6.67430e-11 N m*2 kg**-2 : Gravitational constant
    Rgp = np.double(8.31446261815324 * C_E)  #: 8.31446261815324 J K−1 mol−1 : perfect gas constant
    c = np.double(299792458) * C_L / C_t  #: 299792458 m/s Speed of light
    sigma = np.double(5.670374419 * 1e-8 * C_E / (C_t * C_L * C_L)
                      )  #: 5.670374419 * 1e-8 W m-2 K-4 : Stefan-Boltzmann's constant
    kb = np.double(1.380649e-23 * C_E)  #: 1.380649e-23 J/K Boltzmann constant
    Na = np.double(Rgp / kb)  #: Avogadro constant
    h = np.double(6.62607015e-34 * C_E * C_t)  #: 6.62607015e-34 J/s Planck constant

    # Earth
    Rearth = np.double(6371009 * C_L)  # 6371009 m Earth's radius
    Mearth = np.double(5.9722 * 1e24 * C_M)  # 5.9722 * 1e24 kg Earth's mass
    au = np.double(149597870700 * C_L)  # 149597870700 m : Astronomic unity
    # Sun
    Msun = np.double((1.9884 * 1e30) * C_M)  # 1.9891*10**30 kg Sun's mass
    Lsun = np.double(3.826e26 * C_Lum)  # 3.826 10^26 W Sun's luminosity (presant day)


def chargement_constantes(consts_filename: str) -> None:
    """
    Load all the constantes in const_filename

    Parameters
    ----------
    consts_filename : str
        path to the .npz file containing all the constantes 

    Returns
    -------

    """
    print("Loading the new constants : ", consts_filename)
    dic = dict(np.load(consts_filename))
    update_constantes(dic)


if "constantes.npz" in os.listdir():
    chargement_constantes("constantes.npz")

elif "constantes.npz" in os.listdir(racine):
    chargement_constantes(racine + "/constantes.npz")

# Typical configuration sets

# Solar system like, with only an asteroid belt and an earth-like planet only

sys_sol_ast: dict = {"racine": racine, "results": results,
                     "C_L": C_L, "C_M": C_M, "C_t": C_t,
                     "Ms": Msun, "Ts": np.double(5500.), "Rs": Rsun, "Ls": Lsun,
                     "a_in": .05 * au, "a_out": 100 * au, "a0": 2.65 * au,
                     "T0": T0, "pls_temp": np.double(-1 / 2), "mH_init": zero,
                     "alpha": np.double(1e-3), "gamma": np.double(1.),
                     "AB_mdot_st_init": np.double(1e-10) * Mearth / Myr,
                     "AB_Mbelt": np.double(0.12) * Mearth,
                     "AB_Abelt": np.double(0.06),
                     "AB_a0": np.double(2.65) * au, "AB_delta_r": np.double(2.65 / 2.) * au,
                     "AB_K": np.double(1e-5 * C_L * C_L / C_t), "AB_Phi": np.double(0.6),
                     "AB_rp": np.double(1e-6 * C_L),
                     "AB_f_ice": np.double(0.2), "AB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                     "AB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "AB_sublimation_model": "thermal_full",
                     "AB_const_mdot": np.double(1e-1) * Mearth / Myr,
                     "AB_SD_amax": np.double(1.e6) * C_L, "AB_SD_abig": np.double(120e3) * C_L,
                     "AB_SD_amed": np.double(20.e3) * C_L, "AB_SD_amin": C_L,
                     "AB_SD_qh": np.double(-4.5), "AB_SD_qm": np.double(-1.2), "AB_SD_ql": np.double(-3.6),
                     "AB_sb": np.double(316) * C_L, "AB_e": np.double(0.075), "AB_i": np.double(0.075 / 2.),
                     "AB_As": np.double(5.) * C_E / C_M, "AB_bs": np.double(-0.1), "AB_bg": np.double(0.5),
                     "AB_t0_diss": 150 * Myr, "AB_t1_diss": Myr,
                     "KB_mdot_st_init": zero,
                     "KB_Mbelt": np.double(0.01 * Mearth), "KB_Abelt": np.double(0.06),
                     "KB_a0": np.double(44) * au, "KB_delta_r": np.double(8) * au,
                     "KB_K": np.double(1e-10 * C_L * C_L / C_t), "KB_Phi": np.double(0.6),
                     "KB_rp": np.double(1e-6 * C_L),
                     "KB_f_ice": np.double(0.2), "KB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                     "KB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "KB_sublimation_model": "none",
                     "KB_const_mdot": zero,
                     "KB_SD_amax": np.double(1.e6) * C_L, "KB_SD_abig": np.double(120e3) * C_L,
                     "KB_SD_amed": np.double(20.e3) * C_L, "KB_SD_amin": C_L,
                     "KB_SD_qh": np.double(-4.5), "KB_SD_qm": np.double(-1.2), "KB_SD_ql": np.double(-3.6),
                     "KB_sb": np.double(316) * C_L, "KB_e": np.double(0.075), "KB_i": np.double(0.075 / 2.),
                     "KB_As": np.double(5.) * C_E / C_M, "KB_bs": np.double(-0.1), "KB_bg": np.double(0.5),
                     "KB_t0_diss": np.double(10.) * Gyr, "KB_t1_diss": np.double(10.) * Gyr,
                     "a_planets": np.array([au]), "m_planets": np.array([Mearth]), "f_accr": np.double(0.5),
                     "t0": np.double(5. * Myr),
                     "IT_s": 3000, "dip": 10, "rtol": np.double(1e-6), "atol": np.double(1e-250), "method_root": 'RK45',
                     }

# Only a kuiper belt. By changing the sublimation model to constant rate, the masse generation rate is 0.1 Mearth/Myr.
# This mass generation rate can also be modified if needed in that case

sys_sol_kuip: dict = {"racine": racine, "results": results,
                      "C_L": C_L, "C_M": C_M, "C_t": C_t,
                      "Ms": Msun, "Ts": np.double(5500.), "Rs": Rsun, "Ls": Lsun,
                      "a_in": .05 * au, "a_out": 500 * au, "a0": 44. * au,
                      "T0": T0, "pls_temp": np.double(-1 / 2), "mH_init": zero,
                      "alpha": np.double(1e-3), "gamma": np.double(1.),
                      "AB_mdot_st_init": zero,
                      "AB_Mbelt": np.double(0.12) * Mearth,
                      "AB_Abelt": np.double(0.06),
                      "AB_a0": np.double(2.65) * au, "AB_delta_r": np.double(2.65 / 2.) * au,
                      "AB_K": np.double(1e-5 * C_L * C_L / C_t), "AB_Phi": np.double(0.6),
                      "AB_rp": np.double(1e-6 * C_L),
                      "AB_f_ice": np.double(0.2), "AB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                      "AB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "AB_sublimation_model": "none",
                      "AB_const_mdot": zero,
                      "AB_SD_amax": np.double(1.e6) * C_L, "AB_SD_abig": np.double(120e3) * C_L,
                      "AB_SD_amed": np.double(20.e3) * C_L, "AB_SD_amin": C_L,
                      "AB_SD_qh": np.double(-4.5), "AB_SD_qm": np.double(-1.2), "AB_SD_ql": np.double(-3.6),
                      "AB_sb": np.double(316) * C_L, "AB_e": np.double(0.075), "AB_i": np.double(0.075 / 2.),
                      "AB_As": np.double(5.) * C_E / C_M, "AB_bs": np.double(-0.1), "AB_bg": np.double(0.5),
                      "AB_t0_diss": 150 * Myr, "AB_t1_diss": Myr,
                      "KB_mdot_st_init": np.double(1e-10) * Mearth / Myr,
                      "KB_Mbelt": np.double(0.01 * Mearth), "KB_Abelt": np.double(0.06),
                      "KB_a0": np.double(44) * au, "KB_delta_r": np.double(8) * au,
                      "KB_K": np.double(1e-10 * C_L * C_L / C_t), "KB_Phi": np.double(0.6),
                      "KB_rp": np.double(1e-6 * C_L),
                      "KB_f_ice": np.double(0.2), "KB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                      "KB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "KB_sublimation_model": "thermal_full",
                      "KB_const_mdot": np.double(1e-1) * Mearth / Myr,
                      "KB_SD_amax": np.double(1.e6) * C_L, "KB_SD_abig": np.double(120e3) * C_L,
                      "KB_SD_amed": np.double(20.e3) * C_L, "KB_SD_amin": C_L,
                      "KB_SD_qh": np.double(-4.5), "KB_SD_qm": np.double(-1.2), "KB_SD_ql": np.double(-3.6),
                      "KB_sb": np.double(316) * C_L, "KB_e": np.double(0.075), "KB_i": np.double(0.075 / 2.),
                      "KB_As": np.double(5.) * C_E / C_M, "KB_bs": np.double(-0.1), "KB_bg": np.double(0.5),
                      "KB_t0_diss": np.double(10.) * Gyr, "KB_t1_diss": np.double(10.) * Gyr,
                      "a_planets": np.array([30.069 * au, 19.189 * au, 9.537 * au, 5.202 * au,
                                             1.53 * au, au, 0.723336 * au, 0.387098 * au]),
                      "m_planets": np.array([17.147 * Mearth, 14.536 * Mearth, 95.152 * Mearth, 317.8 * Mearth,
                                             0.107 * Mearth, Mearth, 0.815 * Mearth, 0.055 * Mearth]),
                      "f_accr": np.double(0.5),
                      "t0": np.double(5. * Myr),
                      "IT_s": 2000, "dip": 10, "rtol": np.double(1e-6), "atol": np.double(1e-250),
                      "method_root": 'RK45',
                      }

# Solar system like, primordial belt

sys_sol_ppd: dict = {"racine": racine, "results": results,
                     "C_L": C_L, "C_M": C_M, "C_t": C_t,
                     "Ms": Msun, "Ts": np.double(5500.), "Rs": Rsun, "Ls": Lsun,
                     "a_in": .05 * au, "a_out": 100 * au, "a0": 2.65 * au,
                     "T0": T0, "pls_temp": np.double(-1 / 2), "mH_init": Mearth,
                     "alpha": np.double(1e-3), "gamma": np.double(1.),
                     "AB_mdot_st_init": np.double(1e-10) * Mearth / Myr,
                     "AB_Mbelt": np.double(0.12) * Mearth,
                     "AB_Abelt": np.double(0.06),
                     "AB_a0": np.double(2.65) * au, "AB_delta_r": np.double(2.65 / 2.) * au,
                     "AB_K": np.double(1e-5 * C_L * C_L / C_t), "AB_Phi": np.double(0.6),
                     "AB_rp": np.double(1e-6 * C_L),
                     "AB_f_ice": np.double(0.2), "AB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                     "AB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "AB_sublimation_model": "none",
                     "AB_const_mdot": np.double(1e-1) * Mearth / Myr,
                     "AB_SD_amax": np.double(1.e6) * C_L, "AB_SD_abig": np.double(120e3) * C_L,
                     "AB_SD_amed": np.double(20.e3) * C_L, "AB_SD_amin": C_L,
                     "AB_SD_qh": np.double(-4.5), "AB_SD_qm": np.double(-1.2), "AB_SD_ql": np.double(-3.6),
                     "AB_sb": np.double(316) * C_L, "AB_e": np.double(0.075), "AB_i": np.double(0.075 / 2.),
                     "AB_As": np.double(5.) * C_E / C_M, "AB_bs": np.double(-0.1), "AB_bg": np.double(0.5),
                     "AB_t0_diss": 150 * Myr, "AB_t1_diss": Myr,
                     "KB_mdot_st_init": zero,
                     "KB_Mbelt": np.double(0.01 * Mearth), "KB_Abelt": np.double(0.06),
                     "KB_a0": np.double(44) * au, "KB_delta_r": np.double(8) * au,
                     "KB_K": np.double(1e-10 * C_L * C_L / C_t), "KB_Phi": np.double(0.6),
                     "KB_rp": np.double(1e-6 * C_L),
                     "KB_f_ice": np.double(0.2), "KB_rho_refr": np.double(2.5e3 * C_M / (C_L * C_L * C_L)),
                     "KB_rho_ice": np.double(1.e3 * C_M / (C_L * C_L * C_L)), "KB_sublimation_model": "none",
                     "KB_const_mdot": zero,
                     "KB_SD_amax": np.double(1.e6) * C_L, "KB_SD_abig": np.double(120e3) * C_L,
                     "KB_SD_amed": np.double(20.e3) * C_L, "KB_SD_amin": C_L,
                     "KB_SD_qh": np.double(-4.5), "KB_SD_qm": np.double(-1.2), "KB_SD_ql": np.double(-3.6),
                     "KB_sb": np.double(316) * C_L, "KB_e": np.double(0.075), "KB_i": np.double(0.075 / 2.),
                     "KB_As": np.double(5.) * C_E / C_M, "KB_bs": np.double(-0.1), "KB_bg": np.double(0.5),
                     "KB_t0_diss": np.double(10.) * Gyr, "KB_t1_diss": np.double(10.) * Gyr,
                     "a_planets": np.array([au]), "m_planets": np.array([Mearth]), "f_accr": np.double(0.5),
                     "t0": np.double(5. * Myr),
                     "IT_s": 3000, "dip": 10, "rtol": np.double(1e-6), "atol": np.double(1e-250), "method_root": 'RK45',
                     }
