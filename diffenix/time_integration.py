from __future__ import annotations
import gc
from typing import Any

import numpy as np

# from generation_configs import directory
from diffenix.systems import *
import pandas as pd


def new_simulation(consts_params: dict = None, directory_params: str = None) -> tuple[str, dict[str, Any]]:
    """
    Add directories/files needed to save the new simulation

    Parameters
    ----------

    consts_params : dict, optional
        The dictionary with all constantes and parameter for the simulation, default use the globals values defined in
         Constantes.py
    directory_params : str, optional
        The path to a directory containing several .npz files for a given set of constants. The first non integrated
        set of constant will be used for this new simulation

    Returns
    -------
    tuple[str, dict[str, Any]]
        the name of the new simulation's directory
        the dictionary with all the constants

    """
    data: pd.DataFrame = pd.read_csv(results + "/informations.csv")
    # All the keys informations about the simulations are stored into this csv file
    print("data.Simulation : ", data.Simulation)
    if len(data.Simulation) > 0:
        n: int = int(data.Simulation.values[-1][11:])
    else:
        n: int = 0

    if directory_params is not None:
        if not os.path.exists(directory_params):
            raise UserWarning("New simulation :  the path to the directory containing the sets of constantes :",
                              directory_params, " doesn't exist")
        l = os.listdir(directory_params)
        l.sort()
        name_consts = l.pop()
        consts_params = dict(np.load(directory_params + "/" + name_consts))
        while len(l) > 0 and str(consts_params["Simulation"]) in data["Simulation"]:
            name_consts = l.pop()
            consts_params = dict(np.load(directory_params + "/" + name_consts))
        if len(l) == 0:
            raise UserWarning("new_simulation : All the sets of parameters as already been integrated")
    elif consts_params is None:
        consts_params: dict[str, Any] = constantes()  # The dictionary with all the necessary constantes

    if "Simulation" in consts_params.keys():
        if "Simulation_" not in str(consts_params["Simulation"]):
            name: str = "Simulation_" + str(consts_params["Simulation"])
        else:
            name: str = str(consts_params["Simulation"])
    else:
        name: str = "Simulation_" + str(n + 1)
    simulation: dict[str, Any] = {"Simulation": np.array(name)}

    os.mkdir(results + "/" + name)  # Creating the new directory
    np.savez_compressed(results + "/" + name + "/interrupteur", {})
    consts_params["Simulation"] = name
    consts_params["directory"] = results + "/" + name
    consts_params["n_simu"] = n + 1
    d_enrg: dict[str, np.ndarray] = dict()
    for k in consts_params.keys():
        d_enrg[k] = np.array(consts_params[k])
    np.savez_compressed(results + "/" + name + "/Constantes", d_enrg)
    simulation.update(d_enrg)
    del simulation["a_planets"]
    del simulation["m_planets"]
    simulation["nb_planets"] = len(d_enrg["a_planets"])
    data: pd.DataFrame = pd.concat([data, pd.DataFrame(simulation, index=[0])], ignore_index=True)

    # The table to be saved as the new information csv file
    data.to_csv(results + "/informations.csv", index=False)  # Saving the information file

    return results + "/" + name, consts_params


def time_integration(tf: np.float64 = 100 * Myr, save_time_step: np.float64 = 0.1 * Myr,
                     consts_params: dict = None, directory_params: str = None,
                     integrate_only_surface_density: bool = False
                     ) -> Sol:
    """
    Build a new simulation and integrate it until tf

    Parameters
    ----------
    tf : np.float64, optional, default = 10 Myr

    save_time_step : np.float64, optional, default = 0.1 Myr
        The time-steps to save the simulation

    consts_params : dict, optional
        The dictionary with all constantes and parameter for the simulation, default use the globals values defined in
        Constantes.py

    directory_params : str, optional
        The path to a directory containing several .npz files for a given set of constants. The first non integrated
        set of constant will be used for this new simulation

    integrate_only_surface_density : bool, optional, default=False
        To integrate only the evolution of surfaces densitys

    Returns
    -------
    Sol
        The new simulation

    """
    # if consts_params is None and directory_params is None and os.path.exists("./configs"):
    #     try:
    #         simulation, d = new_simulation(directory_params="./configs")
    #     except UserWarning:
    #         # All the set of constantes has already been integrated
    #         simulation, d = new_simulation()
    # else:
    #     simulation, d = new_simulation(consts_params=consts_params, directory_params=directory_params)
    if consts_params is None or "Simulation" not in consts_params.keys():
        simulation, d = new_simulation(consts_params=consts_params, directory_params=directory_params)
    else:
        os.mkdir(results + "/" + consts_params["Simulation"])  # Creating the new directory
        np.savez_compressed(results + "/" + consts_params["Simulation"] + "/interrupteur", {})
        consts_params["directory"] = results + "/" + consts_params["Simulation"]
        consts_params["n_simu"] = int(str(consts_params["Simulation"])[11:])
        d = consts_params.copy()
    print("Initialization of ", d["Simulation"])
    sol = Sol(consts=d)
    nu0: np.float64 = (Rgp * T0(sol.t0, sol.a0, sol.Ms) * sol.alpha * (sol.a0 ** (3 / 2))
                       / (sol.mu.mean() * np.sqrt(G * sol.Ms)))
    dtmax: np.float64 = min(np.min((sol.r[1:] - sol.r[:-1]) * (sol.r[1:] - sol.r[:-1])
                            / (nu0 * ((((sol.r[1:] + sol.r[:-1]) / 2) / sol.a0) ** (3 / 2 + sol.pls_temp)))),
                            60. * yr)
    # dtmax: np.float64 = min((sol.r[1:] - sol.r[:-1]) * (sol.r[1:] - sol.r[:-1]) / nu(sol)[1:])
    # The maximum timestep to avoid instability in numerical integration

    print("dtmax=", dtmax / (C_t * 3600 * 34 * 365.25))
    print("Minimum number of temporal steps:", int(tf / dtmax), "dtmax=", dtmax / (C_t * 3600 * 24 * 365.25), "y")

    if tf < yr:
        sol.time_integration(tf, dtmax=dtmax, integrate_only_surface_density=integrate_only_surface_density)
    elif tf < 0.5 * Myr:
        sol.time_integration(tf, dtmax=dtmax, integrate_only_surface_density=integrate_only_surface_density)
    else:
        for i in range(1, int(tf / save_time_step) + 1):
            tfi = min(tf, i * save_time_step)
            print("tfi ", tfi / Myr, " Myr")
            sol.time_integration(tfi, dtmax=dtmax, integrate_only_surface_density=integrate_only_surface_density)
    return sol


def restart(number: int | str, tf: np.float64 = 0.1 * Gyr, save_time_step : np.float64 = 0.1 * Myr,
            rtol: np.float64 = None, method: str = None,
            integrate_only_surface_density : bool = False) -> Sol:
    """
    Restart the simulation number

    Parameters
    ----------
    number : int
        The number of the simulation to be restarted

    tf : np.float64, optional, default = 10 Myr

    save_time_step : np.float64, optional, default = 0.1 Myr
        The time-steps to save the simulation

    integrate_only_surface_density : bool, optional, default=False
        To integrate only the evolution of surfaces densitys

    Returns
    -------
    Sol
        The integrated solution

    """
    if isinstance(number, int):
        name: str = "Simulation_" + str(number)
        print("Restarting simulation : ", number)
    else:
        print("Restarting ", number)
        name = number


    if os.path.exists(results + "/" + name + "/infos.npz"):
        infos: dict = dict(np.load(results + "/" + name + "/infos.npz"))
        if infos["status"] == "integrating":
            raise UserWarning("The simulation is already being integrated by another process")
        elif infos["status"] == "integreted":
            raise UserWarning("The simulation has already been integrated until the end")
        infos["status"] = "integrating"
        np.savez_compressed(results + "/" + name + "/infos.npz", **infos)
    else:
        infos: dict = {"status": "integrating"}
        np.savez_compressed(results + "/" + name + "/infos.npz", **infos)

    sol = Sol(directory=results + '/' + name, filename="Simulation_diffusion")
    if rtol is not None:
        sol.rtol = rtol
    if method is not None:
        sol.method = method
    nu0: np.float64 = (Rgp * T0(sol.t0, sol.a0, sol.Ms) * sol.alpha * (sol.a0 ** (3 / 2))
                       / (sol.mu.mean() * np.sqrt(G * sol.Ms)))
    dtmax: np.float64 = min(np.min((sol.r[1:] - sol.r[:-1]) * (sol.r[1:] - sol.r[:-1])
                            / (nu0 * ((((sol.r[1:] + sol.r[:-1]) / 2) / sol.a0) ** (3 / 2 + sol.pls_temp)))),
                            1 * yr)
    # The maximum timestep to avoid instability in numerical integration

    print("Minimum number of temporal steps:", int(tf / dtmax), "dtmax=", dtmax / (C_t * 3600 * 24 * 365.25), "y")
    if not integrate_only_surface_density:
        t0: np.float64 = sol.tps_mf[-1]
    else:
        t0: np.float64 = sol.tps[-1]

    gc.enable()
    if tf < 0 and t0 < 1e3 * Myr:
        tfi = t0 + save_time_step
        print("tfi ", tfi / Myr, " Myr")
        gc.collect()
        sol.time_integration(tfi, dtmax=dtmax, integrate_only_surface_density=integrate_only_surface_density)
        gc.collect()
    else:
        for i in range(int(t0 / save_time_step) + 1,
                       int(tf / save_time_step) + 1):
            tfi = min(tf, i * save_time_step)
            print("tfi ", tfi / Myr, " Myr")
            gc.collect()
            sol.time_integration(tfi, dtmax=dtmax, integrate_only_surface_density=integrate_only_surface_density)
            gc.collect()
    if sol.tps_mf[-1] < 100 * Myr:
        infos["status"] = "done"
    else:
        infos["status"] = "integreted"
    np.savez_compressed(results + "/" + name + "/infos.npz", **infos)

    return sol


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        import os
        if sys.argv[1] == "configs":
            l = os.listdir("../configs")
            l.sort()
            if int(sys.argv[2]) < len(l):
                params = dict(np.load("./configs/" + l[int(sys.argv[2])]))
                print(int(sys.argv[2]), l[int(sys.argv[2])], params["Simulation"])
                if not os.path.exists(results + "/" + params["Simulation"]):
                    time_integration(consts_params=params)
                # else:
                #     restart(params["Simulation"], rtol=np.double(1e-4))
        else:
            print("argv :", sys.argv[1])
            restart(int(sys.argv[1]), tf=100 * Myr, save_time_step=0.1 * Myr)
    else:
        time_integration()

