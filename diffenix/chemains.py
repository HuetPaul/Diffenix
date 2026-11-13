import os
import numpy as np

# Defaults values

racine: str = "."
results: str = racine + "/Results"
data_leiden: str = racine + "/Leiden_database/"
data_stars: str = racine + "/data_stars/data_fus.npz"
data_radmc: str = racine + "/data_radmc"


def load_from_files(path_to_file: str = "paths.txt") -> None:
    """

    Load the path from a txt file

    Parameters
    ----------
    path_to_file : str
        The path to the txt file
        One path per line, the delimiter between the name of path and the path is `,`
        The names are racine, results, data_leiden and data_stars

    Returns
    -------
    None

    """
    if not os.path.exists(path_to_file):
        raise UserWarning("Constante.load_from_files : ", path_to_file, "does not exist")
    global racine, results, data_leiden, data_stars, data_radmc

    paths = dict(np.loadtxt("paths.txt", dtype=str, delimiter=","))
    print("Paths : ", paths)
    if "racine" in paths.keys():
        if not os.path.exists(paths["racine"]):
            raise UserWarning("Constante.load_from_files : ", paths["racine"], "does not exist")
        if not os.path.isdir(paths["racine"]):
            raise UserWarning("Constante.load_from_files : ", paths["racine"], "is not a directory")
        racine = paths["racine"]

    if "results" in paths.keys():
        if not os.path.exists(paths["results"]) and not os.path.exists(racine + paths["results"]):
            raise UserWarning("Constante.load_from_files : ", paths["results"], "does not exist")
        if not os.path.isdir(paths["results"]) and not os.path.isdir(racine + paths["results"]):
            raise UserWarning("Constante.load_from_files : ", paths["results"], "is not a directory")

        if os.path.exists(paths["results"]):
            results = paths["results"]
        else:
            results = racine + paths["results"]

    if "data_leiden" in paths.keys():
        if not os.path.exists(paths["data_leiden"]) and not os.path.exists(racine + paths["data_leiden"]):
            raise UserWarning("Constante.load_from_files : ", paths["data_leiden"], "does not exist")
        if not os.path.isdir(paths["data_leiden"]) and not os.path.isdir(racine + paths["data_leiden"]):
            raise UserWarning("Constante.load_from_files : ", paths["data_leiden"], "is not a directory")

        if os.path.exists(paths["data_leiden"]):
            data_leiden = paths["data_leiden"]
        else:
            data_leiden = racine + paths["data_leiden"]

    if "data_stars" in paths.keys():
        if not os.path.exists(paths["data_stars"]) and not os.path.exists(racine + paths["data_stars"]):
            raise UserWarning("Constante.load_from_files : ", paths["data_stars"], "does not exist")

        if os.path.exists(paths["data_stars"]):
            data_stars = paths["data_stars"]
        else:
            data_stars = racine + paths["data_stars"]

    if "data_radmc" in paths.keys():
        # if not os.path.exists(paths["data_radmc"]) and not os.path.exists(racine + paths["data_radmc"]):
        #     raise UserWarning("Constante.load_from_files : ", paths["data_radmc"], "does not exist")

        if os.path.exists(paths["data_radmc"]):
            data_radmc = paths["data_radmc"]
        else:
            data_radmc = racine + paths["data_radmc"]


if os.path.exists("./paths.txt"):
    load_from_files("./paths.txt")
    if not os.path.exists(racine):
        raise UserWarning("Constante.load_from_files : ", racine, "does not exist")
    if not os.path.isdir(racine):
        raise UserWarning("Constante.load_from_files : ", racine, "is not a directory")

    if not os.path.exists(results):
        raise UserWarning("Constante.load_from_files : ", results, "does not exist")
    if not os.path.isdir(results):
        raise UserWarning("Constante.load_from_files : ", results, "is not a directory")

    if not os.path.exists(data_stars):
        print("User warning : no data to model the central star as been found. A constant luminosity will be assumed.")
        # raise UserWarning("Constante.load_from_files : ", data_stars, "does not exist")

    if not os.path.exists(data_leiden):
        raise UserWarning("Constante.load_from_files : ", data_leiden, "does not exist")
    if not os.path.isdir(data_leiden):
        raise UserWarning("Constante.load_from_files : ", data_leiden, "is not a directory")

    # if not os.path.exists(data_radmc):
    #     raise UserWarning("Constante.load_from_files : ", data_radmc, "does not exist")
    # if not os.path.isdir(data_radmc):
    #     raise UserWarning("Constante.load_from_files : ", data_radmc, "is not a directory")
else:
    print("No path.txt file has been found. Working directory : ", os.getcwd())
