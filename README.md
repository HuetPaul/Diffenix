# Diffenix

![logo](logo_diffenix.png)

## Description

This model aims to folow the gas evolution into debris discs including its accretion onto planets.
The gas can be generated at the warm (exo)asteroid belt (water gas will be produced) and/or at the cold (exo)Kuiper belt (CO gas will be produced)


## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.


You can install this library directly with pip:
`pip install phenigraph`

Alternatively, you can download it from GitLab, after which you will need to compile it.
In the main folder, run the following command :

`python setup.py bdist_wheel`

The installation wheel file will be in the newly created ‘dist’ folder.
To install the library in the local environment, use the following command:

`pip install /path/to/wheelfile.whl`

### Directory and data needed before starting

The library assumes that the working directory has a specific structure. You can change the reference directory and the location of necessary data files by creating a `paths.txt` file in the working directory.

The necessary data are:
- The water cross-section which can be found on this [page](https://home.strw.leidenuniv.nl/~ewine/photo/display_h2o_b89e91ccf14bfd7f485dd7be7d789b0a.html):
        The file should be saved as `H2O.txt` in a directory called `Leiden_database`. 
        The name and the location of this directory can be changed in the `paths.txt` file.

- The interstellar spectrum which can be found at this [page](https://home.strw.leidenuniv.nl/~ewine/photo/radiation_fields.html).
        The file should be saved as `ISRF.dat` in the `Leiden_database` directory
        Several radiations files are available on this page, the standard ISRF works fine for most cases.

- The luminosity data. These data are much more difficult to obtain since they must be produced by Stellar simulation codes such as [cesam2k20](https://www.ias.u-psud.fr/cesam2k20/). The structure of this file will be described in a dedicated section. Without such files, the luminosity will be assumed to be a constant `Ls` that can be set in a configuration dictionary (see later) and the water photodissociation due to the central star will not be modelled.

#### The directory structure :

The main directory referred to as `racine` in the paths file is assumed to be the current working directory.
It is assumed that this main file contains a directory named 'Leiden_database', which contains the 'H2O.txt' and 'ISRF.dat' files.
If the user has data modelling the central star, it is assumed to be in a 'data_star' directory inside the main
`racine` directory, in a .npz file called `data_fus.npz`.
Finally, a 'Results' directory is needed to save the Sol objects.

#### The paths.txt file

If you don't want to work inside the 'racine' directory or want the data to be organised differently, you can write a 'paths.txt' file.
This file is structured as follows:
One line stands for one directory.
Each line starts with the name of the directory to be relocated; this can be 'racine', 'Results', 'data_leiden' or 'data_stars'.
Then a comma separates this name from the path to the new directory.

Example: Reproduce the default structure.

```
racine,.
results,./Results
data_leiden,./Leiden_database
data_stars,./data_stars/data_fus.npz
```

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
