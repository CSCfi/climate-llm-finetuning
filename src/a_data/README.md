# Data

Scientific climate-related articles from Copernicus are used in this project.

# How to

A few alternatives to run the code:
1. (Preferred method) To inspect and run the codes in a notebook-y fashion: 
    - Head to LUMI web interface and launch the Marimo OOD app
        - Select the project and set the working directory to the project folder where this repo is cloned to
        - Adjust the settings (Memory 60 GiB, time 2:00:00)

        ![Marimo LUMI](../../images/1_marimo.png)

    - After opening the marimo app, open the 1_data.py file in the marimo main view
    - Run the notebook

2. To only run the code, execute the [according slurm script](./data_gather.sh) from project root in the LUMI login node with command `sbatch src/a_data/data_gather.sh`

# Some known problems

Currently, as of 20.2.2026, programmatical access to journals in copernicus.org is restricted. We are planning on having the data stored in some storage service provided by CSC.