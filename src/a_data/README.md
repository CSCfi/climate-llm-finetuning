# Data

Scientific climate-related articles from Copernicus are used in this project.

# How to

A few alternatives to run the code:
1. (Preferred method) To inspect and run the codes in a notebook-y fashion: 
    - First, clone the repo to /scratch/<project_xxx>/\<user>/
    - Head to LUMI web interface and launch the Marimo OOD app (currently available only in [testing version](https://ood-testing.lumi.csc.fi/public/), located in *My Interactive Sessions*)
        - Select the project and set the working directory to the project folder where this repo is cloned to (scratch). Marimo opens user's own folder under the project as the default workspace, so the cloned repo should appear in the file list in Marimo view.
        - Adjust the settings (Partition: small, number of CPUs: 14, memory: 60 GiB, time: 4:00:00, module: pytorch)

        ![Marimo LUMI](../../images/1_marimo.png)

    - After opening the marimo app, open the climate-llm-finetuning/src/a_data/data_gather.py file in the marimo main view
    - Once you have opened the notebook, there are a couple variables you can adjust in the first cell (DATA_AMOUNT and REMOVE_FILES)
    - Run the notebook

2. To only run the code, execute the [according slurm script](./data_gather.sh) from project root in the LUMI login node with command `sbatch src/a_data/data_gather.sh`

# Some known problems

Currently, as of 20.2.2026, programmatical access to journals in copernicus.org is restricted. We are planning on having the data stored in some storage service provided by CSC. If running the notebook, you may not be able to download all the articles as PDFs/XMLs.