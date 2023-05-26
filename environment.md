# Export your active environment to a new environment.yml file:

conda/mamba/micromamba env export > environment.yml

# Create the environment from the environment.yml file:

conda/mamba/micromamba env create -f environment.yml

# You can also use explicit specification files to build an identical conda environment on the same operating system platform, either on the same machine or on a different machine.

conda list --explicit > spec-file.txt
mamba/micromamba env export --explicit > spec-file.txt

# To use the spec file to create an identical environment on the same machine or another machine:

conda/mamba/micromamba create --name myenv --file spec-file.txt

# To use the spec file to install its listed packages into an existing environment:

conda/mamba/micromamba install --name myenv --file spec-file.txt
