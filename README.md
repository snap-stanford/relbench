# rtb

Steps to set up for development: (I recommend `mamba` <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html> instead of `conda` as it is much faster, but the CLI is the same)

```bash
# clone repo
git clone https://github.com/snap-stanford/rtb
cd rtb

# create a conda/mamba environment from env.yaml
mamba env create -f env.yaml --force
conda activate rtb

# install pre-commit hooks
pre-commit install

# test that everything works
pytest
```

To add dependencies for now, edit the `env.yaml` file manually, and then update the environment as follows:
```bash
mamba env update -f env.yaml
```

Make sure to commit the updated `env.yaml` to main to ensure reproducibility for others.

At release, we will move dependencies to `pyproject.toml` and remove `env.yaml` from the repo.
