# rtb

Steps to set up for development:

```bash
# create a conda environment
conda create --name rtb python=3.8 flit
conda activate rtb

# clone repo
git clone https://github.com/snap-stanford/rtb
cd rtb

# editable install of package (similar to `pip install -e .`)
flit install --symlink

# install pre-commit hooks
pre-commit install
```
