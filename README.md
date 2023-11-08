# rtb

```bash
conda create --name rtb python=3.10 flit
conda activate rtb

git clone https://github.com/snap-stanford/rtb
cd rtb
flit install --symlink
flit publish  # will put it on PyPI
```
