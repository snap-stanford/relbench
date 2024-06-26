import os
from pathlib import Path
import shutil

import pooch

from relbench.data import Dataset, Task
from relbench.datasets.amazon import AmazonDataset

# TODO: use the versioning feature of pooch
pooch_registry = pooch.create(
    path=pooch.os_cache("relbench"),
    base_url="https://relbench.stanford.edu/staging_data/",  # TODO: change
    registry={
        # extremely small dataset only used for testing download functionality
        "rel-amazon-fashion_5_core/db.zip": "27e08bc808438e8619560c54d0a4a7a11e965b90b8c70ef3a0928b44a46ad028",
        "rel-amazon-fashion_5_core/tasks/rel-amazon-churn.zip": "d98f2240aefa0f175dab2fce4a48a1cc595be584d4960cd9eb750d012326117d",
        "rel-amazon-fashion_5_core/tasks/rel-amazon-ltv.zip": "bd2b7b798efad2838a3701def8386dba816b45ef277a8e831052b79f5448aed8",
        "rel-stackex/db.zip": "deb00ccdf825e569b34935834444429cd1c0074b50226b12d616aab22d36242d",
        "rel-stackex/tasks/rel-stackex-engage.zip": "9afce696507cf2f1a2655350a3d944fd411b007c05a389995fe7313084008d18",
        "rel-stackex/tasks/rel-stackex-votes.zip": "0dab5bebd76a95d689c8a3a62026c1c294a252c561fd940e8d9329d165d98a5a",
        "rel-amazon-books_5_core/db.zip": "2f6bd920bcfe08cbb7d47115f47f8d798a2ec1a034b6c2f3d8d9906e967454b4",
        "rel-amazon-books_5_core/tasks/rel-amazon-churn.zip": "d3890621b1576a9d5b6bc273cdd2ea2084aeaf9c8055c1421ded84be0c48dacb",
        "rel-amazon-books_5_core/tasks/rel-amazon-ltv.zip": "2e91be0ca5d9f591d8e33a40f70b97db346090a8bb9f3a94f49b147f0dc136be",
        "rel-trial/db.zip": "76093dae4365839cae4f949cc2c982c8c8ddf9886e309d84606b37208c8102da",
        "rel-math-stackex/db.zip": "00b193587f1ee0b39c77d2b561385f00fbc3b4ca0929f420e16ddd53115ce3ea",
        "rel-f1/db.zip": "e41ca0d69d54f16b408fe03b6c19b772ae701336cf84260ef5c84fca798a1422",
        "rel-hm/db.zip": "6ff6537f2fed885c5c8a94525364678dea206c57006de0edb4d76ca71c9c114e",
        "rel-avito/db.zip": "09fe913ece4f17f79ca0d2c1d25ed9f6f7e803fa4a08dcf520b7a0e73f34b1ed",
        "rel-event/db.zip": "141f4842600d091250c1f94e4c479c35e76d7ec3aef9155316f83d4828d85e5e",
        "rel-amazon/db.zip": "2fb5d1b6f0d8886374bc25b3a81becbe191ad30e524ac7fb998bab4c87010adc",
        "rel-stack/db.zip": "b703d141f86c210e9e6809807ec0bcf9b3e2fcd32a679835c9d71d8048b89188",
    },
)

dataset_registry = {}


def register_dataset(name, cls, *args, **kwargs):
    dataset_registry[name] = (cls, args, kwargs)


def unzip_processor(fname: str | os.PathLike, action: str, pooch: pooch.Pooch) -> Path:
    zip_path = Path(fname)
    unzip_path = zip_path.parent / zip_path.stem
    if action != "fetch":
        shutil.unpack_archive(zip_path, unzip_path)
    else:  # fetch
        try:  # sanity check if all files are fully extracted comparing size
            for f in ZipFile(zip_path).infolist():
                if not f.is_dir():
                    fsize = os.path.getsize(os.path.join(unzip_path, f.filename))
                    assert f.file_size == fsize
        except Exception:  # otherwise do full unpack
            shutil.unpack_archive(zip_path, unzip_path)

    return unzip_path


def get_dataset(name: str, download=False) -> Dataset:
    if download:
        pooch_registry.fetch(
            f"{name}/db.zip",
            processor=unzip_processor,
            progressbar=True,
        )
    cache_dir = f"{pooch.os_cache('relbench')}/{name}"

    cls, args, kwargs = dataset_registry[name]

    return cls(*args, cache_dir=cache_dir, **kwargs)


def get_task(dataset_name: str, task_name: str, download=False) -> Task:
    if download:
        pooch_registry.fetch(
            f"{dataset_name}/tasks/{task_name}.zip",
            processor=unzip_processor,
            progressbar=True,
        )
    dataset = get_dataset(dataset_name, download=download)
    cache_dir = f"{dataset.cache_dir}/tasks/{task_name}"
    if task_name == "user-churn":
        return UserChurnTask(dataset=dataset, cache_dir=cache_dir)
