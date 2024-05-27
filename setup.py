from setuptools import find_packages, setup

setup(
    name="relbench",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "pooch",
        "pyarrow",
        "numpy",
        "duckdb",
        "requests",
        "tqdm",
        "scikit-learn",
        "typing-extensions",
    ],
    extras_require={
        "example": [
            "torch",
            "pytorch_frame>=0.2.2",
            "torch_geometric",
            "faiss-cpu",
            "sentence-transformers",
            "tensorboard",
        ],
        "test": ["pytest"],
        "dev": ["pre-commit"],
    },
)
