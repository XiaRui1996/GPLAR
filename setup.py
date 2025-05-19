from setuptools import setup, find_packages

requirements = [
    "numpy>=1.16",
    "torch>=1.7",
    "wbml>=0.3",
    "plum-dispatch>=2.5.4",
    "backends>=1",
    "backends-matrix>=1",
    "stheno>=1.1",
    "varz>=0.6",
    'gpflow>=2.5.2',
    'tensorflow>=2.9',
    'tensorflow-probability>=0.17',
    'gpar',
    'scipy',
    'pandas',
]

setup(
    name="gplar",
    version="0.1.0",
    description="GPLAR: Gaussian Process Latent Auto-regressive Regression",
    author='Rui Xia',
    url='https://github.com/XiaRui1996/GPLAR',
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.8",
    install_requires=requirements,
)