from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setup(
    name="PyPortOpt",
    version="0.2.0",
    license="MIT",
    author="Weichuan Deng, Ronak Shah, Fabio Robayo",
    author_email="weichuan.deng@stonybrook.edu, ronak.shah@stonybrook.edu",
    url="https://github.com/PawPol/ML_in_QF_AMS520",
    description="A Python library that implements various Portfolio Optimization method",
    packages = find_packages(),
    install_requires=[req for req in requirements],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.5",
)
