from setuptools import setup, find_packages

setup(
    name='Energy_Landscape_Analysis',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[ #Installed automatically : https://packaging.python.org/ja/latest/discussions/install-requires-vs-requirements/
        "numpy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scipy",
        "pyvis",
        "networkx",
        "tqdm"
    ],
)
