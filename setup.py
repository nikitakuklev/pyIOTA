from setuptools import setup, find_packages
import pyiota


INSTALL_REQUIRES = [
    'numpy>=1.13.3',
    'pandas>=1.0',
    'pydantic>=1.10',
    'scipy',

]
EXTRAS_REQUIRES = {
    "develop": [
        "pytest>=6.0",
    ]
}
LICENSE = "GNU General Public License v3 or later (GPLv3+)"
DESCRIPTION = 'Toolbox for computational and experimental accelerator physics'
CLASSIFIERS = [
    "Programming Language :: Python :: 3.9",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
]

setup(
   name='pyiota',
   version=pyiota.__version__,
   author='Nikita Kuklev',
   author_email='',
   description=DESCRIPTION,
   license=LICENSE,
   url="https://github.com/nikitakuklev/pyIOTA",
   packages=find_packages(),
   platforms="any",
   install_requires=INSTALL_REQUIRES,
   python_requires=">=3.9",
   extras_require=EXTRAS_REQUIRES
)