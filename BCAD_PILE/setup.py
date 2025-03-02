# setup.py
"""
Setup script for the BCAD_PILE package.
"""

from setuptools import setup, find_packages

setup(
    name="bcad_pile",
    version="1.0.0",
    description="Spatial Static Analysis of Pile Foundations",
    author="Converted from Fortran by Claude",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "numba>=0.53.0",
        "plotly>=5.0.0"
    ],
    entry_points={
        'console_scripts': [
            'bcad_pile=bcad_pile.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Civil Engineering",
    ],
    python_requires=">=3.7",
)