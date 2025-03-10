# setup.py
"""
Setup script for the pypile package.
"""

from setuptools import setup, find_packages

setup(
    name="pypile",
    version="1.0.0",
    description="Spatial Static Analysis of Pile Foundations",
    author="Converted from Fortran by Claude",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "art>=6.4",
        "loguru>=0.7.3",
        "matplotlib>=3.10.1",
        "numpy>=2.2.3",
        "plotly>=6.0.0",
        "pydantic>=2.10.6",
    ],
    entry_points={
        "console_scripts": [
            "pypile=pypile.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Civil Engineering",
    ],
    python_requires=">=3.8",
)
