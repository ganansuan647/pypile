#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fortran compilation utilities for PyPile tests
"""

import subprocess
import sys
from pathlib import Path
import importlib.util


def compile_fortran_module(fortran_file_path, module_name=None, working_dir=None):
    """
    Compile a Fortran file to a Python module using numpy.f2py

    Parameters
    ----------
    fortran_file_path : str or Path
        Path to the Fortran file to compile
    module_name : str, optional
        Name of the output Python module. If None, the base name of the Fortran file is used.
    working_dir : str or Path, optional
        Directory to run f2py in. If None, the directory of the Fortran file is used.

    Returns
    -------
    module : module
        The compiled Python module
    """
    fortran_file_path = Path(fortran_file_path)

    # If module_name is not provided, use the Fortran file's base name
    if module_name is None:
        module_name = fortran_file_path.stem + "_fortran"

    # If working_dir is not provided, use the directory of the Fortran file
    if working_dir is None:
        working_dir = fortran_file_path.parent
    else:
        working_dir = Path(working_dir)

    # Compile the Fortran code using f2py
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                str(fortran_file_path),
                "-m",
                module_name,
            ],
            cwd=str(working_dir),
            check=True,
        )
        print(f"Fortran module {module_name} compiled successfully")

        # Load the compiled module
        # Find the pyd file (Windows) or so file (Linux/Mac)
        if sys.platform == "win32":
            module_pattern = f"{module_name}*.pyd"
        else:
            module_pattern = f"{module_name}*.so"

        module_files = list(working_dir.glob(module_pattern))
        if not module_files:
            raise ImportError(
                f"Compiled module {module_name} not found in {working_dir}"
            )

        module_path = str(module_files[0])
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    except subprocess.CalledProcessError as e:
        print(f"Error compiling Fortran code: {e}")
        raise
