from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("src/evaluation/algorithms/coloring_cy.pyx",
compiler_directives={'boundscheck': False, 'wraparound': False}
                            )
)