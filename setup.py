from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "*", ["egt/game_functions/*.pyx"],
        include_dirs=[numpy.get_include()],
        # libraries=[...],
        # library_dirs=[...]
        extra_compile_args=['-O3'],
    ),
    Extension(
        "*", ["egt/minimization_cy.pyx"],
        include_dirs=[numpy.get_include()],
        # include_dirs=[...],
        # libraries=[...],
        # library_dirs=[...]
        extra_compile_args=['-O3'],
    ),
    Extension(
        "*", ["egt/*.pyx"],
        include_dirs=[numpy.get_include()],
        # include_dirs=[...],
        # libraries=[...],
        # library_dirs=[...]
        extra_compile_args=['-O3'],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        annotate=True),
    include_dirs=[numpy.get_include()]
)
