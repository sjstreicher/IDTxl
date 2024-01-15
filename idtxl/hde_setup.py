import numpy
from Cython.Build import cythonize
from setuptools import setup

# to compile, run
# python3 hde_setup.py build_ext --inplace

setup(
    name="hde_fast_embedding",
    ext_modules=cythonize(["hde_fast_embedding.pyx"], annotate=False),
    include_dirs=[numpy.get_include()],
)
