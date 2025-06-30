from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "pfwrapper",
        sources=[
            "pf_wrapper.pyx",
            "pf_src/pattern_finder.cpp",
            "pf_src/search.cpp",
        ],
        include_dirs=["pf_src", numpy.get_include()],
        language="c++",
    )
]

setup(
    name="pfwrapper",
    ext_modules=cythonize(extensions),
)
