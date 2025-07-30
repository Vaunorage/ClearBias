from setuptools import setup
from setuptools.extension import Extension

# Conditional Cython import to handle build environment
try:
    from Cython.Build import cythonize
    import numpy
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython not available, skipping extension build")

if CYTHON_AVAILABLE:
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
    ext_modules = cythonize(extensions)
else:
    ext_modules = []

setup(
    name="pfwrapper",
    version="0.1.0",
    ext_modules=ext_modules,
    setup_requires=[
        "cython>=3.1.2",
        "numpy>=1.21.0",
    ],
    install_requires=[
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
)
