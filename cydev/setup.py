from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension('crystalgrowth',
                sources=["crystalgrowth.pyx"],
                libraries=["m"]
    )
]

setup(name='crystalgrowth',
        ext_modules=cythonize(ext_modules),
        include_dirs=[np.get_include()])
