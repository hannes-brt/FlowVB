from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("cython_utils", ["cython_utils.pyx"],
                         include_dirs=[np.get_include()])]

setup(
  name='FlowVB',
  cmdclass={'build_ext': build_ext},
  ext_modules=ext_modules
)
