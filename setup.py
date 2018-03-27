from setuptools import setup

setup(name='edx_abs',
      version='0.1',
      description='Python scripts for performing absorption correction on 3D STEM-EDX data',
      url='http://github.com/TomSlater/edx_abs',
      author='Tom Slater',
      author_email='tjaslater@gmail.com',
      license='GPL-3.0+',
      packages=['edx_abs'],
      install requires=[
              'numpy',
              'scipy',
              'skimage',
              'matplotlib',
              'astra-toolbox',
              ],
      zip_safe=False)