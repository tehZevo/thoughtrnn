from setuptools import setup, find_packages

setup(name='thoughtrnn',
  version='0.0.0',
  install_requires = [
    'tensorflow',
    'keras',
    'numpy',
  ],
  packages=find_packages())
