from io import open

from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

# Note: removed all identity information from this file

setup(name='npe_convergence',
      version='0.0.1',
      description='Experiments to assess NPE convergence',
      license='GPL',
      packages=['npe_convergence'],
      zip_safe=False,
      python_requires='>=3.7',
      install_requires=requirements
      )