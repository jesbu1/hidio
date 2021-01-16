from setuptools import setup
import os

setup(
    name='hidio',
    version='1.0',
    install_requires=[
        'pybind11==2.5.0',
        'alf@git+https://github.com/horizonrobotics/alf.git@1146c4c78aef06a958c1f0c1d83be6645b11cc31'
    ],
)
os.system("pip install -e ./hidio/cnest")