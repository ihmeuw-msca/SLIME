from setuptools import setup
from setuptools import find_packages

setup(
    name='slime',
    version='0.0.0',
    description='Simple linear mixed effects model'
                'with priors and constraints.',
    url='https://github.com/zhengp0/SLIME',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pytest',
    ],
    zip_safe=False,
)
