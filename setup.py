from setuptools import setup

setup(
    name='uavf_2025',
    version='0.1',
    package_dir = {'': 'src'},
    install_requires=[
        'numpy',
        'opencv-python',
    ],
)
