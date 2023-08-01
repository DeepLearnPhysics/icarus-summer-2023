from skbuild import setup  # This line replaces 'from setuptools import setup'
import argparse

import io,os,sys
this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="pfmatch",
    version="0.0.1",
    include_package_data=True,
    author=['Carolyn Smithhhhh'],
    author_email='carsmith@stanford.edu',
    description='Pytorch Flash Matching (PFMatch)',
    license='MIT',
    keywords='lartpc optical reconstruction flash-matching',
    project_urls={
        'Source Code': 'https://github.com/DeepLearnPhysics/icarus-summer-2023'
    },
    url='https://github.com/DeepLearnPhysics/icarus-summer-2023',
    scripts=['bin/run.py'],
    packages=['pfmatch','data.config'],
    package_dir={'': ''},
    package_data={'data': ['config/*.yml']},
    install_requires=[
        'numpy',
        'scikit-build',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)