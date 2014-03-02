#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

readme = open('README.rst').read()
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='twcounty',
    version='0.1.0',
    description='Twitter County Analyzer',
    long_description=readme + '\n\n' + history,
    author='Aron Culotta',
    author_email='aronwc@gmail.com',
    url='https://github.com/aronwc/twcounty',
    entry_points={
        'console_scripts': [
            'twcounty-expt = twcounty.expt:main',
            'twcounty-json2tsv = twcounty.json2tsv:main',
            'twcounty-tsv2feats = twcounty.tsv2feats:main',
            'twcounty-tsv2stats = twcounty.tsv2stats:main',
        ],
    },

    packages=[
        'twcounty',
    ],
    package_dir={'twcounty': 'twcounty'},
    include_package_data=True,
    install_requires=[
    ],
    license="BSD",
    zip_safe=False,
    keywords='twcounty',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
    ],
    test_suite='tests',
)
