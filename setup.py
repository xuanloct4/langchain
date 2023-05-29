#!/usr/bin/env python

import setuptools

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

core = [ 
]
gui = [
]
api = [
]
setuptools.setup(
    name='langchain_study',
    version='0.0.1',
    description='langchain_study',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Loc Tran',
    author_email='xuanloct4@gmail.com',
    url='https://github.com/xuanloct4/langchain.git',
    license='MIT',
    keywords=[
        'artificial intelligence',
        'chatbot',
        'ai',
        'deep learning',
    ],
    packages=setuptools.find_packages(),
    package_data={
        'langchain_study': [
            'resources/*',
            'model/settings/*.yaml',
            'model/dataset/*.json',
        ]
    },
    install_requires=core,
    extras_require={
        'all': gui+api,
        'gui': gui,
        'api': api
    },
    entry_points={
        'console_scripts': [
            'langchain_study_gui = langchain_study.gui:main',
            'langchain_study_cli = langchain_study.cli:main',
            'langchain_study = langchain_study.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
    ],
)
