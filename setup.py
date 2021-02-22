"""
Harvey Classifier python package configuration.

Aditi Khare
"""

from setuptools import setup

setup(
    name='harvey_classifier',
    version='0.1.0',
    packages=['harvey_classifier'],
    include_package_data=True,
    install_requires=[
        'os',
        'sklearn',
        'numpy',
        'matplotlib',
        'click',
        'stop_words',
        'nltk'
    ],
    python_requires='>=3.6',
)

