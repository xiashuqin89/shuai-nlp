import os

from setuptools import setup, find_packages


this_directory = os.path.abspath(os.path.dirname(__file__))


def read_file(filename):
    with open(os.path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]


with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


packages = find_packages(include=('src', 'src.*'))

requires = [
    'PyYAML==5.4.1',
    'simplejson==3.17.6',
    'jsonschema==3.2.0',
    'packaging==21.3',
    'six==1.16.0',
    'spacy>2.0',
    'sklearn',
    'cloudpickle==2.0.0',
    'future==0.18.2'
]


setup(
    name='bkchat-nlp',
    version='1.0.1',
    license='MIT License',
    author='neo',
    description='nlp',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=packages,
    package_data={
        '': ['*.pyi', 'py.typed'],
    },
    install_requires=read_requirements('requirements.txt'),
    extras_require={
        'scheduler': ['apscheduler'],
    },
    python_requires='>=3.6',
    platforms='any',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Framework :: Robot Framework',
        'Framework :: Robot Framework :: Library',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
)