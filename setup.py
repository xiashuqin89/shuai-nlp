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

packages = find_packages(include=('nonebot', 'nonebot.*'))

setup(
    name='shuai-nlp',
    version='v1',
    url='https://github.com/xiashuqin89/shuai-nlp',
    license='MIT License',
    author='shuai',
    description='bot',
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
        'Typing :: Typed',
    ],
)
