from setuptools import setup, find_packages

setup(
    name="randatoms",
    version="0.1.0",
    description="A package for handling datasets of atomic structures with tools for conversion, merging, and loading.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/your-repo/randatoms",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'randatoms.dataset': ['*.h5', '*.pkl'],
    },
    install_requires=[
        'h5py>=3.0.0',
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'ase>=3.22.0',
        'tqdm>=4.60.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'black>=22.0.0',
            'isort>=5.10.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
