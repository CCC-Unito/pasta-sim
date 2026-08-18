from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="pasta-sim",
    version="0.1",
    include_package_data=True,
    python_requires='>=3.12',
    packages=find_packages(),
    setup_requires=['setuptools-git-versioning'],
    install_requires=requirements,
    author="Valerio BASILE",
    author_email="valerio.basile@unito.it",
    description="A simulation tool for synthetic Perspectivist Annotation of Subjective TAsks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: GPL",
        "Operating System :: OS Independent",
    ],
    version_config={
       "dirty_template": "{tag}",
    }
)
