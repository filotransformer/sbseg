"""
Setup script for Filo-Transformer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="filo-transformer",
    version="1.0.0",
    author="Acauan Cardoso Ribeiro, Eduardo Luzeiro Feitosa, André Carvalho",
    author_email="acauan.ribeiro@ufrr.br",
    description="Filo-Transformer: Um modelo baseado em Grafo de Alinhamento de Árvores Filogenéticas e Transformers para Identificação de Rumores e Fake News",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/filotransformer/sbseg",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter",
            "matplotlib",
            "seaborn",
        ],
        "gpu": [
            "tensorflow-gpu>=2.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "filo-transformer=scripts.run_experiment:main",
            "filo-test=scripts.minimal_test:main",
        ],
    },
    keywords="fake news detection, phylogenetic analysis, transformers, graph neural networks, rumor detection",
    project_urls={
        "Bug Reports": "https://github.com/filotransformer/sbseg/issues",
        "Source": "https://github.com/filotransformer/sbseg",
        "Documentation": "https://github.com/filotransformer/sbseg/blob/main/README.md",
    },
)