"""
Setup configuration for Two-Stage Ranking System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="two-stage-ranking",
    version="1.0.0",
    author="MLOps Engineer",
    author_email="mlops@atg.com",
    description="Production XGBoost Learning-to-Rank for High-Scale Auctions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aminajavaid30/RAG-Ingestion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-ranker=src.train:main",
            "serve-ranker=src.inference:main",
        ],
    },
)
