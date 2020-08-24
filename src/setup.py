from setuptools import setup, find_packages

setup(
    name="PhonationModeling",
    version="0.1.0",
    description="A package for modeling phonation. "
    "Contains asymmetric vocal fold model and vocal tract model, "
    "solvers and optimizers for estimating model parameters.",
    author="Wenbo Zhao",
    author_email="wzhao1@andrew.cmu.edu",
    packages=find_packages(include=["PhonationModeling", "PhonationModeling.*"]),
    package_dir={"": "."},
    package_data={"PhonationModeling": ["*.lst", "*.json"]},
    install_requires=[""],
    python_requires=">=3.6",
    setup_requires=["flake8"],
    scripts=[],
    entry_points={
        "console_scripts": [
            "vocal_fold_estimate = PhonationModeling.main_scripts.vocal_fold_estimate",
            "vocal_tract_estimate = PhonationModeling.main_scripts.vocal_tract_estimate",
        ]
    },
)
