from setuptools import setup, find_packages

setup(
    name="stochastic-gdp-forecasting",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Stochastic optimization for GDP nowcasting",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stochastic-gdp-forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "cvxpy>=1.4.0",
    ],
    extras_require={
        "bayesian": ["pymc>=5.10.0", "arviz>=0.17.0"],
        "full": ["pymc>=5.10.0", "arviz>=0.17.0", "fredapi>=0.5.1", "wbgapi>=1.0.12"],
    },
)
