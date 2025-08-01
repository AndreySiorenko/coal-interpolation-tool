from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="coal-interpolation",
    version="0.1.0",
    author="Coal Interpolation Team",
    author_email="support@coal-interpolation.com",
    description="Interactive program for coal deposit data interpolation from borehole surveys",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/coal-interpolation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "coal-interpolation=src.main:main",
        ],
    },
)