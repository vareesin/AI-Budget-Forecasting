from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-budget-forecasting",
    version="1.0.0",
    author="Varees Adulyasas",
    author_email="vareesadulyasas@gmail.com",
    description="AI-powered budget forecasting system for store profits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/AI-Budget-Forecasting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "budget-forecast=src.prediction:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md"],
        "data": ["*.xlsx"],
    },
)