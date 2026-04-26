"""
Setup configuration for Neural Networks Visualizer package
"""
"""Install dependencies"""
from setuptools import setup, find_packages

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # Basic information
    name="neural-networks-visualizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Interactive visualization tool for neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sinu-ops/neural-networks-visualizer",
    
    # Project location
    packages=find_packages(),
    
    # Classifiers help users find your project
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Minimum Python version
    python_requires=">=3.7",
    
    # Dependencies required to run the project
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tensorflow>=2.7.0",
        "keras>=2.7.0",
        "plotly>=5.0.0",
    ],
    
    # Optional dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.9",
        ],
        "jupyter": [
            "jupyterlab>=3.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Tracker": "https://github.com/sinu-ops/neural-networks-visualizer/issues",
        "Documentation": "https://github.com/sinu-ops/neural-networks-visualizer/blob/main/README.md",
        "Source Code": "https://github.com/sinu-ops/neural-networks-visualizer",
    },
)