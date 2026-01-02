"""
PES Package
Custom handlers, blueprints, and utilities for the PES platform
"""

from setuptools import setup, find_packages

setup(
    name="pes-mod",
    version="1.0.0",
    description="PES custom handlers, blueprints, and utilities",
    author="PES Team",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "requests>=2.32.0",  # For API calls
        "google-search-results>=2.4.2",  # SerpAPI for search functionality
    ],
    include_package_data=True,
    package_data={
        'pes': ['blueprints/*.json'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12",
    ],
)




