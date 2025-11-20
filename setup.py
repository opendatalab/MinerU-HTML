"""
Simplified setup.py - A balanced version with essential improvements.
"""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(file_path: Path) -> list[str]:
    """
    Read requirements from a file and return as a list.

    Args:
        file_path: Path to the requirements file

    Returns:
        List of requirement strings
    """
    if not file_path.exists():
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]


# Read main requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
install_requires = read_requirements(requirements_file)

# Read baseline requirements (optional dependencies for evaluation)
baselines_file = Path(__file__).parent / 'baselines.txt'
baselines_requires = read_requirements(baselines_file)

setup(
    name='dripper',
    version='1.0.0',
    description='HTML main content extractor based on large language models',
    packages=find_packages(include=['dripper*']),
    include_package_data=True,
    python_requires='>=3.12',
    install_requires=install_requires,
    extras_require={
        'baselines': baselines_requires,
    },
    license='Apache License 2.0',
)
