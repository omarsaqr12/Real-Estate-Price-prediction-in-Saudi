"""Minimal source packaging metadata; this does not bundle model or dataset assets."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]

setup(
    name="saudi-real-estate-prediction",
    version="0.1.0",
    author="Omar Saqr and Aabed Elghadbaan",
    description="Research prototype for Saudi real-estate price prediction",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)
