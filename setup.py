from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
ABOUT = {}
exec((ROOT / "pysuqu" / "version.py").read_text(encoding="utf-8"), ABOUT)
README = (ROOT / "README.md").read_text(encoding="utf-8")


setup(
    name="pysuqu",
    version=ABOUT["__version__"],
    author="Naibin Zhou",
    author_email="zhnb@mail.ustc.edu.cn",
    url="https://github.com/znb888/pysuqu",
    description="Python toolkit for superconducting qubit simulation.",
    long_description=README,
    long_description_content_type="text/markdown",
    license="GNU Affero General Public License v3 or later (AGPLv3+)",
    license_files=["LICENSE"],
    packages=find_packages(include=("pysuqu", "pysuqu.*")),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
        "plotly>=5.0.0",
        "qutip>=4.7.0",
        "scipy>=1.7.0",
    ],
    project_urls={
        "Source": "https://github.com/znb888/pysuqu",
        "Documentation": "https://github.com/znb888/pysuqu/tree/main/docs",
        "Issues": "https://github.com/znb888/pysuqu/issues",
    },
    keywords=[
        "superconducting qubit",
        "quantum simulation",
        "transmon",
        "decoherence",
    ],
    classifiers=[
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

