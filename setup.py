from pathlib import Path

import os

from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent
ABOUT = {}
exec((ROOT / "pysuqu" / "version.py").read_text(encoding="utf-8"), ABOUT)
README = (ROOT / "README.md").read_text(encoding="utf-8")


def native_extensions():
    """Enable native propagation explicitly while keeping compiler-free installs."""
    if os.environ.get("PYSUQU_BUILD_NATIVE", "0").lower() not in {"1", "true", "yes", "on"}:
        return []
    compile_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17", "-pthread"]
    return [Extension(
        "pysuqu._native._dynamics",
        sources=["native/dynamics.cpp"],
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=[] if os.name == "nt" else ["-pthread"],
    )]


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
    ext_modules=native_extensions(),
    python_requires=">=3.9",
    install_requires=[
        "matplotlib>=3.4.0",
        "numpy>=1.20.0",
        "plotly>=5.0.0",
        "qutip>=5.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.0.0",
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
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

