from pathlib import Path

import os
from glob import glob
import importlib.util
import sys

from setuptools import Extension, find_packages, setup

ROOT = Path(__file__).resolve().parent
_BUILD_OPTIONS_PATH = ROOT / "pysuqu" / "_native_build_options.py"
_BUILD_OPTIONS_SPEC = importlib.util.spec_from_file_location(
    "pysuqu_native_build_options", _BUILD_OPTIONS_PATH,
)
_BUILD_OPTIONS_MODULE = importlib.util.module_from_spec(_BUILD_OPTIONS_SPEC)
sys.modules[_BUILD_OPTIONS_SPEC.name] = _BUILD_OPTIONS_MODULE
_BUILD_OPTIONS_SPEC.loader.exec_module(_BUILD_OPTIONS_MODULE)
native_build_options = _BUILD_OPTIONS_MODULE.native_build_options
ABOUT = {}
exec((ROOT / "pysuqu" / "version.py").read_text(encoding="utf-8"), ABOUT)
README = (ROOT / "README.md").read_text(encoding="utf-8")


def native_extensions():
    """Enable native propagation explicitly while keeping compiler-free installs."""
    if os.environ.get("PYSUQU_BUILD_NATIVE", "0").lower() not in {"1", "true", "yes", "on"}:
        return []
    options = native_build_options()
    family = options["compiler_family"] or ("msvc" if os.name == "nt" else "gnu")
    if options["march_native"] and family == "msvc":
        raise RuntimeError("PYSUQU_NATIVE_MARCH_NATIVE is unsupported with MSVC")
    compile_args = ["/O2", "/EHsc", "/std:c++17"] if family == "msvc" else ["-O3", "-std=c++17", "-pthread"]
    link_args = [] if family == "msvc" else ["-pthread"]
    if options["march_native"]:
        compile_args.append("-march=native")
    if options["pgo"] == "generate":
        flag = "-fprofile-generate" + (f"={options['profile_dir']}" if options["profile_dir"] else "")
        compile_args.append(flag)
        if family != "msvc":
            link_args.append(flag)
    elif options["pgo"] == "use":
        flag = "-fprofile-use" + (f"={options['profile_dir']}" if options["profile_dir"] else "")
        compile_args.extend([flag, "-fprofile-correction"])
        if family != "msvc":
            link_args.append(flag)
    native_details = sorted(glob("native/detail/*.inc"))
    return [Extension(
        "pysuqu._native._dynamics",
        sources=["native/dynamics.cpp"],
        depends=native_details,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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

