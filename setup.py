"""
Setup script for AlphaScrabble.

This file is used by setuptools to build and install the package.
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import cmake
import subprocess
import os
from pathlib import Path

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

class CMakeExtension(Extension):
    """CMake extension for pybind11."""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """Build extension with CMake."""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        debug = int(os.environ.get('DEBUG', 0)) if self.debug is None else self.debug
        cfg = 'Debug' if debug else 'Release'
        
        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={self.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
            f'-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        ]
        
        # Add pybind11 include
        cmake_args += [f'-Dpybind11_DIR={pybind11.get_cmake_dir()}']
        
        build_args = ['--config', cfg]
        
        # Add build type specific flags
        if cfg == 'Release':
            cmake_args += ['-DCMAKE_CXX_FLAGS=-O3 -fPIC']
        else:
            cmake_args += ['-DCMAKE_CXX_FLAGS=-g -fPIC']
        
        # Set build type
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Configure
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        
        # Build
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

# Check if Quackle is available
def check_quackle():
    """Check if Quackle is available for building."""
    quackle_path = Path("third_party/quackle")
    if not quackle_path.exists():
        print("Warning: Quackle not found. C++ extension will not be built.")
        print("Run 'git clone https://github.com/quackle/quackle.git third_party/quackle' to enable C++ support.")
        return False
    return True

# Setup configuration
setup(
    name="alphascrabble",
    version="0.1.0",
    author="AlphaScrabble Team",
    author_email="alphascrabble@example.com",
    description="AlphaZero-style Scrabble engine with MCTS and neural networks",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/alphascrabble/alphascrabble",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "alphascrabble=alphascrabble.cli:main",
        ],
    },
    ext_modules=[
        CMakeExtension("qlex", "cpp") if check_quackle() else None
    ],
    cmdclass={"build_ext": CMakeBuild} if check_quackle() else {},
    zip_safe=False,
    include_package_data=True,
    package_data={
        "alphascrabble": ["*.so", "*.dll", "*.dylib"],
    },
)
