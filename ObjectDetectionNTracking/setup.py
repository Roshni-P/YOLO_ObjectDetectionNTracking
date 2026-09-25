import os
import sys
import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Detect active Conda environment path dynamically
conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
conda_lib = os.path.join(conda_prefix, "lib")
conda_include = os.path.join(conda_prefix, "include")

# Include directories for OpenCV and ONNX Runtime
include_dirs = [
    "include",
    conda_include,
    os.path.join(conda_include, "opencv4"),
    os.path.join(conda_include, "opencv4", "opencv2"),
    # Standard Conda ONNX Runtime header directory
    os.path.join(conda_include, "onnxruntime"),
    # Some conda distributions place headers directly or inside core/session
    os.path.join(conda_include, "onnxruntime", "core", "session"),
]

# Shared library search path
library_dirs = [
    conda_lib,
]

# Libraries to link against (.so files)
libraries = [
    # OpenCV binaries
    "opencv_core",
    "opencv_imgproc",
    "opencv_dnn",
    "opencv_imgcodecs",
    # ONNX Runtime binary (links libonnxruntime.so)
    "onnxruntime",
]

# Compiler & Linker Flags
extra_compile_args = ["-O3", "-std=c++17", "-fPIC"]
extra_link_args = [
    f"-Wl,-rpath,{conda_lib}",
    "-Wl,-rpath,$ORIGIN",
]

sources = glob.glob("src/*.cpp")

ext_modules = [
    Pybind11Extension(
        "yoloDetector",
        sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
    Pybind11Extension(
        "queueUtils",
        ["src/processQueue.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),    
]

setup(
    name="yoloDetector",
    version="0.1.0",
    description="YOLOv8 C++ ONNX Runtime Detector with Pybind11 for Conda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)