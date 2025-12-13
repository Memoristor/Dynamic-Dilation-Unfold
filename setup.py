import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_extensions():
    extensions = []
    
    ext_modules = [
        CUDAExtension(
            'dynamic_dilation_unfold._ext',
            [
                'src/cuda/dynamic_dilation_unfold_cuda.cpp',
                'src/cuda/dynamic_dilation_unfold_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-DCUDA_HAS_FP16=1',
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        ),
    ]
    
    return ext_modules

setup(
    name='dynamic_dilation_unfold',
    version='1.0.1',
    author='cjdeng',
    description='Dynamic Dilation Unfold with CUDA acceleration',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.8.0',
        'numpy',
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
    ],
)