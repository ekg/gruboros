from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nau_gru_cuda',
    ext_modules=[
        CUDAExtension('nau_gru_cuda', [
            'nau_gru_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)