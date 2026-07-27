from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_sm75',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name='flash_attn_sm75_cuda',
            sources=[
                'src/bindings/pytorch_extension.cpp',
                'src/kernels/naive_attention.cu',
                'src/kernels/flash_attention_forward.cu',
            ],
            extra_compile_args={
                'nvcc': [
                    '-O3',
                    '-arch=sm_75',
                    '--use_fast_math',
                    '-lineinfo',
                ],
                'cxx': ['-O3'],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
