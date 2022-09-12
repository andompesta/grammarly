# build with ``python setup.py build_ext --inplace``

from setuptools import setup, Extension


class NumpyExtension(Extension):
    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs

extensions = [
    NumpyExtension(
        "data_utils_fast",
        sources=["data_utils_fast.pyx"],
        language="c++"
    ),
    NumpyExtension(
        "utils_token_block_fast",
        sources=["utils_token_block_fast.pyx"],
        language="c++"
    )
]

setup(
    name="dataset utils app",
    ext_modules=extensions,
    zip_safe=False
)