#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：zk_server 
@File    ：setup.py
@Author  ：gjg
@Date    ：2025/10/14 19:33 
@explain : init
'''
# from setuptools import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize("viewsx.py")  # 指定要编译的.py文件
# )

from setuptools import setup, Extension
from Cython.Build import cythonize

# 定义扩展模块，添加编译器参数
ext_modules = [
    Extension(
        "views",  # 模块名（与.py文件同名）
        sources=["viewsx.py"],  # 源文件
        extra_compile_args=["-std=c99"]  # 关键：强制使用C99标准
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)