# Copyright (C) 2021 Jean Da Costa machado.
# Jean3dimensional@gmail.com
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.


from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.util import get_platform
import re
from sys import argv
from os import path

debug = 'debug' in argv
if debug:
    argv.remove('debug')

platf = get_platform().lower()
print(f'compiling for platform: {platf}')
if 'linux' in platf:
    extra_compile_args = ['-ffast-math', '-fopenmp', '-O3'] + (['-g3'] if debug else [])
    extra_link_args = ['-fopenmp'] + (['-g3'] if debug else [])

elif 'win' in platf:
    extra_compile_args = ['/O2', '/openmp', '/fp:fast'] + (['/debug'] if debug else [])
    extra_link_args = ['/openmp'] + (['/debug'] if debug else [])

else: # mac
    extra_compile_args = ['-ffast-math', '-O3', '-std=c++2a'] + (['-g3'] if debug else [])
    extra_link_args = ['-g3'] if debug else []



def line_directive_add(filestr):
    modfied_tag = '/* Postprocessed to add #line directives */'
    if filestr.startswith(modfied_tag):
        return filestr

    lines = filestr.split('\n')
    line_comment_re = re.compile(r"^\s*\/\*\s*\"(.+)\":(\d+)")
    define_re = re.compile(r'\s*#\s*define')

    curr_line = None
    curr_file = None
    in_comment = False
    in_define = False

    new_lines = [modfied_tag]
    for line in lines:
        if "/*" in line:
            in_comment = True
            curr_file = None
            curr_line = None

        match = line_comment_re.match(line)
        if match:
            filename = match.group(1)
            line_num = match.group(2)
            if path.isfile(filename):
                # curr_file = path.abspath(filename)
                curr_file = filename
                curr_line = line_num

        if define_re.match(line):
            in_define = True

        if not in_comment and not in_define and line.strip():
            if curr_file:
                new_lines.append(f'\n# line {curr_line} "{curr_file}"')

        if '*/' in line:
            in_comment = False

        if not line.endswith('\\'):
            in_define = False

        new_lines.append(line)

    return '\n'.join(new_lines) + '\n'


modules = cythonize(
    Extension(
        'softwrap_core2',
        # ['core/core.pyx'],
        ['core/core2.pyx'],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=["core"]
    )
)

print('adding #line directives')
with open('core/core2.cpp', 'r+') as f:
    new_filestr = line_directive_add(f.read())
    f.seek(0)
    f.truncate()
    f.write(new_filestr)

setup(
    name='softwrap_core2',
    ext_modules=modules
)
