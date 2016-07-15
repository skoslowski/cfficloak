# Copyright (c) 2016, Andrew Leech <andrew@alelec.net>
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The full license is also available in the file LICENSE.apache-2.0.txt

import os
from setuptools import setup

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ['cfficloak.py']

with open(os.path.join(os.path.dirname(__file__), 'README.txt')) as readme:
    long_description = readme.read()
    
setup(name = 'cfficloak',
    version = '0.1',
    description = 'A simple but flexible module for creating '
                  'object-oriented, pythonic CFFI wrappers.',
    long_description = long_description,
    author = 'Andrew Leech',
    author_email = 'andrew@alelec.net',
    url = 'https://github.com/andrewleech/cfficloak',
    py_modules = ['cfficloak']
) 
