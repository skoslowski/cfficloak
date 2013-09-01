#!/usr/bin/env python
# Copyright (c) 2013, Isaac Freeman <memotype@gmail.com>
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

from pytest import *

from wrapper import WrapObj

import cffi

def get_x(obj):
    return obj._x

def add_x(obj, y):
    return obj.x + y

class TestWrapper(WrapObj):
    _props = {'x': get_x}
    _meths = {'add': add_x}
    _x = 2

wrap = TestWrapper()
def test_attrs():
    assert wrap.x == wrap._x
def test_noattrs():
    with raises(AttributeError):
        wrap.y
def test_meths():
    assert wrap.add(3) == wrap._x + 3
def test_nomeths():
    with raises(AttributeError):
        wrap.sub
def test_methargs():
    with raises(TypeError):
        wrap.add(3, 4)
def test_methnoargs():
    with raises(TypeError):
        wrap.add()



