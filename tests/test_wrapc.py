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

# The level of testing here might seem a bit anal, but it was pretty tricky
# getting all the metaclass stuff, and the multiple layers of function wrapping
# and indirection just right, especially with inheritence of dictionary keys, 
# etc. :P

from IPython.core.debugger import Pdb

import os

from pytest import *

import cffi
import cffiwrap as wrap


### FFI boilerplate ###

ffi = cffi.FFI()

ffi.cdef('''
int myint_succ(int i);
int myint_succ2(int i);
int myint_doubled(int i);
int myint_add(int i, int j);
int myint_add2(int i, int j);
int myint_mult(int i, int j);
int* myintp_null(int i);

float myfloat_add(float i, float j);
float myfloat_succ(float i);
float* myfloatp_null(float i);

int set_ptr_succ(int i, int *j);
int set_ptr_add(int i, int *j);

double complicated(int in, 
                   float *out,
                   int *inout,
                   unsigned long long in2,
                   double *inout2);

int myint_add_array(int j, int *a, int n);

typedef struct { 
    int x;
    int y;
} point_t;

point_t* make_point(int x, int y);
void del_point(point_t* p);
int point_x(point_t* p);
int point_y(point_t* p);
point_t* point_setx(point_t* p, int x);
point_t* point_sety(point_t* p, int y);
double point_dist(point_t* p1, point_t* p2);
''')

srcpath = os.path.dirname(os.path.abspath(__file__))
api = ffi.verify('#include "test.h"',
                 include_dirs=[srcpath],
                 library_dirs=[srcpath],
                 runtime_library_dirs=[srcpath],
                 libraries=['test'],
                 depends=[os.path.join(srcpath, 'test.h')])

# You could also import all of the functions in to the current module with:
# globals().update(wrap.CFunction.wrapall(ffi, myapi))
# This is useful to put all of the "raw" C functions in to a module (or sub-
# module) within your package.
cfuncs = wrap.CFunction.wrapall(ffi, api)


class MyError(Exception): pass

class MyInt(wrap.CObject):
    def __init__(self, i):
        self.i = i
        self._cdata = i
        super(MyInt, self).__init__()
    def _checkerr(self, cfunc, args, retval):
        ''' Checks for NULL return values and raises MyError. '''
        if retval == cffi.FFI.NULL:
            raise MyError('NULL returned by {0} with args {1}. '
                            .format(cfunc.cname, args, retval))
        else:
            return retval
            

### Basic wrapper tests ###

class MyInt1(MyInt):
    _props = {
        'succ': cfuncs['myint_succ'],
    }
    _meths = {
        'add': cfuncs['myint_add'],
        's_add': staticmethod(cfuncs['myint_add']),
        'null': cfuncs['myintp_null'],
    }

class TestBasic:
    @fixture(scope='class')
    def myone(self):
        return MyInt1(1)

    def test_succ(self, myone):
        assert hasattr(myone, 'succ')
        assert myone.succ == 1+1

    def test_succ_call_fail(self, myone):
        with raises(TypeError):
            myone.succ()

    def test_add(self, myone):
        assert hasattr(myone, 'add')
        assert myone.add(2) == 1+2

    def test_add_more_args_fail(self, myone):
        assert raises(TypeError, myone.add, (1, 2))

    def test_add_no_args_fail(self, myone):
        assert raises(TypeError, myone.add, ())

    def test_null_my_checkerr(self, myone):
        with raises(MyError):
            myone.null()


    # Test staticmethods

    def test_s_add(self, myone):
        assert myone.s_add(1, 2) == 1+2

    def test_s_add_self_fail(self, myone):
        assert raises(TypeError, myone.s_add, (2,))

    def test_s_add_more_args_fail(self, myone):
        assert raises(TypeError, myone.s_add, (1,2,3))


# Basic MyFloat tests

class MyFloat(wrap.CObject):
    _props = {
        'succ': cfuncs['myfloat_succ'],
    }
    _meths = {
        'add': cfuncs['myfloat_add'],
        'null': cfuncs['myfloatp_null'],
    }
    def __init__(self, f):
        self.f = f
        super(MyFloat, self).__init__()
    def __float__(self):
        return self.f

class TestFloat:
    @fixture(scope='class')
    def myonef(self):
        return MyFloat(1.0)

    def test_succ(self, myonef):
        assert hasattr(myonef, 'succ')
        assert myonef.succ == 1.0+1.0

    def test_add(self, myonef):
        assert hasattr(myonef, 'add')
        assert myonef.add(2.0) == 1.0+2.0

    def test_add_more_args_fail(self, myonef):
        assert raises(TypeError, myonef.add, (1.0, 2.0))

    def test_add_no_args_fail(self, myonef):
        assert raises(TypeError, myonef.add, ())

    def test_null_checkerr(self, myonef):
        with raises(wrap.NullError):
            myonef.null()


### Inheritance tests ###

class MyInt2(MyInt1):
    _props = {
        'doubled': cfuncs['myint_doubled'],
    }
    _meths = {
        'mult': cfuncs['myint_mult'],
    }

class TestInherit:
    @fixture(scope='class')
    def mytwo(self):
        return MyInt2(2)

    def test_doubled(self, mytwo):
        assert mytwo.doubled == 2*2

    def test_mult(self, mytwo):
        assert mytwo.mult(3) == 2*3

    # Make sure we inherited the succ property from MyInt.
    def test_succ(self, mytwo):
        assert hasattr(mytwo, 'succ')
        assert mytwo.succ == 2+1

    # Again for the add method.
    def test_add(self, mytwo):
        assert hasattr(mytwo, 'add')
        assert mytwo.add(2) == 2+2


### Override inheritanc tests ###

class MyInt3(MyInt2):
    _props = {
        'succ': cfuncs['myint_succ2'],
        'doubled': cfuncs['myint_doubled'],
    }
    _meths = {
        'add': cfuncs['myint_add2'],
        'mult': cfuncs['myint_mult'],
    }

class TestOverride:
    @fixture(scope='class')
    def mythree(self):
        return MyInt3(3)

    # MyInt has a succ method. Make sure we get the one from MyInt3.
    def test_succ(self, mythree):
        assert hasattr(mythree, 'succ')
        assert mythree.succ == 3+2

    # MyInt doesn't have doubled. Make sure it's not lost in the dict merge.
    def test_doubled(self, mythree):
        assert hasattr(mythree, 'doubled')
        assert mythree.doubled == 3*2

    # Same as for succ, but with a method
    def test_add(self, mythree):
        assert hasattr(mythree, 'add')
        assert mythree.add(2) == 3+2+2

    # Same as for doubled, but with a method
    def test_mult(self, mythree):
        assert hasattr(mythree, 'mult')
        assert mythree.mult(3) == 3*3

    # Make sure the null method 'falls through'.
    def test_null(self, mythree):
        assert hasattr(mythree, 'null')
        with raises(MyError):
            mythree.null()


### Outarg tests ###

class MyOutInt(MyInt):
    _meths = {
        'setp': (cfuncs['set_ptr_succ'], [1]),
        'addp': (cfuncs['set_ptr_add'], [], [1]),
        'complicated': (cfuncs['complicated'], [1], [2, 4])
    }

class TestOutargs:
    @fixture(scope='class')
    def myoutone(self):
        return MyOutInt(1)

    def test_out_setp(self, myoutone):
        assert hasattr(myoutone, 'setp')
        #set_trace()
        assert myoutone.setp() == (42, 2)

    def test_inout_addp(self, myoutone):
        assert hasattr(myoutone, 'addp')
        assert myoutone.addp(7) == (23, 8)

    def test_complicated(self, myoutone):
        assert hasattr(myoutone, 'complicated')
        # Remember, 'self' is still passed in as the first arg, so the first
        # 'in' variable will be 1
        assert myoutone.complicated(30, 8, 3.14) == (42.0, 2.0, 31, 11.14)


### Array tests ###

class MyInt4(MyInt):
    _meths = {
        'add_array': (cfuncs['myint_add_array'], {'arrays': [1]})
    }

## C arrays

class TestCArrays:
    @fixture(scope='class')
    def myfour(self):
        return MyInt4(4)

    # A few sanity checks first
    def test_carray(self):
        a = wrap.carray([1,2])
        assert list(a) == [1,2]
        a = wrap.carray(2)
        assert len(a) == 2
        a = wrap.carray([1,2], 4)
        assert list(a) == [1,2,0,0]

    def test_carray_myint_add_array(self):
        a = wrap.carray([1,2])
        cfuncs['myint_add_array'](1, a, len(a))
        assert list(a) == [2,3]

    # Test wrap.CObject handling of arrays
    def test_add_array(self, myfour):
        (retval, retarr) = myfour.add_array([4,2], 2)
        assert retval == 0
        assert list(retarr) == [4+4,2+4]

## numpy arrays

try:
    try:
        import numpypy
    except ImportError:
        pass
    import numpy

    class TestNPArrays:
        @fixture(scope='class')
        def myfive(self):
            return MyInt4(5)

        def test_nparrayptr_myint_add_array(self):
            np_a = numpy.array([1,2], dtype=numpy.int32)
            a = wrap.nparrayptr(np_a)
            cfuncs['myint_add_array'](1, a, len(np_a))
            assert list(np_a) == [2,3]

        def test_add_array(self, myfive):
            np_a = numpy.array([8, 9], dtype=numpy.int32)
            (retval, retarr) = myfive.add_array(np_a, len(np_a))
            assert retval == 0
            assert list(np_a) == [8+5,9+5]
except ImportError:
    pass

## Struct tests

# First just test passing and receiving CFFI structs

class MyPoint(wrap.CObject):
    _props = {
        'x': (cfuncs['point_x'], cfuncs['point_setx']),
        'y': (cfuncs['point_y'], cfuncs['point_sety']),
    }
    _meths = {
        '_make': staticmethod(cfuncs['make_point']),
        '__del__': cfuncs['del_point'],
        'dist': cfuncs['point_dist'],
    }

    def __init__(self, x, y):
        super(MyPoint, self).__init__()
        self._cdata = self._make(x, y)


class TestMyPoint:
    @fixture(scope='class')
    def mypoint(self):
        return MyPoint(4, 5)
    
    @fixture(scope='class')
    def mypoint2(self):
        return MyPoint(5, 4)

    def test_mypoint_props(self, mypoint):
        assert mypoint.x == 4
        assert mypoint.y == 5

    def test_mypoint_meths(self, mypoint, mypoint2):
        from math import sqrt
        d = mypoint.dist(mypoint2)
        assert d == sqrt((mypoint2.x - mypoint.x)**2 
                         + (mypoint2.x - mypoint.x)**2)

    #def test_del(self, mypoint):
    #    del mypoint
    #    # TODO This will be hard to test because Pypy's GC has delays. Might 
    #    # just have to test in CPython and assume it works in Pypy.


# Now to test CStructType

structs = None
point_t = None
def test_CStructType_wrapall():
    global structs, point_t

    structs = wrap.CStructType.wrapall(ffi)
    assert isinstance(structs, dict)
    assert 'point_t' in structs
    assert isinstance(structs['point_t'], wrap.CStructType)
    point_t = structs['point_t']

def test_CStructType_name_create():
    point_t = wrap.CStructType(ffi, 'point_t')
    assert point_t.cname == 'point_t'
    point = point_t(x=1, y=2)
    assert point.x == 1
    assert point.y == 2

class TestMyPointStruct:
    def test_pos_create(self):
        p = point_t(32, 45)
        assert p.x == 32
        assert p.y == 45

    def test_kw_create(self):
        p = point_t(x=12, y=61)
        assert p.x == 12
        assert p.y == 61

    def test_pos_kw_overlap(self):
        with raises(TypeError):
            p = point_t(1, x=2, y=3)

    def test_too_many_args(self):
        with raises(TypeError):
            p = point_t(1, 2, 3)

    def test_array(self):
        pa = point_t.array(10)
        assert len(pa) == 10
        assert pa[9].x == 0
        with raises(IndexError):
            pa[10].x == 0

