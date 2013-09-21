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


''' A collection of convenience classes and functions for CFFI wrappers. '''


import types
import collections
from functools import wraps
import cffi

try:
    try:
        import numpypy
    except ImportError:
        pass
    import numpy
except ImportError:
    numpy = None


__all__ = ['CFunction', 'CObject', 'WrapError', 'NullError']


_empty_ffi = cffi.FFI()


class WrapError(Exception):
    pass


class NullError(WrapError):
    pass


class CFunction(object):
    def __init__(self, ffi, cfunc, checkerr=None, **kwargs):
        ''' Adds some low-ish-level introspection to CFFI C functions and
        provides a convenience function for wrapping all the functions in an
        API. This is useful for automatic or "batch" wrapping of general C
        functions. Most other wrapper classes expect API functions to be
        wrapped in a CFunction. See ``wrapall()`` below.

        * ``ffi``: The FFI object the C function is from.
        * ``cfunc``: The C function object from CFFI.
        * ``checkerr``: An optional callback passed the CFunction object, the
          args which were passed to the cfunc and the return value.
        * Any extra keyword args are passed to ``set_outargs``.

        Attributes added to instances:

        * ``cfunc``: The C function object.
        * ``ffi``: The FFI object the C function is from.
        * ``typeof``: ffi.typeof(cfunc)
        * ``cname``: From typeof.
        * ``args``: From typeof.
        * ``kind``: From typeof.
        * ``result``: From typeof.

        Callable: when called, the cfunc is called directly and it's result
        is returned.

        '''

        # This is basically a hack to work around the lack of introspection
        # built-in to CFFI CData objects. The overhead should be negligable
        # since the CFFI function is directly assigned to __call__ (this also
        # prevents it from being called like a bound method - we do that later
        # in when bind it to the CObject instance below).

        self.cfunc = cfunc
        self.ffi = ffi
        self._checkerr = checkerr

        self.typeof = ffi.typeof(cfunc)
        self.args = self.typeof.args
        self.cname = self.typeof.cname
        self.kind = self.typeof.kind
        self.result = self.typeof.result

        # TODO Profile to see if this is really much faster...
        #self.__call__ = func

        self.set_outargs(**kwargs)

    def __get__(self, obj, objtype=None):
        if hasattr(obj, '_checkerr'):
            return lambda *args: self(*args, checkerr=obj._checkerr)
        else:
            return self

    def __call__(self, *args, **kwargs):
        # Most of this code has been heavily profiled with several different
        # approaches and algorithms. However, if you think of a faster/better
        # way to do this, I'm open to ideas. This code should be fairly fast
        # because it will be the primary interface to the underlying C library,
        # potentially having wrapper functions called in tight loops.

        # pypy: 1000000 loops, best of 3: 229 ns per loop

        # Actually, looking at the profiler output, by far the biggest cost is
        # in CFFI itself (specifically calls to the _optimize_charset function
        # in the compile_sre.py module) so I don't think it's worth it to
        # squeeze much more performance out of this code...

        # TODO IDEA: Consider using some kind of format string(s) to specify
        # outargs, arrays, retargs, etc? This is getting complicated enough
        # that it might make things simpler for the user?
        # Maybe something like "iioxiai" where 'i' is for "in" arg, 'o' for out
        # 'x' for in/out. Could then maybe do computed args, like array lengths
        # with something like "iiox{l5}iai" where "{l5}i" means the length of
        # the 6th (0-indexed) argument. Just something to think about...

        # TODO: Also, maybe this should support some way to change the position
        # of the 'self' argument to allow for libraries which have inconsistent
        # function signatures...

        if len(args) != self.numargs:
            raise TypeError('wrapped Function {0} requires exactly {1} '
                            'arguments ({2} given)'.format(self.cname,
                                                           self.numargs,
                                                           len(args)))

        for argi, arg in enumerate(args):
            if hasattr(arg, '_cdata') and arg._cdata is not None:
                args = args[:argi] + (arg._cdata,) + args[argi+1:]

        # If this function has out or in-out pointer args, create the pointers
        # for each, and insert/replace them in the argument list before passing
        # to the underlying C function.
        retargs = False
        if self.outargs:
            retargs = []

            # A few optimizations because looking up local variables is much
            # faster than looking up object attributes.
            retargs_append = retargs.append
            cargs = self.args
            cfunc = self.cfunc
            ffi = self.ffi

            for argi, inout in self.outargs:
                argtype = cargs[argi]
                if inout == 'o':
                    inptr = ffi.new(argtype.cname)
                    args = args[:argi] + (inptr,) + args[argi:]
                elif inout == 'x':
                    inptr = ffi.new(argtype.cname, args[argi])
                    args = args[:argi] + (inptr,) + args[argi+1:]
                elif inout == 'a':
                    inptr = self.get_arrayptr(args[argi], ctype=argtype)
                    args = args[:argi] + (inptr,) + args[argi+1:]
                retargs_append((inptr, inout))

        retval = self.cfunc(*args)
        checkerr = kwargs.get('checkerr')
        if checkerr is not None:
            check = checkerr(self, args, retval)
        else:
            check = self.checkerr(self, args, retval)
        retval = check or retval

        # TODO: use self.outargs to determine which args should be in the
        # output and use -1 to indicate the actual return code. Also test
        # if len(retval) == 1 and return retval_t[0].
        if retargs:
            retval = (retval,)  # Return tuples, because it's prettier :)
            for retarg, inout in retargs:
                if inout == 'a':
                    retval += (retarg,)  # Return arrays as-is
                else:
                    # TODO: In some cases we don't want them unboxed... need a
                    # good way to know when not to...
                    retval += (retarg[0],)  # Unbox other pointers

        return retval

    def set_outargs(self, checkerr=None, outargs=(), inoutargs=(), arrays=(),
                    retargs=None):
        ''' Specify which parameter positions are intended to return values.

        This feature helps to simplify dealing with pointer parameters which
        are meant to be "return" parameters. If any of these are specified, the
        return value from the wrapper function will be a tuple containing the
        actual return value from the C function followed by the values of the
        pointers which were passed in. Each list should be a list of parameter
        position numbers (0 for the first parameter, etc).. See CObject for
        examples and short-hand for setting up ``Function``\ s.

        * ``outargs``: These will be omitted from the wrapper function
          parameter list, and fresh pointers will be allocated (with types
          derived from the C function signature) and inserted in to the
          arguments list to be passed in to the C function. The pointers will
          then be dereferenced and the value included in the return tuple.

        * ``inoutargs``: Arguments passed to the wrapper function for these
          parameters will be cast to pointers before being passed in to the C
          function. Pointers will be unboxed in the return tuple.

        * ``arrays``: Arguments to these parameters can be python lists or
          tuples, numpy arrays or integers.

          Python lists/tuples will be copied in to newly allocated CFFI
          arrays and the pointer passed in. The generated CFFI array will be
          in the return tuple.

          Numpy arrays will have their data buffer pointer cast to a CFFI
          pointer and passed in directly (no copying is done). The CFFI
          pointer to the raw buffer will be returned, but any updates to the
          array data will also be reflected in the original numpy array, so
          it's recommended to just keep using that. (TODO: This behavior may
          change to remove these CFFI pointers from the return tuple or maybe
          replace the C array with the original numpy object.)

          Integers will indicate that a fresh CFFI array should be allocated
          with a length equal to the int an initialized to zeros. The generated
          CFFI array will be included in the return tuple.

        For example, a C function with this signature::

            int cfunc(int inarg, int *outarg, float *inoutarg);

        with ``outargs`` set to ``[1]`` and ``inoutargs`` set to ``[2]`` can be
        called from python as::

            >>> ret, ret_outarg, ret_inoutarg = wrapped_cfunc(inarg, inoutarg)

        Returned values will be unboxed python values.

        ``set_outargs`` returns self.

        '''

        self.numargs = len(self.args) - len(outargs)

        outargs =  [(i, 'o') for i in outargs]
        outargs += ((i, 'x') for i in inoutargs)
        outargs += ((i, 'a') for i in arrays)

        self.outargs = sorted(outargs)
        self.retargs = retargs
        self._checkerr = checkerr

        return self

    def get_arrayptr(self, array, ctype=None):
        ''' Get a CFFI compatible pointer object for an array.

        Supported ``array`` types are:

        * numpy ndarrays: The pointer to the underlying array buffer is cast
          to a CFFI pointer. Value returned from __call__ will be a pointer,
          but the numpy C buffer is updated in place, so continue to use the
          numpy ndarray object.
        * CFFI CData pointers: If the user is already working with C arrays
          (i.e., ffi.new(``int[10]``)) these will be returned as given.
        * Python ints and longs: These will be interpretted as the length of a
          newly allocated C array. The pointer to this array will be
          returned. ``ctype`` must be provided (CFunction's __call__ method
          does this automatically).
        * Python collections: A new C array will be allocated with a length
          equal to the length of the iterable (``len()`` is called and the
          iterable is iterated over, so don't use exhaustable generators, etc).
          ``ctype`` must be provided (Function's __call__ method does this
          automatically).

        '''

        if numpy and isinstance(array, numpy.ndarray):
            return self.ffi.cast('void *',
                                 array.__array_interface__['data'][0])
        elif isinstance(array, self.ffi.CData):
            return array
        else:
            # Assume it's an iterable or int/long. CFFI will handle the rest.
            return self.ffi.new(self.ffi.getctype(ctype.item.cname, '[]'),
                                array)

    @staticmethod
    def checkerr(cfunc, args, retval):
        ''' Default error checker. Checks for NULL return values and raises
        NullError.

        Can be overridden by subclasses. If ``_checkerr`` returns anything
        other than ``None``, that value will be returned by the property or
        method, otherwise original return value of the C call will be returned.
        Also useful for massaging returned values.

        '''

        #TODO: Maybe should generalize to "_returnhandler" or something?

        #if self._checkerr is not None:
        #    self._checkerr(cfunc, args, retval)

        if retval == cffi.FFI.NULL:
            raise NullError('NULL returned by {0} with args {1}. '
                            .format(cfunc.cname, args))
        else:
            return retval

class CMethod(object):
    def __init__(self, cfunc):
        self.cfunc = cfunc
        

class CStructType(object):
    ''' Provides introspection to CFFI ``StructType``s and ``UnionType``s. '''
    def __init__(self, ffi, structtype):
        ''' Create a new CStructType.

        * ``ffi``: The FFI object.
        * ``structtype``: a CFFI StructType or a string for the type name
          (wihtout any trailing '*' or '[]').

        Instances have the following attributes:

        * ``ffi``: The FFI object this struct is pulled from.
        * ``cname``: The C name of the struct.
        * ``ptrname``: The C pointer type signature for this struct.
        * ``fldnames``: A list of fields this struct has.

        '''

        if isinstance(structtype, str):
            structtype = ffi._parser.parse_type(structtype)

        self._struct_type = structtype
        self.ffi = ffi

        # Sometimes structtype.name starts with a '$'...?
        self.cname = structtype.get_c_name()
        self.ptrname = ffi.getctype(self.cname, '*')
        self.fldnames = structtype.fldnames

    def __call__(self, *args, **kwargs):
        ''' Returns a pointer to a new CFFI struct instance for the given type.

        Struct fields can be passed in as positional arguments or keyword
        arguments. ``TypeError`` is raised if positional arguments overlap with
        given keyword arguments.

        '''

        if self.fldnames is None:
            if args or kwargs:
                raise TypeError('CStructType call with arguments on opaque '
                                'CFFI struct {0}.'.format(self.cname))
            return self.ffi.new(self.ptrname)
        else:
            if len(args) > len(self.fldnames):
                raise TypeError('CStructType got more arguments than struct '
                                'has fields. {0} > {1}'
                                .format(len(args), len(self.fldnames)))
            retval = self.ffi.new(self.ptrname)
            for fld, val in zip(self.fldnames, args):
                if fld in kwargs:
                    raise TypeError('CStructType call got multiple values for '
                                    'field name {0}'.format(fld))
                setattr(retval, fld, val)
            for fld, val in kwargs.iteritems():
                setattr(retval, fld, val)

            return retval

    def array(self, shape):
        ''' Constructs a C array of the struct type with the given length.

        * ``shape``: Either an int for the length of a 1-D array, or a tuple
          for the length of each of len dimensions. I.e., [2,2] for a 2-D array
          with length 2 in each dimension. Hint: If you want an array of
          pointers just add an extra demension with length 1. I.e., [2,2,1] is
          a 2x2 array of pointers to structs.

        No initialization of the elements is performed. CFFI initializes newly
        allocated memory to zeros.

        '''

        if isinstance(shape, collections.Iterable):
            suffix = '[%i]' * len(shape) % tuple(shape)
        else:
            suffix = '[%i]' % (shape,)

        # TODO Allow passing initialization args? Maybe factor out some of the
        # code in __call__?
        return self.ffi.new(self.ffi.getctype(self.cname + suffix))


class _MetaWrap(type):
    ''' See ``CObject``. '''
    def __new__(mcs, name, bases, attrs):
        # Allow sub-subclasses to inherit _meth entries from their parents.
        if '_meths' in attrs:
            for base in bases:
                if hasattr(base, '_meths'):
                    # Inherits keys from parent class without overridding
                    # existing keys in subclass, or mucking up the parent
                    # class's _meths dict. Simplest way I've found to do it.
                    meths = base._meths.copy()
                    meths.update(attrs['_meths'])
                    attrs['_meths'] = meths

        return super(_MetaWrap, mcs).__new__(mcs, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        for prop, funcs in attrs.get('_props', {}).iteritems():
            # TODO: Support outargs in props for getter functions that take a
            # second argument like a struct or string pointer which is filled
            # out with the requested data.
            if isinstance(funcs, (tuple, list)):
                #attrs[prop] = property(*funcs)
                setattr(cls, prop, property(*funcs))
            else:
                #attrs[prop] = property(funcs)
                setattr(cls, prop, property(funcs))

        for meth, cfunc in attrs.get('_meths', {}).iteritems():

            # Support for (cfunc, [outargs...], [inoutargs...]) form
            kwargs = {}
            if isinstance(cfunc, collections.Sequence):

                # (cfunc, {outargs=[1,2,3], ...}) form
                if isinstance(cfunc[-1], dict):
                    kwargs = cfunc[-1]
                    cfunc = cfunc[:-1]

                # (cfunc, [1,2,3], ...) outargs form
                if len(cfunc) >= 2:
                    kwargs['outargs'] = cfunc[1]

                # (cfunc, [1,2,3], [4,5,6]) outargs + inoutargs form
                if len(cfunc) >= 3:
                    kwargs['inoutargs'] = cfunc[2]

                # (cfunc, [1,2,3], [4,5,6], [7,8,9]) ... + arrays form
                if len(cfunc) == 4:
                    kwargs['arrays'] = cfunc[3]

                # Unsupported form
                if len(cfunc) > 4 or len(cfunc) < 1:
                    raise WrapError('Wrong number of items in method ' +
                                    'spec tuple. Must contain 1 to 4 ' +
                                    'items. Got: {0}'.format(repr(cfunc)))

                cfunc = cfunc[0]

            # C functions that don't accept the wrapped object
            if isinstance(cfunc, staticmethod):
                cfunc = cfunc.__func__
                cfunc.set_outargs(**kwargs)

            # Usually C functions that create object. Default as static.
            elif meth == '_cnew':
                cfunc.set_outargs(**kwargs)

            # Other methods treated like instance methods
            else:
                cfunc.set_outargs(**kwargs)
                cfunc = types.MethodType(cfunc, None, cls)

            #attrs[meth] = cfunc
            setattr(cls, meth, cfunc)


class CObject(object):
    ''' A pythonic representation of a C "object", usually representing a set
    of C functions that operate over a common peice of data. Many C APIs have
    lots of functions which accept some common struct pointer or identifier as
    the first argument being manipulated. CObject provides a convenient
    abstrtaction to making this convention more "object oriented". See the
    example below. More examples can be found in the unit tests.

    Intended for subclassing. Subclass should have a ``_props`` and/or
    ``_meths`` attribute.

    The ``_props`` class attribute is a dict mapping ``propery`` names to
    Function functions. Keys in this dict are accessed from the Python object
    as attributes and return the result of calling the Function object with
    self as the only argument. For example:

    libexample.h::

        typedef int point_t;
        point_t make_point(int x, int y);
        int point_x(point_t p);
        int point_y(point_t p);
        int point_setx(point_t p, int x);
        int point_sety(point_t p, int y);
        int point_move(point_t p, int x, int y);

        int point_x_abs(point_t p);
        int point_movex(point_t p, int x);

    Python usage (where libexample is an API object from ``ffi.verify()``)::

        >>> import cffiwrap as wrap
        >>> class Point(wrap.CObject):
        ...     _props = {
        ...         'x': libexample.point_x,
        ...         'y': libexample.point_y
        ...     }
        ...     def __init__(self, x, y):
        ...         self.id = ffi.make_point(x, y)
        ...         super(Point, self).__init__()
        ...     def __int__(self):
        ...         """
        ...         Called automatically by CFFI when being passed as an int
        ...         argument to a C function.
        ...         """
        ...         return self.id
        ... 
        >>> p = Point(4, 2)
        >>> p.x
        4
        >>> p.y
        2

    The wrapped C functions are called each time the property is retrieved from
    the Point instance.

    Properties can also have "setter" functions. Simply replace the values in
    the ``_props`` dict with a 2-tuple with the "getter" and "setter"
    functions::

        >>> class Point2(Point):
        ...     _props = {
        ...         'x': (libexample.point_x, libexample.point_setx),
        ...         'y': (libexample.point_y, libexample.point_sety)
        ...     }
        ... 
        >>> p2 = Point2(7, 4)
        >>> p2.x
        7
        >>> p2.x = 8
        >>> p2.x
        8

    Subclasses can also define a ``_meths`` dict for more general methods. If a
    method named "``_cnew``" is defined, this will be called by ``__init__``
    and the return value assigned to the ``_cdata`` instance attribute::

        >>> class Point3(Point):
        ...     _meths = {'move': libexample.point_move}
        ... 
        >>> p3 = Point3(8, 2)
        >>> p3.move(2, 2)
        0
        >>> p3.x
        10
        >>> p3.y
        4

    Sub-subclasses of CObject will also inherit individual methods and
    properties from their parents:

        >>> class Point4(Point3):
        ...     _meths = {'x_abs': libexample.point_x_abs}
        ...     _props = {'movex': libexample.point_movex}
        ... 
        >>> p4 = Point4(-5, 10)
        >>> p4.x_abs
        5
        >>> p4.x
        -5
        >>> p4.movex(3)
        0
        >>> (p4.x, p4.y)
        (-2, 10)
        >>> p4.move(7, -20)
        0
        >>> (p4.x, p4.y)
        (5, -10)

    The values in the _meths dict can also be a tuple with the C function as
    the first element, the second and third being the "outargs" and "inoutargs"
    to the Function constructor.

    TODO: Come up with not-too-ridiculously-contrived example. For now, check
    out the unit tests for some examples.

    TODO: Document arrays and passing kwargs to Function.

    Optionally, for C types which are not automatically coerced/converted by
    CFFI (such as struct pointers) the subclass can set a class- or instance-
    attribute named ``_cdata`` which will be passed to the CFFI functions
    instead of ``self``. For example:

    libexample cdef::

        typedef struct { int x; ...; } mystruct;
        mystruct* make_mystruct(int x);
        int mystruct_x(mystruct* ms);

    python::

        >>> import cffiwrap as wrap
        >>> class MyStruct(wrap.CObject):
        ...     _props = {'x': libexample.mystruct_x}
        ...     _meths = {'_make': libexample.make_mystruct}
        ...     def __init__(self, x):
        ...         self._cdata = self._make(x)
        ... 
        >>> ms = MyStruct(4)
        >>> ms.x
        4

    Note: stack-passed structs are not supported yet* but pointers to
    structs work as expected if you set the ``_cdata`` attribute to the
    pointer.

    * https://bitbucket.org/cffi/cffi/issue/102

    An ``CObject`` can also specify a method named ``_cnew`` which will be
    called when the class is instantiated. This can be declared in the
    ``_meths`` dict or be a normal method on the class. Any arguments given
    when the class is called will be passed to this method. The return value
    will be automatically assigned to the ``_cdata`` instance attribute.

    If _cdata is set, attributes of the cdata object can also be retrieved from
    the CObject instance, e.g., for struct fields, etc.

    You can also specify a destructor with a ``_del`` method in the same way as
    ``_cnew``.

    '''

    __metaclass__ = _MetaWrap
    _cdata = None

    def __init__(self, *args):
        if hasattr(self, '_cnew') and self._cnew is not None:
            self._cdata = self._cnew(*args)

    def __getattr__(self, attr):
        if self._cdata is not None and hasattr(self._cdata, attr):
            return getattr(self._cdata, attr)

    def __del__(self):
        if hasattr(self, '_del'):
            self._del()


def wrapall(ffi, api):
    ''' Convenience function to wrap CFFI functions structs and unions.

    Reads functions, structs and unions from an API/Verifier object and wrap
    them with the respective wrapper functions.

    ``ffi``: The FFI object (needed for it's ``typeof()`` method)
    ``api``: As returned by ``ffi.verify()``

    Returns a dict mapping object names to wrapper instances. Hint: in
    a python module that only does CFFI boilerplate, try something like::

        globals().update(wrapall(myffi, myapi))

    '''

    # TODO: Support passing in a checkerr function to be called on the
    # return value for all wrapped functions.
    cobjs = {}
    for attr in dir(api):
        if not attr.startswith('_'):
            cobj = getattr(api, attr)
            if (isinstance(cobj, collections.Callable)
                    and ffi.typeof(cobj).kind == 'function'):
                cobj = CFunction(ffi, cobj)
            cobjs[attr] = cobj

        # The things I go through for a little bit of introspection.
        # Just hope this doesn't change too much in CFFI's internals...

    decls = ffi._parser._declarations
    for _, ctype in decls.iteritems():
        if isinstance(ctype, (cffi.model.StructType,
                              cffi.model.UnionType)):
            cobjs[ctype.get_c_name()] = CStructType(ffi, ctype)

    return cobjs


def nparrayptr(nparr):
    ''' Convenience function for getting the CFFI-compatible pointer to a numpy
    array object. '''

    return _empty_ffi.cast('void *', nparr.__array_interface__['data'][0])


def carray(items_or_size=None, size=None, ctype='int'):
    ''' Convenience function for creating C arrays. '''

    # TODO: Support multi-dimensional arrays? Maybe it's just easier to stick
    # with numpy...

    if isinstance(items_or_size, (int, long)) and size is None:
        size = items_or_size
        items = None
    else:
        items = items_or_size

    if items and size > len(items):
        size = max(len(items), size or 0)
        arr = _empty_ffi.new(_empty_ffi.getctype(ctype, '[]'), size)
        for i, elem in enumerate(items):
            arr[i] = elem
        return arr
    else:
        return _empty_ffi.new(_empty_ffi.getctype(ctype, '[]'), items or size)

# This is starting to get kind of long... not sure when I should break it out
# in to a package... I kind of like keeping it a simple module, but, I don't
# know...
