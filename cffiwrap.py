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


__all__ = [
    'CFunction',
    'CObject',
    'NullError',
    'cmethod',
    'cstaticmethod',
    'cproperty',
]


_empty_ffi = cffi.FFI()


class NullError(Exception):
    pass


class CFunction(object):
    def __init__(self, ffi, cfunc):
        ''' Adds some low-ish-level introspection to CFFI C functions and
        provides a convenience function for wrapping all the functions in an
        API. This is useful for automatic or "batch" wrapping of general C
        functions. Most other wrapper classes expect API functions to be
        wrapped in a CFunction. See ``wrapall()`` below.

        * ``ffi``: The FFI object the C function is from.
        * ``cfunc``: The C function object from CFFI.
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
        # built-in to CFFI CData function objects. The overhead should be
        # negligable since the CFFI function is directly assigned to __call__
        # (this also prevents it from being called like a bound method - we do
        # that later with the cmethod module function).

        self.cfunc = cfunc
        self.ffi = ffi

        self.typeof = ffi.typeof(cfunc)
        self.args = self.typeof.args
        self.cname = self.typeof.cname
        self.kind = self.typeof.kind
        self.result = self.typeof.result

        # TODO Profile to see if this is really much faster...
        #self.__call__ = func

    def __call__(self, *args, **kwargs):
                 #outargs=() retargs=None):
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
        # Update: This seems to no longer be the case in newer pypy/cffi?

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

        outargs = kwargs.get('outargs')
        retargs = kwargs.get('retargs')

        # This guard is semantically useless, but is substantially faster in
        # cpython than trying to iterate over enumerate([]). (No diff in pypy)
        if args:
            for argi, arg in enumerate(args):
                if hasattr(arg, '_cdata') and arg._cdata is not None:
                    args = args[:argi] + (arg._cdata,) + args[argi+1:]

        # If this function has out or in-out pointer args, create the pointers
        # for each, and insert/replace them in the argument list before passing
        # to the underlying C function.
        retvals = False
        if outargs:
            # TODO: use retargs to determine which args should be in the
            # return and use -1 to indicate the actual return code. Also test
            # if len(retval) == 1 and return retval_t[0].
            retvals = []

            # A few optimizations because looking up local variables is much
            # faster than looking up object attributes.
            retvals_append = retvals.append
            cargs = self.args
            cfunc = self.cfunc
            ffi = self.ffi

            for argi, inout in outargs:
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
                retvals_append((inptr, inout))

        retval = self.cfunc(*args)
        # This is a tad slower in pypy but substantially faster in cpython than
        # checkerr = kwargs.get('checkerr'); if checkerr is not None: ...
        if 'checkerr' in kwargs and kwargs['checkerr'] is not None:
            retval = kwargs['checkerr'](self, args, retval)
        else:
            retval = self.checkerr(self, args, retval)

        if retvals:
            retval = (retval,)  # Return tuples, because it's prettier :)
            for retarg, inout in retvals:
                if inout == 'a':
                    retval += (retarg,)  # Return arrays as-is
                else:
                    # TODO: In some cases we don't want them unboxed... need a
                    # good way to specify when not to...
                    retval += (retarg[0],)  # Unbox other pointers

        return retval

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
          ``ctype`` must be provided (CFunction's __call__ method does this
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


def cmethod(cfunc=None, outargs=(), inoutargs=(), arrays=(), retargs=None,
           checkerr=None):
    ''' Wrap cfunc to simplify handling outargs, etc.

    This feature helps to simplify dealing with pointer parameters which
    are meant to be "return" parameters. If any of these are specified, the
    return value from the wrapper function will be a tuple containing the
    actual return value from the C function followed by the values of the
    pointers which were passed in. Each list should be a list of parameter
    position numbers (0 for the first parameter, etc)..

    * ``outargs``: These will be omitted from the cmethod-wrapped function
      parameter list, and fresh pointers will be allocated (with types
      derived from the C function signature) and inserted in to the
      arguments list to be passed in to the C function. The pointers will
      then be dereferenced and the value included in the return tuple.

    * ``inoutargs``: Arguments passed to the wrapper function for these
      parameters will be cast to pointers before being passed in to the C
      function. Pointers will be unboxed in the return tuple.

    * ``arrays``: Arguments to these parameters can be python lists or
      tuples, numpy arrays or integers.

      * Python lists/tuples will be copied in to newly allocated CFFI
        arrays and the pointer passed in. The generated CFFI array will be
        in the return tuple.

      * Numpy arrays will have their data buffer pointer cast to a CFFI
        pointer and passed in directly (no copying is done). The CFFI
        pointer to the raw buffer will be returned, but any updates to the
        array data will also be reflected in the original numpy array, so
        it's recommended to just keep using that. (TODO: This behavior may
        change to remove these CFFI pointers from the return tuple or maybe
        replace the C array with the original numpy object.)

      * Integers will indicate that a fresh CFFI array should be allocated
        with a length equal to the int an initialized to zeros. The generated
        CFFI array will be included in the return tuple.

    * ``retargs``: (Not implemented yet.) A list of values to be returned from
      the cmethod-wrapped function. Normally the returned value will be a tuple
      containing the actual return value of the C function, followed by the
      final value of each of the ``outargs``, ``inoutargs``, and ``arrays`` in
      the order they appear in the C function's paramater list.

    As an example of using ``outargs`` and ``inoutargs``, a C function with
    this signature::

        int cfunc(int inarg, int *outarg, float *inoutarg);

    with an ``outargs`` of ``[1]`` and ``inoutargs`` set to ``[2]`` can be
    called from python as::

        >>> wrapped_cfunc = cmethod(cfunc, outargs=[1], inoutargs=[2])
        >>> ret, ret_outarg, ret_inoutarg = wrapped_cfunc(inarg, inoutarg)

    Returned values will be unboxed python values unless otherwise documented
    (i.e., arrays).

    '''

    # TODO: retargs...

    if cfunc is None:
        # TODO: There's probably something interesting to do in this case...
        # maybe work like a decorator if cfunc isn't given?
        return None

    if not isinstance(cfunc, CFunction):
        # Can't do argument introspection... TODO: raise an exception?
        return cfunc

    numargs = len(cfunc.args) - len(outargs)

    outargs =  [(i, 'o') for i in outargs]
    outargs += ((i, 'x') for i in inoutargs)
    outargs += ((i, 'a') for i in arrays)

    outargs.sort()
    
    @wraps(cfunc.cfunc)
    def wrapper(*args):
        if len(args) != numargs:
            raise TypeError('wrapped Function {0} requires exactly {1} '
                            'arguments ({2} given)'
                            .format(cfunc.cname, numargs, len(args)))

        if checkerr is None and hasattr(args[0], '_checkerr'):
            _checkerr = args[0]._checkerr
        else:
            _checkerr = checkerr
        return cfunc(*args, outargs=outargs, retargs=retargs,
                     checkerr=_checkerr)

    return wrapper


def cstaticmethod(cfunc, **kwargs):
    ''' Shortcut for staticmethod(cmethod(cfunc, [kwargs ...])) '''
    return staticmethod(cmethod(cfunc, **kwargs))


def cproperty(fget=None, fset=None, fdel=None, doc=None, checkerr=None):
    ''' Shortcut to create ``cmethod`` wrapped ``property``\ s. '''
    return property(fget=cmethod(fget, checkerr=checkerr),
                    fset=cmethod(fset, checkerr=checkerr),
                    fdel=cmethod(fdel, checkerr=checkerr),
                    doc=doc)


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

        Instances of this class are essentially struct/union generators.
        Calling an instance of ``CStructType`` will produce a newly allocated 
        struct or union. See the ``__call__`` and ``array`` doc strings for
        more details.

        The module convenience function ``wrapall`` creates ``CStructType``\ s
        for each struct and union imported from the FFI.

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

        # TODO: Factor out and integrate with carray function below?
        if isinstance(shape, collections.Iterable):
            suffix = '[%i]' * len(shape) % tuple(shape)
        else:
            suffix = '[%i]' % (shape,)

        # TODO Allow passing initialization args? Maybe factor out some of the
        # code in __call__?
        return self.ffi.new(self.ffi.getctype(self.cname + suffix))


class CObject(object):
    ''' A pythonic representation of a C "object"
    
    Usually representing a set of C functions that operate over a common peice
    of data. Many C APIs have lots of functions which accept some common struct
    pointer or identifier int as the first argument being manipulated. CObject
    provides a convenient abstrtaction to making this convention more "object
    oriented". See the example below. More examples can be found in the
    cffiwrap unit tests.

    Use ``cproperty`` and ``cmethod`` to wrap CFFI C functions to behave like
    instance methods, passing the instance in as the first argument. See the
    doc strings for each above.

    For C types which are not automatically coerced/converted by CFFI (such as
    C functions accepting struct pointers, etc) the subclass can set a class-
    or instance-attribute named ``_cdata`` which will be passed to the CFFI
    functions instead of ``self``. The CObject can also have a ``_cnew`` static
    method (see ``cstaticmethod``) which will be called by the base class's
    ``__init__`` and the returned value assigned to the instances ``_cdata``.

    For example:

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
        ...     x = cproperty(libexample.point_x, libexample.point_setx)
        ...     y = cproperty(libexample.point_y, libexample.point_sety)
        ...     _cnew = cstaticmethod(libexample.make_point)
        ... 
        >>> p = Point(4, 2)
        >>> p.x
        4
        >>> p.x = 8
        >>> p.x
        8
        >>> p.y
        2

    You can also specify a destructor with a ``_cdel`` method in the same way
    as ``_cnew``.

    Alternatively you can assign a CFFI compatible object (either an actual
    CFFI CData object, or something CFFI automatically converts like and int)
    to the instance's _cdata attribute.

    ``cmethod`` wraps a CFunction to provide an easy way to handle 'output'
    pointer arguments, arrays, etc. (See the ``cmethod`` documentation.)::

        >>> class Point2(Point):
        ...     move = cmethod(libexample.point_move)
        ... 
        >>> p2 = Point2(8, 2)
        >>> p2.move(2, 2)
        0
        >>> p2.x
        10
        >>> p2.y
        4

    If _cdata is set, attributes of the cdata object can also be retrieved from
    the CObject instance, e.g., for struct fields, etc.

    libexample cdef::

        typedef struct { int x; int y; ...; } mystruct;
        mystruct* make_mystruct(int x, int y);
        int mystruct_x(mystruct* ms);

    python::

        >>> class MyStruct(wrap.CObject):
        ...     x = cproperty(libexample.mystruct_x)
        ...     _cnew = cstaticmethod(libexample.make_mystruct)
        ... 
        >>> ms = MyStruct(4, 2)
        >>> ms.x  # Call to mystruct_x via cproperty
        4
        >>> ms.y  # direct struct field access
        2

    Note: stack-passed structs are not supported yet* but pointers to
    structs work as expected if you set the ``_cdata`` attribute to the
    pointer.

    * https://bitbucket.org/cffi/cffi/issue/102


    '''

    _cdata = None

    def __init__(self, *args):
        if hasattr(self, '_cnew'):
            self._cdata = self._cnew(*args)

    def __getattr__(self, attr):
        if self._cdata is not None and hasattr(self._cdata, attr):
            return getattr(self._cdata, attr)
        else:
            raise AttributeError("{0} object has no attribute {1}"
                                 .format(repr(self.__class__), repr(attr)))

    def __del__(self):
        if hasattr(self, '_cdel'):
            self._cdel()


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
