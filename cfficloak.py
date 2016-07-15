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


''' A collection of convenience classes and functions for CFFI wrappers. '''


import collections
import six
import types
from functools import wraps
try:
    import cffi
except ImportError:
    cffi = None

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
    'CStruct',
    'CUnion',
    'CStructType',
    'CUnionType',
    'CObject',
    'NullError',
    'cmethod',
    'cstaticmethod',
    'cproperty',
    'wrap',
    'wrapall',
    'wrapenum',
    'carray',
    'nparrayptr',
]


if cffi:
    _global_ffi = cffi.FFI()
else:
    _global_ffi = None


class NullError(Exception):
    pass


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class CFunction(object):
    ''' Adds some low-ish-level introspection to CFFI C functions.

    Most other wrapper classes and fuctions expect API functions
    to be wrapped in a CFunction. See ``wrapall()`` below.

    * ``ffi``: The FFI object the C function is from.
    * ``cfunc``: The C function object from CFFI.

    Attributes added to instances:

    * ``cfunc``: The C function object.
    * ``ffi``: The FFI object the C function is from.
    * ``typeof``: ffi.typeof(cfunc)
    * ``cname``: From typeof.
    * ``args``: From typeof.
    * ``kind``: From typeof.
    * ``result``: From typeof.

    Callable: when called, the cfunc is called directly and it's result
    is returned. See ``cmethod`` for more uses.

    '''

    def __init__(self, ffi, cfunc):
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
                elif arg is None:
                    args = args[:argi] + (self.ffi.NULL,) + args[argi + 1:]
                elif isinstance(arg, six.text_type):
                    if 'wchar' in self.args[argi].cname:
                        arg = self.ffi.new('wchar[]', arg)
                    elif 'char' in self.args[argi].cname:
                        arg = self.ffi.new('char[]', arg.encode())
                    args = args[:argi] + (arg,) + args[argi + 1:]
                elif isinstance(arg, six.binary_type):
                    if 'wchar' in self.args[argi].cname:
                        arg = self.ffi.new('wchar[]', arg.decode())
                    elif 'char' in self.args[argi].cname:
                        arg = self.ffi.new('char[]', arg)
                    args = args[:argi] + (arg,) + args[argi + 1:]

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
                    if ffi.typeof(args[argi]) == argtype:
                        inptr = args[argi]
                    else:
                        inptr = ffi.new(argtype.cname, args[argi])
                    args = args[:argi] + (inptr,) + args[argi+1:]
                elif inout == 'a':
                    inptr = self.get_arrayptr(args[argi], ctype=argtype)
                    args = args[:argi] + (inptr,) + args[argi+1:]
                retvals_append((inptr, inout))

        retval = self.cfunc(*args)

        if self.result.kind == 'enum':
            retval = wrapenum(retval, self.result)

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
          (i.e., ``ffi.new("int[10]"))`` these will be returned as given.
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

    def checkerr(self, cfunc, args, retval):
        ''' Default error checker. Checks for NULL return values and raises
        NullError.

        Can be overridden by subclasses. If ``_checkerr`` returns anything
        other than ``None``, that value will be returned by the property or
        method, otherwise original return value of the C call will be returned.
        Also useful for massaging returned values.

        '''

        #TODO: Maybe should generalize to "returnhandler" or something?

        #if self._checkerr is not None:
        #    self._checkerr(cfunc, args, retval)

        if retval == self.ffi.NULL:
            raise NullError('NULL returned by {0} with args {1}. '
                            .format(cfunc.cname, args))
        else:
            return retval


def wrap(ffi, cobj):
    '''
    Convenience function to wrap CFFI functions structs and unions.
    '''
    if (isinstance(cobj, collections.Callable)
        and ffi.typeof(cobj).kind == 'function'):
        cobj = CFunction(ffi, cobj)

    elif isinstance(cobj, ffi.CData):
        kind = ffi.typeof(cobj).kind
        if kind == 'pointer':
            kind = ffi.typeof(cobj).item.kind

        if kind == 'struct':
            cobj = CStruct(ffi, cobj)
        elif kind == 'union':
            cobj = CUnion(ffi, cobj)

    elif isinstance(cobj, int):
        pass
    else:
        print("Unknown: %s" % cobj)
    return cobj


def wrapall(ffi, api):
    '''
    Convenience function to wrap CFFI functions structs and unions.

    Reads functions, structs and unions from an API/Verifier object and wrap
    them with the respective wrapper functions.

    * ``ffi``: The FFI object (needed for it's ``typeof()`` method)
    * ``api``: As returned by ``ffi.verify()``

    Returns a dict mapping object names to wrapper instances. Hint: in
    a python module that only does CFFI boilerplate and verification, etc, try
    something like this to make the C values available directly from the module
    itself::

        globals().update(wrapall(myffi, myapi))

    '''

    # TODO: Support passing in a checkerr function to be called on the
    # return value for all wrapped functions.

    global _global_ffi
    if _global_ffi is None:
        _global_ffi = ffi

    cobjs = dotdict()
    for attr in dir(api):
        if not attr.startswith('_'):
            cobj = getattr(api, attr)
            cobj = wrap(ffi, cobj)
            cobjs[attr] = cobj

        # The things I go through for a little bit of introspection.
        # Just hope this doesn't change too much in CFFI's internals...

    try:
        typedef_names, names_of_structs, names_of_unions = ffi.list_types()
        for ctypename in names_of_structs:
            try:
                cobjs[ctypename] = CStructType(ffi, ctypename)
            except ffi.error as ex:
                pass
        for ctypename in names_of_unions:
            try:
                cobjs[ctypename] = CUnionType(ffi, ctypename)
            except ffi.error as ex:
                pass
        for ctypename in typedef_names:
            try:
                cobjs[ctypename] = CType(ffi, ctypename)
            except ffi.error as ex:
                pass

    except AttributeError:
        try:
            decls = ffi._parser._declarations
        except AttributeError:
            decls = {}
        for _, ctype in decls.items():
            if isinstance(ctype, cffi.model.StructType):
                cobjs[ctype.get_c_name()] = CStructType(ffi, ctype)
            elif isinstance(ctype, cffi.model.UnionType):
                cobjs[ctype.get_c_name()] = CUnionType(ffi, ctype)

    return cobjs


def cmethod(cfunc=None, outargs=(), inoutargs=(), arrays=(), retargs=None,
           checkerr=None, doc=None):
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
        with a length equal to the int and initialized to zeros. The generated
        CFFI array will be included in the return tuple.

    * ``retargs``: (Not implemented yet.) A list of values to be returned from
      the cmethod-wrapped function. Normally the returned value will be a tuple
      containing the actual return value of the C function, followed by the
      final value of each of the ``outargs``, ``inoutargs``, and ``arrays`` in
      the order they appear in the C function's paramater list.

    * ``doc``: Optional string/object to attach to the returned function's docstring

    As an example of using ``outargs`` and ``inoutargs``, a C function with
    this signature::

        ``int cfunc(int inarg, int *outarg, float *inoutarg);``

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

    if doc:
        wrapper.__doc__ = doc
    return wrapper


def cstaticmethod(cfunc, **kwargs):
    ''' Shortcut for staticmethod(cmethod(cfunc, [kwargs ...])) '''
    return staticmethod(cmethod(cfunc, **kwargs))


def cproperty(fget=None, fset=None, fdel=None, doc=None, checkerr=None):
    ''' Shortcut to create ``cmethod`` wrapped ``property``\ s.

    E.g., this:

        >>> class MyCObj(CObject):
        ...     x = property(cmethod(get_x_cfunc), cmethod(set_x_cfunc))

    becomes:

        >>> class MyCObj(CObject):
        ...     x = cproperty(get_x_cfunc, set_x_cfunc)

    If you need more control of the outargs/etc of the cmethods, stick to the
    first form, or create and assign individual cmethods and put them in a
    normal property.

    '''

    return property(fget=cmethod(fget, checkerr=checkerr),
                    fset=cmethod(fset, checkerr=checkerr),
                    fdel=cmethod(fdel, checkerr=checkerr),
                    doc=doc)


class CStruct(object):
    ''' Provides introspection to an instantiation of a CFFI ``StructType``s and ``UnionType``s.

    Instances of this class are essentially struct/union wrappers.
    Field names are easily inspected and transparent conversion of data types is done
    where possible.

    Struct fields can be passed in as positional arguments or keyword
    arguments. ``TypeError`` is raised if positional arguments overlap with
    given keyword arguments.

    The module convenience function ``wrapall`` creates ``CStruct``\ s
    for each instantiated struct and union imported from the FFI.

    '''

    def __init__(self, ffi, struct):
        '''

        * ``ffi``: The FFI object.
        * ``structtype``: a CFFI StructType or a string for the type name
          (wihtout any trailing '*' or '[]').

        '''

        self.__fldnames = None
        self.__pfields = {}  # This is used to hold python wrappers that are linked to the underlying fields cdata

        assert isinstance(struct, ffi.CData)

        self._cdata = struct
        self.__struct_type = ffi.typeof(struct)

        if self.__struct_type.kind == 'pointer':
            self.__struct_type = self.__struct_type.item

        self._ffi = ffi

        # Sometimes structtype.name starts with a '$'...?
        try:
            self._cname = self.__struct_type.cname
        except AttributeError:
            self._cname = self.__struct_type.get_c_name()

        self.__fldnames = None if self.__struct_type.fields is None else {detail[0]: detail[1].type for detail in self.__struct_type.fields}

    def __dir__(self):
        """
        List the struct fields as well
        """
        return super(CStruct, self).__dir__() + (list(self.__fldnames.keys()) if self.__fldnames else [])

    def __getattr__(self, item):
        if item != '_CStruct__fldnames' and self.__fldnames and item in self.__fldnames:
            attr = self.__pfields.get(item, self._cdata.__getattribute__(item))
            if isinstance(attr, self._ffi.CData):
                pattr = wrap(self._ffi, attr)
                if pattr is not attr:
                    self.__pfields[item] = pattr
                    if isinstance(pattr, types.LambdaType):
                        attr = pattr(attr)
                    else:
                        attr = pattr
            return attr
        return super(CStruct, self).__getattribute__(item)

    def __setattr__(self, key, value):
        if key != '_CStruct__fldnames' and self.__fldnames and key in self.__fldnames:
            if self.__fldnames[key].cname == 'unsigned char *':
                if isinstance(value, numpy.ndarray):
                    self.__pfields[key] = value
                    value = nparrayptr(value)
                elif isinstance(value, (bytes, str)):
                    self.__pfields[key] = lambda x: self._ffi.string(x)
                    value = self._ffi.new('char[]', value) # todo untested
            elif hasattr(value, '_cdata') and value._cdata is not None:
                value = value._cdata
            return setattr(self._cdata, key, value)
        else:
            return super(CStruct, self).__setattr__(key, value)


class CStructType(object):
    ''' Provides introspection to CFFI ``StructType``s and ``UnionType``s.

    Instances have the following attributes:

    * ``ffi``: The FFI object this struct is pulled from.
    * ``cname``: The C name of the struct.
    * ``ptrname``: The C pointer type signature for this struct.
    * ``fldnames``: A list of fields this struct has.

    Instances of this class are essentially struct/union generators.
    Calling an instance of ``CStructType`` will produce a newly allocated
    struct or union.

    Struct fields can be passed in as positional arguments or keyword
    arguments. ``TypeError`` is raised if positional arguments overlap with
    given keyword arguments.

    Arrays of structs can be created with the ``array`` method.

    The module convenience function ``wrapall`` creates ``CStructType``\ s
    for each struct and union imported from the FFI.

    '''

    def __init__(self, ffi, structtype):
        '''

        * ``ffi``: The FFI object.
        * ``structtype``: a CFFI StructType or a string for the type name
          (wihtout any trailing '*' or '[]').

        '''

        self.fldnames = None
        self._cdata = None

        if isinstance(structtype, str):
            try:
                self.__struct_type = ffi.typeof(structtype.lstrip('_'))
            except AttributeError:
                self.__struct_type = ffi._parser.parse_type(structtype)

        elif isinstance(structtype, ffi.CType):
            self.__struct_type = structtype

        else:
            raise NotImplementedError("Don't know how to handle structtype of %s" % type(structtype))

        if self.__struct_type.kind == 'pointer':
            self.__struct_type = self.__struct_type.item

        self.ffi = ffi

        # Sometimes structtype.name starts with a '$'...?
        try:
            self.cname = self.__struct_type.cname
        except AttributeError:
            self.cname = self.__struct_type.get_c_name()

        self.ptrname = ffi.getctype(self.cname, '*')

        try:
            self.fldnames = None if self.__struct_type.fields is None else [detail[0] for detail in self.__struct_type.fields]
        except AttributeError:
            self.fldnames = self.__struct_type.fldnames

    def __call__(self, *args, **kwargs):
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
            for fld, val in kwargs.items():
                setattr(retval, fld, val)

            return wrap(self.ffi, retval)

    def array(self, shape):
        ''' Constructs a C array of the struct type with the given length.

        * ``shape``: Either an int for the length of a 1-D array, or a tuple
          for the length of each of len dimensions. I.e., [2,2] for a 2-D array
          with length 2 in each dimension. Hint: If you want an array of
          pointers just add an extra demension with length 1. I.e., [2,2,1] is
          a 2x2 array of pointers to structs.

        No explicit initialization of the elements is performed, however CFFI
        itself automatically initializes newly allocated memory to zeros.

        '''

        # TODO: Factor out and integrate with carray function below?
        if isinstance(shape, collections.Iterable):
            suffix = ('[%i]' * len(shape)) % tuple(shape)
        else:
            suffix = '[%i]' % (shape,)

        # TODO Allow passing initialization args? Maybe factor out some of the
        # code in CStructType.__call__?
        return self.ffi.new(self.ffi.getctype(self.cname + suffix))


class CUnion(CStruct):
    def __init__(self, ffi, uniontype):
        super(CUnion, self).__init__(ffi, uniontype)


class CUnionType(CStructType):
    def __init__(self, ffi, uniontype):
        super(CUnionType, self).__init__(ffi, uniontype)


class CType(object):
    def __init__(self, ffi, typedef):
        self.typedef = typedef
        self.ffi = ffi
        self.ctype = None

        try:
            desc = ffi.typeof(typedef + '*').item
            if desc.kind == 'struct':
                self.ctype = CStructType(ffi, desc)
            elif desc.kind == 'union':
                self.ctype = CUnionType(ffi, desc)

        except Exception as ex:
            print(ex)

    def __repr__(self):
        return "type: %s" % self.typedef

    def __call__(self, *args, **kwargs):
        if self.ctype is None:
            raise TypeError("'%s' object is not callable", self.typedef)
        return self.ctype(*args, **kwargs)


class Enum(int):
    """
    This is a base class for wrapping enum ints
    wrapenum() below will subtype it for a particular enum
    and return a wrapped result which will still work as an int
    but display/print as the string representation from the enum
    """
    _names = {}

    def __new__(cls, *args, **kwargs):
        return super(Enum, cls).__new__(cls, *args, **kwargs)

    def __str__(self):
        return self._names.get(int(self), str(int(self)))

# Cache generated enum types
_enumTypes = {}


def wrapenum(retval, enumTypeDescr):
    """
    Wraps enum int in an auto-generated wrapper class. This is used automatically when
    cmethod() returns an enum type
    :param retval: integer
    :param enumTypeDescr: the cTypeDescr for the enum
    :return: subclass of Enum
    """
    def _newEnumType(enumTypeDescr):
        _enumTypes[enumTypeDescr.cname] = type(enumTypeDescr.cname, (Enum, ), {"_names": enumTypeDescr.elements})
        return _enumTypes[enumTypeDescr.cname]
    enum = _enumTypes.get(enumTypeDescr.cname, _newEnumType(enumTypeDescr))
    return enum(retval)


class CObject(object):
    ''' A pythonic representation of a C "object"
    
    Usually representing a set of C functions that operate over a common peice
    of data. Many C APIs have lots of functions which accept some common struct
    pointer or identifier int as the first argument being manipulated. CObject
    provides a convenient abstrtaction to making this convention more "object
    oriented". See the example below. More examples can be found in the
    cfficloak unit tests.

    Use ``cproperty`` and ``cmethod`` to wrap CFFI C functions to behave like
    instance methods, passing the instance in as the first argument. See the
    doc strings for each above.

    For C types which are not automatically coerced/converted by CFFI (such as
    C functions accepting struct pointers, etc) the subclass can set a class-
    or instance-attribute named ``_cdata`` which will be passed to the CFFI
    functions instead of ``self``. The CObject can also have a ``_cnew`` static
    method (see ``cstaticmethod``) which will be called by the base class's
    ``__init__`` and the returned value assigned to the instance's ``_cdata``.

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

    Python usage (where ``libexample`` is an API object from
    ``ffi.verify()``)::

        >>> from cfficloak import CObject, cproperty, cmethod, cstaticmethod
        >>> class Point(CObject):
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

        >>> class MyStruct(CObject):
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

    def __init__(self, *args, **kwargs):
        if not hasattr(self, '_cdata') or self._cdata is None:
            if hasattr(self, '_cnew'):
                # C functions don't accept kwargs, so we just ignore them.
                self._cdata = self._cnew(*args)
            else:
                self._cdata = None

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

    if _global_ffi:
        return _global_ffi.cast('void *', nparr.__array_interface__['data'][0])


def carray(items_or_size=None, size=None, ctype='int'):
    ''' Convenience function for creating C arrays. '''

    # TODO: Support multi-dimensional arrays? Maybe it's just easier to stick
    # with numpy...

    if _global_ffi:
        if isinstance(items_or_size, int) and size is None:
            size = items_or_size
            items = None
        else:
            items = items_or_size

        if items and size > len(items):
            size = max(len(items), size or 0)
            arr = _global_ffi.new(_global_ffi.getctype(ctype, '[]'), size)
            for i, elem in enumerate(items):
                arr[i] = elem
            return arr
        else:
            return _global_ffi.new(_global_ffi.getctype(ctype, '[]'), items or size)

