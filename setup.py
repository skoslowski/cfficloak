from distutils.core import setup

#This is a list of files to install, and where
#(relative to the 'root' dir, where setup.py is)
#You could be more specific.
files = ['cffiwrap.py']

setup(name = 'cffiwrap',
    version = '0.1',
    description = 'A simple but flexible module for creating '
                  'object-oriented, pythonic CFFI wrappers.',
    author = 'Isaac Freeman',
    author_email = 'memotype@gmail.com',
    url = 'https://bitbucket.org/memotype/cffiwrap',
    py_modules = ['cffiwrap']
) 
