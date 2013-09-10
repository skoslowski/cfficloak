#!/bin/bash

# TODO: Also need to add doctests...?

make -C tests >/dev/null

export PYTHONPATH="$PWD:$PYTHONPATH"
py.test $@ tests/
