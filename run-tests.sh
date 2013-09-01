#!/bin/bash

# TODO: Also need to add doctests...?

make -C tests >/dev/null

py.test $@ tests/
