#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py
# cython: profile=True

import pstats, cProfile
from test_graphnet import train_all

cProfile.runctx("train_all()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
