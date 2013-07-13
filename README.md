Neuroparser
===========

Setup.py Installation
---------------------
The usual 

    python setup.py install

should do the trick if you have the dependencies installed.

SCONS Installation
------------------

If you have scons installed, simply type

    scons

in the top directory (neuroparser). 

Warning!
--------

This code is currently in an early state of development for public use, and will undergo many changes in the coming months. Use it at your own risk! It will likely be good to download the latest code with some frequency and rebuild by for example using 

    scons -c 

to clear away old files, and then

    scons

to rebuild them.

Usage
-----

Some basic examples of how the code can be called are in:

/neuroparser/examples/graphnet_example.py

and some more use cases can be found in:

/neuroparser/optimization/cwpath/tests/profile.py
/neuroparser/optimization/cwpath/tests/test_graphnet.py

Please report bugs or feature requests to logang@stanford.edu or on the GitHub site. --L.G. Jul 2013.

Logan Grosenick
Kiefer Katovich
Brad Klingenberg
Jonathan Taylor 

