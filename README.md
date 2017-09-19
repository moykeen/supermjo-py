# supermjo-py

Python interface to [Super-Mjograph](http://www.mjograph.net/).
You can use it as an alternative to matplotlib.

![screen shot](ex1.gif)

## Installation
I will distribute the module via pip in future.
But as of now, manually install by doing
1. download supermjo.py
2. install the following dependency:
  * py-applescript, pyobjc, inspect, numpy, pandas



## Usage example

Assume you have launched SuperMjograph.app manually. Then,
```python:sample
import supermjo as mjo
import numpy as np

x = np.random.randn(100)
# mjo.figure() # if a window does not automatically appear
mjo.plot(x)
```


The argment accepts
* `numpy.ndarray`
* `pandas.DataFrame` and `Series`

The normal Python list is TBI

## Features

to be documented.

## Development phase

Just confirmed it almost works.
Also note that Super-Mjograph itself has not reached a stable release, which means interface can be nontrivially changed.
