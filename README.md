# supermjo-py

Python interface to [Super-Mjograph](http://www.mjograph.net/).
You can use it as an alternative to matplotlib.
Although the functionality is not fully comparable to matplotlib, it offers to easily create publication-quality charts, by leveraging the rich GUI of macOS-native application.

![screen shot](ex1.gif)

## Installation
I will distribute this module via pip in future.
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


The argument accepts
* normal `list`
* `numpy.ndarray`
* `pandas.DataFrame` and `Series`

## Features

* Every series property (such as line and marker styles)
can be prescribed in optional arguments of the plot command.
* Very fast. Data are transferred in-memory. Hence, there is no disk I/O overhead. As a result, It takes less than 1 s for data with million of samples to complete visualization.


## API

Documented in https://github.com/moykeen/supermjo-doc/wiki/Scripting

## Development phase

I myself heavily use this module for machine learning.
In my environment, it works stably.
However, note that Super-Mjograph itself has not reached a stable release, which means this python interface can also be nontrivially changed.
