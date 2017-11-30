# supermjo-py

Python interface to [Super-Mjograph](http://www.mjograph.net/), which you can use as an alternative to matplotlib.
In terms of 2D plot, it is fully competent for data science, even though it does not support 3D functionality.
You can easily create publication-quality charts, by leveraging the rich GUI of macOS-native application.

![screen shot](ex1.gif)

## Installation
I will distribute this module via pip in future.
But as of now, manually install by doing
1. download supermjo.py
2. install the following dependency:
  * py-applescript, pyobjc, inspect, numpy, pandas



## Example

```python:sample
import supermjo as mjo
import numpy as np

x = np.random.randn(100)
mjo.plot(x)
```
Note that you need to launch SuperMjograph.app manually before invoking the plot command.


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

I myself heavily use this module for machine learning. In my environment, it works stably.
