# exlib
[![PyPI](https://img.shields.io/pypi/v/exlib)](https://pypi.org/project/exlib/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrachioLab/exlib/blob/master/LICENSE)

exlib is a comprehensive package showcasing our lab's work on explanation methods, featuring user-friendly modules for easy application of various techniques. 

## Installation
To install:
```
pip install exlib
```

If you have exlib already installed, please check that you have the latest version:
```
python -c "import exlib; print(exlib.__version__)"
# This should print "0.0.2". If it does not, update the package by running:
pip install -U exlib
```

In order to use `pytorch-gradcam`, install the lab version to handle more diverse shapes at 
```
grad-cam@git+https://github.com/brachiolab/pytorch-grad-cam
```

## Projects
We list below some relevant projects that use exlib heavily.

### The FIX Benchmark: Extracting Features Interpretable to eXperts
* You can find the FIX benchmark README [here](https://github.com/BrachioLab/exlib/tree/dev/fix). 
* [<a href="https://github.com/BrachioLab/brachiolab.github.io/blob/live/fix/jin2024fix.pdf">Paper</a>] [<a href="https://brachiolab.github.io/fix/">Website</a>] 

