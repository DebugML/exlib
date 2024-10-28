# exlib
[![PyPI](https://img.shields.io/pypi/v/exlib)](https://pypi.org/project/exlib/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrachioLab/exlib/blob/master/LICENSE)

`exlib` is a comprehensive package showcasing our lab's work on explanation methods, featuring user-friendly modules for easy application of various techniques. 

## Installation
```
pip install exlib
```

<!--
If you have exlib already installed, please check that you have the latest version:
```
python -c "import exlib; print(exlib.__version__)"
# This should print "0.1.0". If it does not, update the package by running:
pip install -U exlib
```
-->

To use `pytorch-gradcam`, install our customized and expanded version at
```
pip install grad-cam@git+https://github.com/brachiolab/pytorch-grad-cam
```

## Projects
We list below some relevant projects that use exlib heavily.

### The FIX Benchmark: Extracting Features Interpretable to eXperts
* Documentation available [here](https://github.com/BrachioLab/exlib/tree/main/fix).
* Quick-start tutorial notebook at [`fix_demo.py`](https://colab.research.google.com/github/BrachioLab/exlib/blob/main/fix/fix_demo.ipynb)
* [<a href="https://arxiv.org/abs/2409.13684">Paper</a>] [<a href="https://brachiolab.github.io/fix/">Website</a>] [<a href="https://debugml.github.io/fix/">Blog Post</a>] 


