# The FIX Benchmark: Extracting Features Interpretable to eXperts
--------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrachioLab/exlib/blob/master/LICENSE)

## Overview
FIX is a benchmark for extracting features that are interpretable to real-world experts, spanning diverse data modalities and applications, from doctors performing gall bladder surgery to cosmologists studying supernovae.

For more information, please visit [our website](https://brachiolab.github.io/fix/) or read the main FIX [paper]().
<!-- For questions and feedback, please post on the [discussion board](https://github.com/BrachioLab/exlib/discussions). -->

## Getting Started
Tutorial notebooks for each FIX setting are located in the [notebooks/fix](https://github.com/BrachioLab/exlib/blob/master/notebooks/fix) folder.
The main dependencies needed to run them are all installed in exlib or alternatively you can use our [Dockerfile](https://github.com/BrachioLab/dockerfiles/blob/main/riceric22/exlib/Dockerfile).

## Datasets
FIX currently includes 6 datasets, which we've briefly listed below. For full dataset descriptions, please see our [paper]().

| Dataset                 | Modality    | Labeled splits   | Expert Features  | Citations                                            |
| ----------------------- | ----------- | ---------------- | ---------------- |----------------------------------------------------- |
| massmaps                | Image       | train, val, test | Implicit         |                                                      |
| supernova               | Time Series | train, val, test | Implicit         |                                                      | 
| multilingual_politeness | Text        | train, val, test | Implicit         |                                                      |
| emotion                 | Text        | train, val, test | Implicit         |                                                      |
| chestx                  | Image       | train, test      | Explicit         |                                                      |
| cholec                  | Image       | train, test      | Explicit         |                                                      |


## Citation
Please cite the paper as follows if you use the data or code from the FIX benchmark:
```
@article{jin2024fix,
  title={The FIX Benchmark: Extracting Features Interpretable to eXperts},
  author={Helen Jin and Shreya Havaldar and Chaehyeon Kim and Anton Xue and Weiqiu You and Helen Qu and Marco Gatti and Daniel A Hashimoto and Bhuvnesh Jain and Amin Madani and Masao Sako and Lyle Ungar and Eric Wong},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024},
  primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'}
}
```

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to helenjin@seas.upenn.edu.

Thank you for your interest in the FIX benchamrk. 




