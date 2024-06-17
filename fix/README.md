# The FIX Benchmark: Extracting Features Interpretable to eXperts
<!-- -------------------------------------------------------------------------------- -->

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrachioLab/exlib/blob/master/LICENSE)

## Overview
FIX is a benchmark for extracting features that are interpretable to real-world experts, spanning diverse data modalities and applications, from doctors performing gall bladder surgery to cosmologists studying supernovae. 

The FIX package contains:

 * Data loaders that automatically handle data downloading, processing, and splitting, and
 * Dataset evaluators that standardize model evaluation for each dataset.

In addition, we include an example script that runs all baselines for every setting.

For more information, please visit [our website](https://brachiolab.github.io/fix/) or read the main FIX [paper](https://github.com/BrachioLab/brachiolab.github.io/blob/live/fix/jin2024fix.pdf).
<!-- For questions and feedback, please post on the [discussion board](https://github.com/BrachioLab/exlib/discussions). -->

## Getting Started
### Installation
To use FIX, you must first install the exlib package (which has a separate README outside this FIX folder), as follows:
```
pip install exlib
```

If you have exlib already installed, please check that you have the latest version:
```
python -c "import exlib; print(exlib.__version__)"
# This should print "1.0.0". If it does not, update the package by running:
pip install -U exlib

```

### FIX Notebooks
Tutorial notebooks for each FIX setting are located in the [../notebooks/fix](https://github.com/BrachioLab/exlib/blob/master/notebooks/fix) folder.
The main dependencies needed to run them are all installed in exlib or alternatively you can use our [Dockerfile](https://github.com/BrachioLab/dockerfiles/blob/main/riceric22/exlib/Dockerfile).

### FIX Baselines
To run all baselines for every dataset setting, you can run the following script:
```
./run_fix_baselines.sh
```
The baseline feature extractors for differenet data modalities (e.g. text, time series, data) are located in [../src/exlib/features](https://github.com/BrachioLab/exlib/blob/master/src/features) folder.

## Datasets
FIX currently includes 6 datasets, which we've briefly listed below. For full dataset descriptions, please see our [paper](https://github.com/BrachioLab/brachiolab.github.io/blob/live/fix/jin2024fix.pdf).

| Dataset                 | Modality    | Labeled splits   | Expert Features  |
| ----------------------- | ----------- | ---------------- | ---------------- |
| massmaps                | Image       | train, val, test | Implicit         |
| supernova               | Time Series | train, val, test | Implicit         |
| multilingual_politeness | Text        | train, val, test | Implicit         |
| emotion                 | Text        | train, val, test | Implicit         |
| chestx                  | Image       | train, test      | Explicit         |
| cholec                  | Image       | train, test      | Explicit         |

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

### Original Datasets Citations
#### Mass Maps:
```
@article{Kacprzak_2023,
   title={CosmoGridV1: a simulated 𝗐CDM theory prediction for map-level cosmological inference},
   volume={2023},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2023/02/050},
   DOI={10.1088/1475-7516/2023/02/050},
   number={02},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Kacprzak, Tomasz and Fluri, Janis and Schneider, Aurel and Refregier, Alexandre and Stadel, Joachim},
   year={2023},
   month=feb, pages={050} }
```
#### Supernova:
```
@misc{theplasticcteam2018photometric,
      title={The Photometric LSST Astronomical Time-series Classification Challenge (PLAsTiCC): Data set},
      author={The PLAsTiCC Team and Tarek Allam Jr. au2 and Anita Bahmanyar and Rahul Biswas and Mi Dai and Lluís Galbany and Renée Hložek and Emille E. O. Ishida and Saurabh W. Jha and David O. Jones and Richard Kessler and Michelle Lochner and Ashish A. Mahabal and Alex I. Malz and Kaisey S. Mandel and Juan Rafael Martínez-Galarza and Jason D. McEwen and Daniel Muthukrishna and Gautham Narayan and Hiranya Peiris and Christina M. Peters and Kara Ponder and Christian N. Setzer and The LSST Dark Energy Science Collaboration and The LSST Transients and Variable Stars Science Collaboration},
      year={2018},
      eprint={1810.00001},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM}
}
```

#### Multilingual Politeness:
```
@inproceedings{havaldar-etal-2023-multilingual,
    title = "Multilingual Language Models are not Multicultural: A Case Study in Emotion",
    author = "Havaldar, Shreya  and
      Singhal, Bhumika  and
      Rai, Sunny  and
      Liu, Langchen  and
      Guntuku, Sharath Chandra  and
      Ungar, Lyle",
    editor = "Barnes, Jeremy  and
      De Clercq, Orph{\'e}e  and
      Klinger, Roman",
    booktitle = "Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment, {\&} Social Media Analysis",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wassa-1.19",
    doi = "10.18653/v1/2023.wassa-1.19",
    pages = "202--214",
    abstract = "Emotions are experienced and expressed differently across the world. In order to use Large Language Models (LMs) for multilingual tasks that require emotional sensitivity, LMs must reflect this cultural variation in emotion. In this study, we investigate whether the widely-used multilingual LMs in 2023 reflect differences in emotional expressions across cultures and languages. We find that embeddings obtained from LMs (e.g., XLM-RoBERTa) are Anglocentric, and generative LMs (e.g., ChatGPT) reflect Western norms, even when responding to prompts in other languages. Our results show that multilingual LMs do not successfully learn the culturally appropriate nuances of emotion and we highlight possible research directions towards correcting this.",
}
```

#### Emotion:
```
@inproceedings{demszky-etal-2020-goemotions,
    title = "{G}o{E}motions: A Dataset of Fine-Grained Emotions",
    author = "Demszky, Dorottya  and
      Movshovitz-Attias, Dana  and
      Ko, Jeongwoo  and
      Cowen, Alan  and
      Nemade, Gaurav  and
      Ravi, Sujith",
    editor = "Jurafsky, Dan  and
      Chai, Joyce  and
      Schluter, Natalie  and
      Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.372",
    doi = "10.18653/v1/2020.acl-main.372",
    pages = "4040--4054",
    abstract = "Understanding emotion expressed in language has a wide range of applications, from building empathetic chatbots to detecting harmful online behavior. Advancement in this area can be improved using large-scale datasets with a fine-grained typology, adaptable to multiple downstream tasks. We introduce GoEmotions, the largest manually annotated dataset of 58k English Reddit comments, labeled for 27 emotion categories or Neutral. We demonstrate the high quality of the annotations via Principal Preserved Component Analysis. We conduct transfer learning experiments with existing emotion benchmarks to show that our dataset generalizes well to other domains and different emotion taxonomies. Our BERT-based model achieves an average F1-score of .46 across our proposed taxonomy, leaving much room for improvement.",
}
```

#### Chest X-Ray
```
@article{majkowska2020chest,
  title={Chest radiograph interpretation with deep learning models: assessment with radiologist-adjudicated reference standards and population-adjusted evaluation},
  author={Majkowska, Anna and Mittal, Sid and Steiner, David F and Reicher, Joshua J and McKinney, Scott Mayer and Duggan, Gavin E and Eswaran, Krish and Cameron Chen, Po-Hsuan and Liu, Yun and Kalidindi, Sreenivasa Raju and others},
  journal={Radiology},
  volume={294},
  number={2},
  pages={421--431},
  year={2020},
  publisher={Radiological Society of North America}
}
```

#### Laparoscopic Cholecystectomy Surgery:
```
@article{stauder2016tum,
  title={The TUM LapChole dataset for the M2CAI 2016 workflow challenge},
  author={Stauder, Ralf and Ostler, Daniel and Kranzfelder, Michael and Koller, Sebastian and Feu{\ss}ner, Hubertus and Navab, Nassir},
  journal={arXiv preprint arXiv:1610.09278},
  year={2016}
}

@article{twinanda2016endonet,
  title={Endonet: a deep architecture for recognition tasks on laparoscopic videos},
  author={Twinanda, Andru P and Shehata, Sherif and Mutter, Didier and Marescaux, Jacques and De Mathelin, Michel and Padoy, Nicolas},
  journal={IEEE transactions on medical imaging},
  volume={36},
  number={1},
  pages={86--97},
  year={2016},
  publisher={IEEE}
}
```

## Contact
Please reach out to us if you have any questions or suggestions. You can submit an issue or pull request, or send an email to helenjin@seas.upenn.edu.

Thank you for your interest in the FIX benchamrk.


