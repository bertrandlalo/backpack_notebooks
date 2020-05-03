==================================
Online SSVEP in Riemanian manifold
==================================

Installation
============
You have the choice between virtual env or conda env:

- conda env
   .. code-block:: console

      $ conda env create -f  environment.yml
      $ conda activate eeg_ssvep_homework-env

- virtual env
   .. code-block:: console

      $ python3 -m venv eeg_ssvep_homework-env
      $ source eeg_ssvep_homework-env/bin/activate
      $ pip install -r requirements.txt

Usage
======
- Notebooks
   .. code-block:: console

      $ jupyter notebook

References
===========
- data: MOABB/SSVEPExo dataset from E. Kalunga PhD in University of Versailles [1]_. (url). (classes = rest, 13Hz, 17Hz, 21Hz)
- matlab implementation: https://github.com/emmanuelkalunga/Online-SSVEP
- paper SSVEP: https://hal.archives-ouvertes.fr/hal-01351623/document
- paper RPF: ttps://hal.archives-ouvertes.fr/hal-02015909/document
