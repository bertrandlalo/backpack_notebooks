==================================
Online SSVEP in Riemanian manifold
==================================

Installation
============
You have the choice between virtual env or conda env:

- conda env
   .. code-block:: console

      $ conda env create -f  environment.yml
      $ conda activate ssvep-env

- virtual env
   .. code-block:: console

      $ python3 -m venv ssvep-env
      $ source ssvep-env/bin/activate
      $ pip install -r requirements.txt

Usage
======
- Notebooks
   .. code-block:: console

      $ jupyter notebook
- Script to get HDF5 data
    .. code-block:: console

      $ python make_hdf5.py

    - this should download data from MOABB and convert them in HDF5
   timeflux-replayable data that will be stored in folder ./data

- Timeflux
    .. code-block:: console

      $ timeflux -d graphs/main.yaml

    - This should display events and predictions in the console.
    - If you want to try an choose the subject to replay, change line 8 in file
    `graphs/replay.yaml`. Default is `filename: data/12.hdf5`.
    - The output events will be dumped in a csv with name set lin 14 of file
    `graphs/dump.yaml`.  Default is `predictions_12.csv`.
    -  Output looks like :
    .. csv-table:: predictions_12.csv
       :file: predictions_12.csv
       :header-rows: 7

References
===========
- data: MOABB/SSVEPExo dataset from E. Kalunga PhD in University of Versailles [1]_. (url). (classes = rest, 13Hz, 17Hz, 21Hz)
- matlab implementation: https://github.com/emmanuelkalunga/Online-SSVEP
- paper SSVEP: https://hal.archives-ouvertes.fr/hal-01351623/document
- paper RPF: ttps://hal.archives-ouvertes.fr/hal-02015909/document
