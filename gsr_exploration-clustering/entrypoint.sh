#!/bin/bash

source activate gsr_sandbox-env
jupyter notebook \
        --port 8888 --ip 0.0.0.0 \
        --allow-root \
        --NotebookApp.token='' --NotebookApp.password='' \
        --NotebookApp.open_browser=False
