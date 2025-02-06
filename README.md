# opm-preproc
OPM preprocessing pipeline built on [mne-python](https://github.com/mne-tools/mne-python) and [osl-ephys](https://github.com/OHBA-analysis/osl-ephys). 

Grateful for advice and code from [Lukas Rier](https://github.com/LukasRier) and [Robert Seymour](https://github.com/neurofractal).


## Installation
```terminal
git clone https://github.com/harrisonritz/opm-preproc.git
cd opm-preproc
conda env create -f environment.yml
conda activate opm-preproc
```

download the (noisy) oddball dataset [here](https://www.dropbox.com/scl/fo/w7zjyajlxgdeh8t7o9ta3/AFIjA0MpV9tJfaztgkJxYdY?rlkey=7sq0eg0hpsonf7jweypf9pekh&dl=0), and add to `examples/oddball`.


## Core Functions
- `src/opm_format_bids.py`: BIDSify Cerca OPM data, annotating experimental events
- `src/opm_preproc.py`: run preprocessing pipeline for a single participant
  - `set_participant_params()`: set participant information
  - `set_preproc_params()`: set pipeline options
  - `run_preproc()`: run pipeline (top-level function)

## Pipeline Steps
1. reject channels [osl]
3. harmonic field correction
5. temporal filter
6. continuous segment reject [osl]
7. fit ICA
8. epoch, apply ICA, and resample
9. reject epochs [osl]

## todo
- source reconstruction
- argument handling for batch scripting
