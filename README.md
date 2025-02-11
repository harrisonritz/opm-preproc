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

download the (noisy) oddball dataset [here](https://www.dropbox.com/scl/fo/ic57kcszwilhr9sj7vggh/AA_oRr6hdeufnjo_4nGYURA?rlkey=0ke5mk35e5nasfyjfn30spx8w&dl=0), and add to `examples/oddball`.


## Preproc Config
Preprocessing parameters are configured through [YAML](https://learnxinyminutes.com/yaml/) text files. Inspired by the approach in osl-ephys, these human-readable configuration files let you keep a record of subject analyses for ease and reproducability.
You can call a config file when you execute `run_preproc_sensors.py`, and the contents of this config file will overwrite the defaults set in `set_preproc_params`. e.g.,
```zsh
python run_preproc_sensors PATH/TO/CONFIG.yaml
```
Your configuration file can update just a subset of the parameters, for example just the participant information:
```YAML
participant:
  id: 2
  session: 1
  run: 1
  task: "oddball"
  datatype: "meg"
  known_bads: [
      '2E[X]', '2E[Y]', '2E[Z]', 
      '2Z[X]', '2Z[Y]', '2Z[Z]', 
      '29[X]', '29[Y]', '29[Z]',
  ]
  do_BIDS: True
  data_root: "/Users/hr0283/Projects/opm-preproc/examples/oddball/bids" # UPDATE THIS TO YOUR PATH

```
This allows you use a similar preprocessing pipeline across multiple participants and experiments, and to more easily batch script your pipeline.


## Core Functions
- `src/opm_preproc_sensors.py`: run preprocessing pipeline for a single participant
- `src/format/opm_format_bids.py`: BIDSify Cerca OPM data, annotating experimental events
- `src/utils/amm.py`: python port of SPM functions for [Adaptative Multipole Model](https://onlinelibrary.wiley.com/doi/10.1002/hbm.26596)
- `src/utils/plot_ica_axis.py`: version of MNE's ICA GUI adapted for OPM (handling spatially-overlapping sensors)

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
