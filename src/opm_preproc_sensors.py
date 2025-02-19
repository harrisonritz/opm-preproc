## Preprocess OPM data using MNE-Python & OSL
# Harrison Ritz (2025)


# %% import packages ==========================================================================================================

# basic imports ---------------------------------------------------------
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import gc
import psutil
import time
import yaml
import sys
from copy import deepcopy


# MEG imports ---------------------------------------------------------
import mne
from mne_bids import (
    BIDSPath,
    read_raw_bids,
)

from autoreject import (
    Ransac,
    AutoReject,
)


# local imports ---------------------------------------------------------
from utils.osl.osl_wrappers import (
    detect_badchannels, 
    detect_badsegments,
    drop_bad_epochs,
)

from utils.plot_ica_axis import plot_ica_axis
from utils.amm import compute_proj_amm

import eval.eval_preproc





# add axis methods to mne classes ==========================================================================================================

def misc_axis(self, axis='Z', sensor_wildcard=lambda axis: f"^..\\[{axis}]$"):
    """Pick sensors axis by setting non-axis channels to 'misc'.

    Creates a copy of the data where channels not matching the specified axis 
    pattern are set to 'misc' type, while preserving the original axis channels.

    Parameters
    ----------
    axis : str
        Sensor axis to keep, must be 'X', 'Y' or 'Z' (default 'Z')
    
    sensor_wildcard : callable
        Function that returns a regular expression pattern for the sensor names

    Returns
    -------
    mne.io.Raw
        A copy of the data with non-axis channels set to 'misc' type

    Notes
    -----
    - Creates a copy to preserve the original data

    Harrison Ritz, 2025
    """
   
    axis_picks = mne.pick_channels_regexp(self.info['ch_names'], sensor_wildcard(axis))

    # Create a mask for non-matching columns
    non_axis_picks = np.ones(self.info['nchan'], dtype=bool)
    non_axis_picks[axis_picks] = False

    # Change channel type to 'misc' for non-axis channels
    self_copy = self.copy()
    for i, is_non_axis in enumerate(non_axis_picks):
        if is_non_axis:
            self_copy.info['chs'][i]['kind'] = mne.io.constants.FIFF.FIFFV_MISC_CH


    return self_copy

mne.preprocessing.ICA.get_axis = misc_axis
mne.time_frequency.Spectrum.get_axis = misc_axis


def pick_axis(inst, axis='Z', sensor_wildcard=lambda axis: f"^..\\[{axis}]$"):
    """Pick sensors axis based on regular expression pattern.

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        The MNE object to pick channels from.
    axis : str
        Sensor axis to keep, must be 'X', 'Y' or 'Z' (default 'Z')
    sensor_wildcard : callable
        Function that returns a regular expression pattern for the sensor names

    Returns
    -------
    List[int]
        The indices of the picked channels.
    """
    return mne.pick_channels_regexp(inst.info['ch_names'], regexp=sensor_wildcard(axis))





def print_memory_usage():
    """Print the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")





# %% functions ==========================================================================================================



def set_preproc_params(cfg, config_path=""):
    """
    Initialize and update preprocessing parameters using a baseline configuration and an optional external config file.

    Loads the baseline configuration for an oddball experiment and updates the settings
    if a configuration file is provided. Also sets sensor-specific configurations and adjusts
    general parameters (e.g., for speed run mode).

    Parameters
    ----------
    cfg : dict
        An empty or pre-existing configuration dictionary.
    config_path : str, optional
        File path to an external YAML configuration file, by default "".

    Returns
    -------
    dict
        The updated configuration dictionary with all preprocessing parameters.
    """

    # set-up configuration ==========================================================================================================
    print("\n\n\nloading configuration ---------------------------------------------------\n")

    # baseline configuration (set for oddball example) ---------------------------------------------------------
    base_config = """
    participant:
        id: 2
        session: 1
        task: "oddball"
        datatype: "meg"
        known_bads: [
            '2E[X]', '2E[Y]', '2E[Z]', 
            '2Z[X]', '2Z[Y]', '2Z[Z]', 
            '29[X]', '29[Y]', '29[Z]',
        ]
        do_BIDS: True
        data_root: "/Users/hr0283/Projects/opm-preproc/examples/oddball/bids" # UPDATE THIS TO YOUR PATH
        data_path:          # set manually if not using BIDS
        emptyroom_path:     # set manually if not using BIDS

    info:
        sample_rate:
        line_freq: 60.0
        sensor_wildcard: '^..\\[{axis}]$'

    general:
        save_label: "test-preproc_"
        save_raw: True
        save_epochs: True
        save_param: True
        save_report: True
        n_jobs: -1
        speed_run: False

    read_data:
        plot: False

    eval_preproc:
        run: [False, False, True]
        function: eval.eval_preproc.eval_oddball
        cv: 10
        plot_axes: ['Z']
        save: False

    channel_reject:
        run: True
        plot: False
        method: "osl"  
        dur: 5.0
        filter: True
        eSSS: False
        sec: 1.0
        interpolate_bads: False

    HFC:
        run: True
        plot: True
        amm: True
        external_order: 4 # use this for standard HFC
        internal_order: 8
        corr_lim: .95
        apply: True

    temporal_filter:
        run: True
        plot: False
        plot_topos: False
        plot_bands: {
            'Delta (0-4 Hz)': [0, 4], 
            'Theta (4-8 Hz)': [4, 8], 
            'Alpha (8-12 Hz)': [8, 12], 
            'Beta (12-30 Hz)': [12, 30], 
            'Gamma (30-45 Hz)': [30, 45]
        }
        plot_bands_trouble: {'OPM Trouble (13-16 Hz)': [13, 16]}
        plot_axis: ['X', 'Y', 'Z']
        range: [0.1, 30]
        window: "blackman"
        notch_spectrum: True

    segment_reject:
        run: True
        plot: False
        thresh: 0.05
        sec: 1.0

    ICA:
        run: True
        plot_axes: ['Z'] 
        n_components: 64
        method: "picard"
        params: {"ortho": True, "extended": True}
        decim: 4
        tstep: 1.0
        max_iter: 1000
        random_state: 99
        auto_label: False
        apply: True
        save: False
        load: False

    epoch:
        plot: False
        tmin: -0.5
        tmax: 0.5
        decim: 2

    epoch_reject:
        run: True
        reject_plot: False
        method: 'osl'
        ar-interp: [0, 1, 2, 3]
    
    """
    


    # Load config file  ---------------------------------------------------------
    cfg = yaml.safe_load(base_config)
    if config_path:
        print(f"\n\nloading config: {config_path}\n")
        
        with open(config_path, 'r') as stream:
            proc = yaml.safe_load(stream)

        cfg.update(proc)
        cfg['participant']['config_path'] = config_path

    # Evaluate sensor wildcard ---------------------------------------------------------
    wildcard = cfg['info']['sensor_wildcard'] 
    cfg['info']['sensor_wildcard'] = lambda axis: wildcard.format(axis=axis)


    # Adjust general parameters if needed ---------------------------------------------------------
    if cfg['general']['speed_run']:
        print('\nSPEED RUN ========================================== \n')
        cfg['HFC']['plot'] = False
        cfg['HFC']['amm'] = False
        cfg['temporal_filter']['plot'] = False
        cfg['temporal_filter']['notch_spectrum'] = False
        cfg['epoch']['plot'] = False
        cfg['epoch_reject']['plot'] = False
        cfg['ICA']['n_components'] = 8
        cfg['ICA']['save'] = False
        cfg['ICA']['apply'] = True
        cfg['ICA']['auto_label'] = False
        cfg['ICA']['decim'] = 10
        cfg['eval_preproc']['run'] = [False, False, False]
    
    print("\nCONFIGURATION:")
    print(cfg)
    print('\nsave label: ', cfg['general']['save_label'])
    print('sensor wildcard: ', cfg['info']['sensor_wildcard']('<axis>'))

    if any(cfg['eval_preproc']):
        print(f"custom eval function: {cfg['eval_preproc']['function']}\n")

        module_name, func_name = cfg['eval_preproc']['function'].rsplit('.', 1)
        mod = __import__(module_name, fromlist=[func_name])
        cfg['eval_preproc']['function'] = getattr(mod, func_name)
       


    print("\n---------------------------------------------------\n")
    return cfg
    


def make_paths(cfg):
    """
    Construct and verify directory paths for data storage and output based on the configuration.

    Creates necessary directories for ICA, preprocessed data, and reports. Updates the configuration
    with the file paths required to save raw data, epoched data, parameters, and reports.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing participant and general settings.

    Returns
    -------
    dict
        The updated configuration with all required file paths.
    """

    print("\n\n\nMaking paths ---------------------------------------------------\n")
    if (cfg['participant']['do_BIDS']):
        
        bids_path = BIDSPath(
            subject = f"{cfg['participant']['id']:03}", 
            session = f"{cfg['participant']['session']:02}", 
            task = cfg['participant']['task'],
            datatype = cfg['participant']['datatype'], 
            root = cfg['participant']['data_root']
        )
        cfg['participant']['data_path'] = bids_path

        emptyroom_path = BIDSPath(
            subject = f"{cfg['participant']['id']:03}", 
            session = f"{cfg['participant']['session']:02}", 
            task = 'emptyroom',
            datatype = cfg['participant']['datatype'], 
            root = cfg['participant']['data_root']
        )
        cfg['participant']['emptyroom_path'] = emptyroom_path

    elif cfg['participant']['data_path']:
        print('BIDS path not set. EDIT THIS LINE.')
    else:
        warnings.warn("BIDS path not set. Add code here to set the file path manually.")

   

    ica_dir = os.path.join(cfg['participant']['data_root'], "derivatives", "ICA")
    os.makedirs(ica_dir, exist_ok=True)
    cfg['general']['ica_savename'] = os.path.join(ica_dir, f"{bids_path.basename}-ica.fif")

    preproc_dir = os.path.join(cfg['participant']['data_root'], "derivatives", "preproc")
    os.makedirs(preproc_dir, exist_ok=True)

    cfg['general']['raw_savename'] = os.path.join(preproc_dir, f"{bids_path.basename}_{cfg['general']['save_label']}_preproc-raw.fif")
    cfg['general']['epochs_savename'] = os.path.join(preproc_dir, f"{bids_path.basename}_{cfg['general']['save_label']}_preproc-epo.fif")
    cfg['general']['param_savename'] = os.path.join(preproc_dir, f"{bids_path.basename}_{cfg['general']['save_label']}_preproc-params.json")

    report_dir = os.path.join(cfg['participant']['data_root'], "derivatives", "report")
    os.makedirs(report_dir, exist_ok=True)
    cfg['general']['report_savename'] = os.path.join(report_dir, f"{bids_path.basename}_{cfg['general']['save_label']}_preproc_report.html")

    print("\n---------------------------------------------------\n")
    return cfg



def read_data(cfg):
    """
    Read raw and empty room data from a BIDS-compliant dataset.

    Loads the raw MEG file along with the corresponding empty room file using mne_bids.
    Optionally plots the power spectral density (PSD) of the data, and updates the configuration
    with the sampling frequency and line frequency.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary specifying data paths and participant parameters.

    Returns
    -------
    tuple
        A tuple containing the updated configuration, the raw data, and the empty room raw data.
    """

    # read-in data ==========================================================================================================
    print("\n\n\nReading in data ---------------------------------------------------\n")
    print(f"Participant: {cfg['participant']['id']}, Session: {cfg['participant']['session']}, Task: {cfg['participant']['task']}, Datatype: {cfg['participant']['datatype']}")
    print(f"data path: {cfg['participant']['data_path']}")


    # Read in raw file
    raw = read_raw_bids(
        bids_path=cfg['participant']['data_path'], 
        extra_params=dict(preload=True))


    # Read in emptyroom file
    raw_emptyroom = read_raw_bids(
        bids_path=cfg['participant']['emptyroom_path'], 
        extra_params=dict(preload=True))



    # plot PSD
    if cfg['read_data']['plot']:
        _, axes = plt.subplots(2, 1, figsize=(10, 8))

        raw.compute_psd(fmin=0.1, fmax=150, picks="mag").plot(
            amplitude=True,
            picks="mag",
            axes=axes[0],
            show=False
        )
        axes[0].set_title('PSD - Raw')

        raw_emptyroom.compute_psd(fmin=0.1, fmax=150, picks="mag").plot(
            amplitude=True,
            picks="mag",
            axes=axes[1],
            show=False
        )
        axes[1].set_title('PSD - Empty Room')

        plt.show()


    # set sampling frequency and line frequency
    cfg['info']['sample_rate'] = raw.info['sfreq']
    cfg['info']['line_freq'] = raw.info['line_freq']
    if not cfg['info']['line_freq']:
        cfg['info']['line_freq'] = 60.0
    print(f"Sampling rate: {cfg['info']['sample_rate']} Hz")


    print("\n---------------------------------------------------\n")
    return cfg, raw, raw_emptyroom



def channel_reject(cfg, raw, raw_emptyroom=None):
    """
    Identify and mark bad channels in MEG data using various rejection methods.

    Applies one of several channel rejection methods (e.g., OSL, Maxwell, manual, RANSAC) 
    as specified in the configuration. Optionally plots the results of the channel rejection.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with channel rejection settings.
    raw : mne.io.Raw
        The raw MEG data to process.
    raw_emptyroom : mne.io.Raw, optional
        The empty room data required for some rejection methods, by default None.

    Returns
    -------
    tuple
        Updated configuration and raw data with bad channels marked.
    """
    # channel rejection ==========================================================================================================
    
    
    print("\n\n\nChannel rejection ---------------------------------------------------\n")


    # add known bad channels
    if len(cfg['participant']['known_bads']) > 0:
        print("Adding known bad channels...")
        raw.info['bads'].extend(cfg['participant']['known_bads'])
        print("Known bads: ", raw.info['bads'])


    ransac = False
    match cfg['channel_reject']['method']:

        case "osl":

            print("Detecting bad channels using OSL")

            if cfg['channel_reject']['filter']:

                raw_filt = raw.copy().filter(l_freq=cfg['temporal_filter']['range'][0], h_freq=cfg['temporal_filter']['range'][1], method='iir')

                raw_filt = detect_badchannels(raw_filt, "mag", 
                                        ref_meg=None, 
                                        significance_level=0.05, 
                                        segment_len=round(raw.info['sfreq']*cfg['channel_reject']['sec']),
                                        )
                
                raw.info['bads'] = raw_filt.info['bads']
                del raw_filt

            else:
                raw = detect_badchannels(raw, "mag", 
                                        ref_meg=None, 
                                        significance_level=0.05, 
                                        segment_len=round(raw.info['sfreq']*cfg['channel_reject']['sec']),
                                        )
            

        case "maxwell":

            print("Detecting bad channels using maxwell")
            start_time = time.time()

            if cfg['channel_reject']['eSSS']:

                print("-- running eSSS")
                raw_emptyroom.info['bads'] = raw.info['bads']
                print('raw info\n', raw.info, '\nemptyroom info\n', raw_emptyroom.info, '\n')
                empty_room_projs = mne.compute_proj_raw(raw_emptyroom, n_mag=3)
               
                print('proj_size', empty_room_projs[0]['data']['data'].shape )
               

                noisy_chans, flat_chans, scores = mne.preprocessing.find_bad_channels_maxwell(raw.copy(), 
                                                                                              return_scores=True, 
                                                                                              coord_frame='meg',
                                                                                              extended_proj=empty_room_projs)
                

            else:

                print("-- running SSS")
                noisy_chans, flat_chans, scores = mne.preprocessing.find_bad_channels_maxwell(raw.copy(), return_scores=True, coord_frame='meg')


            print(f"Maxwell channel rejection took {time.time()-start_time:0.2f} seconds")
            print('maxwell noisy: ', noisy_chans)
            raw.info['bads'].extend(noisy_chans)
            print('maxwell flat: ', flat_chans)
            raw.info['bads'].extend(flat_chans)

            if cfg['channel_reject']['plot']: 
                # Plot noisy channel scores as heatmap
                plt.figure(figsize=(10, 6))
                plt.imshow(scores['scores_noisy'], aspect='auto')
                plt.yticks(range(len(scores['ch_names'])), scores['ch_names'], ha='right')
                plt.colorbar(label='Score')
                plt.set_cmap('Reds')
                plt.clim(np.nanmin(scores['limits_noisy']), None)
                plt.title('Maxwell Filter Noise Scores by Channel')
                plt.tight_layout()
                plt.show()


        case "manual":

            print("Manually rejecting channels")
            raw.plot(block=True)


        case "ransac":

            print("Detecting bad channels using RANSAC")

            # Create epochs for RANSAC
            epochs_ransac = mne.Epochs(raw, 
                                    events=None, 
                                    tmin=cfg['epoch']['tmin'], tmax=cfg['epoch']['tmax'], 
                                    baseline=None, 
                                    preload=True)

            # Fit RANSAC
            ransac = Ransac(verbose=True, picks='mag', n_jobs=cfg['general']['n_jobs'], random_state=99)
            ransac = ransac.fit(epochs_ransac)

            # Apply RANSAC
            raw.info['bads'].extend(ransac.bad_chs_)
            del epochs_ransac
            del ransac


        case "None":

            print("No channel rejection method specified. Skipping channel rejection.")


        case _:
            raise Exception("channel reject not recognized")

    print(f"identified {len(raw.info['bads'])} bad channels...")
    print('bads: ', raw.info['bads'])

    # interpolate bad channels
    if cfg['channel_reject']['interpolate_bads']:
        raw.interpolate_bads()
        print('interpolated bad channels')
    else:
        print('no channel interpolation')



    if cfg['channel_reject']['plot']:

        raw_filt = raw.copy().pick('mag').filter(l_freq=.1, h_freq=150, method='iir')
        raw_filt.plot(block=True, scalings={"mag": 8e-12}, n_channels=32, duration=120)

        raw.info['bads'] = raw_filt.info['bads'] # transfer bads
        del raw_filt

    print("\n---------------------------------------------------\n")
    return cfg, raw



def hfc_proj(cfg,raw):
    """
    Apply harmonic field correction (HFC) or AMM to the raw MEG data.

    Computes the appropriate projection for HFC/AMM based on the configuration settings.
    Adds the computed projection to the raw data and, if enabled, applies the projection.
    Also provides visual comparison of the power spectra before and after applying HFC.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing HFC settings.
    raw : mne.io.Raw
        The raw MEG data.

    Returns
    -------
    tuple
        The updated configuration and the raw data after the HFC projection.
    """
    # harmonic field correction ==========================================================================================================
    print("\n\n\nHarmonic Field Correction ---------------------------------------------------\n")


    # compute HFC
    raw_pre = raw.copy()

    # HFC
    if not cfg['HFC']['amm']:
        print(f"computing HFC order {cfg['HFC']['external_order']}")
        hfc_proj = mne.preprocessing.compute_proj_hfc(raw.info, order=cfg['HFC']['external_order'], picks="mag")
    else:
        print(f"computing AMM (external: {cfg['HFC']['external_order']}, internal: {cfg['HFC']['internal_order']})")
        hfc_proj = compute_proj_amm(raw, 
                                    Lout=cfg['HFC']['external_order'], 
                                    Lin=cfg['HFC']['internal_order'], 
                                    corr=cfg['HFC']['corr_lim']
                                    )
    raw.add_proj(hfc_proj)


    # apply HFC    
    if cfg['HFC']['apply']:
        raw.apply_proj(verbose="error")
        print("applied HFC")
    else:
        print("HFC not applied")



    


    # plot HFC
    if cfg['HFC']['plot']:


        # raw.plot(block=True)


        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 16))
        
        # Plot PSD before HFC
        psd_orig = raw_pre.compute_psd(fmin=0, 
                                       fmax=2*cfg['info']['line_freq'], 
                                       picks="mag",
                                       n_fft=2000)
        psd_orig.plot(
            axes=ax1,
            picks="mag",
            show=False)
        ax1.set_title('PSD before HFC')

        
        # Plot PSD after HFC
        psd_hfc = raw.copy().apply_proj(verbose="error").compute_psd(fmin=0, 
                                                                     fmax=2*cfg['info']['line_freq'], 
                                                                     picks="mag",
                                                                     n_fft=2000)
        
        psd_hfc.plot(
            axes=ax2,
            picks="mag",
            show=False)
        ax2.set_title('PSD after HFC')


        # Set ylims to be the same
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ax1.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))
        ax2.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))



        # HFC shielding
        shielding = 10 * np.log10(psd_orig[:] / psd_hfc[:])
        plot_kwargs = dict(lw=1, alpha=0.5)
        plot_kwargs2 = {'lw': 4, 'alpha': 1, 'color': 'black'}
        plot_kwargs0 = {'lw': 2, 'alpha': .5, 'color': 'black'}

        ax3.plot(psd_orig.freqs, shielding.T, **plot_kwargs)
        ax3.plot(psd_orig.freqs, shielding.mean(axis=0), **plot_kwargs2)
        ax3.plot(psd_orig.freqs, np.zeros(psd_orig.freqs.shape[0]), **plot_kwargs0) 
        ax3.grid(True, ls=":")
        ax3.set(
            xlim=(0, 120),
            title="Shielding After HFC",
            xlabel="Frequency (Hz)",
            ylabel="Shielding (dB)",
        )

        plt.show()
        del psd_orig, psd_hfc 




    del raw_pre 




    print("\n---------------------------------------------------\n")
    return cfg, raw



def temporal_filter(cfg, raw):
    """
    Filter temporal signals from the raw MEG data.

    Performs a notch filter (using spectrum fit or traditional method) followed by
    separate high-pass and low-pass filtering. Optionally, it plots the power spectral
    density before and after filtering and may display topographic maps.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with temporal filtering parameters.
    raw : mne.io.Raw
        The raw MEG data.

    Returns
    -------
    tuple
        The updated configuration and the temporally filtered raw data.
    """
    # resample & filter ==========================================================================================================
    print("\n\n\nTemporal Filter ---------------------------------------------------\n")


    # plot before filter
    if cfg['temporal_filter']['plot']:
        _, axs = plt.subplots(2, 1, figsize=(10, 8))
        raw.compute_psd(fmin=0, 
                        fmax=120,
                        n_fft=2000, 
                        picks="mag",
                        ).plot(
                            picks="mag",
                            xscale='log',
                            axes = axs[0])

    # Notch Filter ---------------------------------------------------------
    if cfg['temporal_filter']['notch_spectrum']:

        print('\n\nnotch filter: spectrum fit ----------\n')
        raw.notch_filter(freqs=None, 
                         method='spectrum_fit', 
                         filter_length='10s',
                         n_jobs=cfg['general']['n_jobs'],
                         )
        
    else:

        print('\n\nnotch filter: traditional method ----------\n')
        raw.notch_filter(cfg['info']['line_freq'])
        if (2*cfg['info']['line_freq']) <  (cfg['temporal_filter']['range'][1]+10):
            for ff in range(2, int(1+np.ceil((cfg['temporal_filter']['range'][1] + 10) / cfg['info']['line_freq']))):
                print(f"\n\nnotch filter: {cfg['info']['line_freq']*ff} Hz ----------\n")
                raw.notch_filter(cfg['info']['line_freq']*ff)


    # seperately high-pass filter then low-pass filter ---------------------------------------------------------
    raw.filter(l_freq=cfg['temporal_filter']['range'][0], 
               h_freq=None, 
               fir_window=cfg['temporal_filter']['window'],
               ).filter(l_freq=None, 
               h_freq=cfg['temporal_filter']['range'][1], 
               fir_window=cfg['temporal_filter']['window'],
               )


    # plot after filter
    if cfg['temporal_filter']['plot']:

        spec_filt = raw.compute_psd(fmin=0, 
                                    fmax=120, 
                                    picks="mag",
                                    n_fft=2000)
        
        # plot PSD after filter
        spec_filt.plot(
            picks="mag",
            xscale='log',
            axes = axs[1])
        plt.show()

        # plot filtered topomaps
        if cfg['temporal_filter']['plot_topos']:
            for axis in cfg['temporal_filter']['plot_axis']:

                if cfg['temporal_filter']['plot_bands_trouble']:
                    spec_filt.get_axis(axis, sensor_wildcard=cfg['info']['sensor_wildcard']).plot_topomap(
                         bands=cfg['temporal_filter']['plot_bands_trouble'],
                         ch_type='mag',
                         normalize=True,
                         show_names=True,
                         show=True)
                

                if cfg['temporal_filter']['plot_bands']:
                    spec_filt.get_axis(axis, sensor_wildcard=cfg['info']['sensor_wildcard']).plot_topomap(
                        bands=cfg['temporal_filter']['plot_bands'], 
                        ch_type='mag', 
                        normalize=True, 
                        show_names=True,
                        show=True)
            
        del spec_filt

        


    # avoid memeory leak
    gc.collect()

    print("\n---------------------------------------------------\n")
    return cfg, raw



def segment_reject(cfg,raw,metric='std'):
    """
    Remove continuous segments of the data based on statistical metrics.

    Uses either the 'kurtosis' or 'std' metric to detect and reject segments containing artifacts.
    Optionally allows for review of the affected segments through plotting.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with segment rejection parameters.
    raw : mne.io.Raw
        The raw MEG data.
    metric : str, optional
        The metric for segment rejection ('kurtosis' or 'std'), by default 'std'.

    Returns
    -------
    tuple
        Updated configuration and raw data with bad segments removed.
    """
    # reject continious segments ==========================================================================================================
    print('\n\nsegment rejection ---------------------------------------------------\n')


    if metric=='kurtosis':

        raw = detect_badsegments(
            raw,
            picks='mag',
            detect_zeros=False,
            segment_len=round(raw.info['sfreq']*1.0),
            significance_level=0.05,
            metric='kurtosis',
            channel_wise = False,
        )

    else:

        raw = detect_badsegments(
                raw,
                picks="mag",
                ref_meg=False,
                metric="std",
                detect_zeros=False,
                channel_wise=False,
                segment_len=round(raw.info['sfreq']*cfg['segment_reject']['sec']),
                channel_threshold=cfg['segment_reject']['thresh'],
                significance_level=cfg['segment_reject']['thresh'],
                )
        
        # raw = detect_badsegments(
        #         raw,
        #         picks="mag",
        #         ref_meg=False,
        #         metric="std",
        #         mode="diff",
        #         detect_zeros=False,
        #         channel_wise=False,
        #         segment_len=round(raw.info['sfreq']*cfg['segment_reject']['sec']),
        #         channel_threshold=cfg['segment_reject']['thresh'],
        #         significance_level=cfg['segment_reject']['thresh'],
        #         )
    
    if cfg['segment_reject']['plot']:
        raw.plot(block=True)


    print("\n---------------------------------------------------\n")
    return cfg, raw



def fit_ica(cfg, raw):
    """
    Fit an Independent Component Analysis (ICA) model to the MEG data.

    Prepares the data for ICA by filtering and excluding bad channels. Loads a precomputed ICA
    model if available; otherwise, computes a new ICA solution. Optionally performs automatic
    labeling of artifact components and visualizes the ICA components and sources.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing ICA parameters.
    raw : mne.io.Raw
        The raw MEG data.

    Returns
    -------
    tuple
        The updated configuration and the fitted ICA object.
    """
    # ICA ==========================================================================================================
    print("\n\n\nICA ---------------------------------------------------\n")

    # filter for ICA
    if cfg['temporal_filter']['range'][0] < 1.0:
        raw_ica = raw.copy().filter(l_freq=1, h_freq=None, fir_window=cfg['temporal_filter']['window']).pick(picks="meg", exclude=raw.info['bads'])
    else:
        raw_ica = raw.copy().pick(picks="meg", exclude=raw.info['bads'])

    # load for run
    if cfg['ICA']['load'] and os.path.isfile(cfg['general']['ica_savename']):

        print(f"loading ICA from {cfg['general']['ica_savename']}")
        ica = mne.preprocessing.read_ica(cfg['general']['ica_savename'])

    else:        

        print("Fitting ICA...")
        ica = mne.preprocessing.ICA(n_components=cfg['ICA']['n_components'], 
                                    max_iter=cfg['ICA']['max_iter'],
                                    random_state=cfg['ICA']['random_state'], 
                                    method=cfg['ICA']['method'],
                                    fit_params=cfg['ICA']['params'],
                                    )
        

        # fit ICA ---------------------------------------------------------
        ica.fit(raw_ica, 
                decim=cfg['ICA']['decim'],
                tstep=cfg['ICA']['tstep'],
                reject_by_annotation=True,
                )
        
        var_explained = ica.get_explained_variance_ratio(raw_ica, ch_type='mag')
        
        print('\nICA info ----------\n', ica, '\n', ica.info, '\n')
        print(f"varience expalined: {100*var_explained['mag']:0.2f}%")
        print(f"\nICA fit complete ----------\n\n")




            
    # auto-label
    if cfg['ICA']['auto_label']:
        print('\nfind bad muscles components ---- \n')
        ica.exclude.extend(ica.find_bads_muscle(raw_ica)[0])
        print('\nfind bad ECG components ---- \n')
        ica.exclude.extend(ica.find_bads_ecg(raw_ica)[0])


    print(f"\nICA exclude: {ica.exclude} ----------\n")


    # plot ICA ---------------------------------------------------------

    # plot ICA components
    for axis in cfg['ICA']['plot_axes']:
        
        # plot all components
        plot_ica_axis(ica, 
                      raw_ica, 
                      axis=axis,  
                      sensor_wildcard=cfg['info']['sensor_wildcard']
                      )

       

    ica.plot_sources(raw_ica, block=True)
    
    del raw_ica

    if cfg['ICA']['save']:
        print(f"saving ICA to {cfg['general']['ica_savename']}")
        ica.save(cfg['general']['ica_savename'], overwrite=True)
    else:
        print("not saving ICA")


    print("\n---------------------------------------------------\n")

    return cfg, ica



def create_epoch(cfg, raw, ica):
    """
    Create epochs from the preprocessed raw data and apply ICA corrections.

    Generates epochs using predefined time windows and decimation. If an ICA model is provided,
    overlays its effects on the epochs and applies the ICA correction.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with parameters for epoch creation.
    raw : mne.io.Raw
        The raw MEG data.
    ica : mne.preprocessing.ICA or None
        The ICA object for artifact removal, if available.

    Returns
    -------
    tuple
        The updated configuration and the created epochs object.
    """
    # create standard & ICA epochs ==========================================================================================================
    print("\n\n\nEpoch ---------------------------------------------------\n")


    epochs = mne.Epochs(raw, 
                events=None, 
                tmin=cfg['epoch']['tmin'], tmax=cfg['epoch']['tmax'], 
                baseline=None, # don't baseline before ICA
                preload=True,
                decim=cfg['epoch']['decim'],
                )
   
    if ica is not None:
        ica.plot_overlay(epochs.average(), exclude=ica.exclude, picks="mag")
        ica.apply(epochs, exclude=ica.exclude)

    print('\nEpoch info ----------\n', epochs, '\n', epochs.info, '\n')
    print("\n---------------------------------------------------\n")
    return cfg, epochs



def reject_epoch(cfg, epochs):
    """
    Reject epochs containing artifacts based on configured criteria.

    Supports epoch rejection using either OSL or Autoreject methods. If a method is incorrectly
    specified, an exception is raised.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with epoch rejection settings.
    epochs : mne.Epochs
        The epoched MEG data.

    Returns
    -------
    tuple
        Updated configuration and epochs with bad epochs removed.
    """
    # Epoch rejection 1 ==========================================================================================================
    print("\n\n\nEpoch rejection ---------------------------------------------------\n")


    # detect bad epochs
    match cfg['epoch_reject']['method']:

        case "osl":
            print("Detecting bad epochs using OSL")
            _, bad_epochs = drop_bad_epochs(
                epochs.copy(),
                ref_meg=None,
                picks="mag",
            )

        case "autoreject" | "ar":
            print("Detecting bad epochs using autoreject")

            # fit autoreject
            epochs_ar = epochs.copy().pick("mag")

            ar = AutoReject(n_interpolate=cfg['epoch_reject']['ar-interp'], random_state=99, 
                            picks="mag",
                            n_jobs=cfg['general']['n_jobs'], verbose=True)
            ar.fit(epochs_ar)
            
            # remove rejected epochs
            reject_log = ar.get_reject_log(epochs_ar)
            del epochs_ar

            if np.any(reject_log.bad_epochs):
                epochs.drop(np.nonzero(reject_log.bad_epochs)[0],
                            reason='bad_autoreject')
    
        case "None":
            print("No epoch rejection method specified. Skipping epoch rejection.")
        

        case _:
            raise Exception("epoch reject not recognized")


    print("\n---------------------------------------------------\n")
    return cfg, epochs



def save_preproc(cfg, raw, epochs):
    """
    Save the preprocessed raw and epoched MEG data to disk.

    Checks configuration flags and writes the preprocessed data files to the specified locations.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with saving preferences.
    raw : mne.io.Raw
        The preprocessed raw MEG data.
    epochs : mne.Epochs
        The epoched MEG data.

    Returns
    -------
    None
    """
    # save preproc data ==========================================================================================================
    print("\n\n\nSaving preproc data ---------------------------------------------------\n")

    if cfg['general']['save_raw']:
        print(f"saving raw data to {cfg['general']['raw_savename']}")
        raw.save(cfg['general']['raw_savename'], overwrite=True)

    if cfg['general']['save_epochs']:
        print(f"saving epoched data to {cfg['general']['epochs_savename']}")
        epochs.save(cfg['general']['epochs_savename'], overwrite=True)


   
    print("\n---------------------------------------------------\n")



def save_params(cfg):
    """
    Save preprocessing parameters to a JSON file.

    Converts non-serializable items within the configuration to strings before saving.
    Writes the complete configuration to disk as a JSON file.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary containing all preprocessing parameters.

    Returns
    -------
    None
    """
    # save preproc data ==========================================================================================================
    print("\n\n\nSaving fitting parameters ---------------------------------------------------\n")

    # Save parameters to json
    print(f"saving parameters to {cfg['general']['param_savename']}")

    # Convert any non-serializable objects to strings
    param_save = deepcopy(cfg)
    for section in cfg.keys():
        for key, value in param_save[section].items():
            if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                param_save[section][key] = str(value)

    # Save to json
    with open(cfg['general']['param_savename'], 'w') as f:
        json.dump(param_save, f, indent=4)

    del param_save


    print("\n---------------------------------------------------\n")



def save_report(cfg, raw, raw_emptyroom, epochs, ica):
    """
    Generate and save a comprehensive report of the preprocessing pipeline.

    Constructs an MNE Report by incorporating system information,
    raw data, empty room data, ICA outputs, and epochs, and saves the report as an HTML file.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with report generation settings.
    raw : mne.io.Raw
        The preprocessed raw MEG data.
    raw_emptyroom : mne.io.Raw
        The preprocessed empty room data.
    epochs : mne.Epochs
        The epoched MEG data.
    ica : mne.preprocessing.ICA
        The ICA object used for artifact correction.

    Returns
    -------
    None
    """
    # save report ==========================================================================================================
    print("\n\n\nSaving report ---------------------------------------------------\n")

    report = mne.Report(verbose=True,
                        info_fname=cfg['general']['epochs_savename'],
                        subject=f"{cfg['participant']['id']:03}",
                        title="Preprocessing Report",
                        )
    
    report.add_sys_info(title="System Information")
    

    # raw
    report.add_raw(raw, 
                   title="Raw Data",
                   butterfly=5,
                   psd=True,
                   )
    
    # empty room
    report.add_raw(raw_emptyroom, 
                title="emptyroom data",
                butterfly=5,
                psd=True,
                )
    
    # projections
    # evoked = epochs.get_axis("Z").average()
    # report.add_projs(info=evoked.info, 
    #                  title="Projections (Z)",
    #                  )
    
    # ICA
    report.add_ica(ica.get_axis("Z", sensor_wildcard=cfg['info']['sensor_wildcard']),
                    title="ICA (Z)",
                    inst=epochs,
                    n_jobs=cfg['general']['n_jobs'],
                    )
    
    # epochs
    report.add_epochs(epochs,
                      title="Epochs",
                      )


    # save report
    print(f"saving report to {cfg['general']['report_savename']}")
    report.save(cfg['general']['report_savename'], overwrite=True)


                      





    











# %% run preproc ==========================================================================================================


def run_preproc(config_path=""):
    """
    Execute the full preprocessing pipeline for OPM data.

    Initializes configurations, reads data, applies artifact rejection, filtering,
    harmonic field correction, ICA, epoch creation and rejection, evaluation, and finally
    saves the processed data, parameter configurations, and a detailed report.

    Parameters
    ----------
    config_path : str, optional
        Optional file path to an external YAML configuration file to override baseline parameters,
        by default "".

    Returns
    -------
    None
    """
    # %% init ==========================================================================================================

    # set params ---------------------------------------------------------
    cfg = dict()
    cfg = set_preproc_params(cfg, config_path)
    cfg = make_paths(cfg)


    # load data ---------------------------------------------------------
    cfg, raw, raw_emptyroom = read_data(cfg)
    

    # initial fit ---------------------------------------------------------
    if cfg['eval_preproc']['run'][0]:
        cfg['eval_preproc']['function'](cfg, raw)
    else:
        print("\nno evaluation ------------------------------------\n")



    # %% artifact rejection ==========================================================================================================
    

    # reject segments ---------------------------------------------------------
    if cfg['segment_reject']['run']:
        cfg, raw = segment_reject(cfg, raw, metric='kurtosis')
    else:
        print("\nno segment rejection ------------------------------------\n")



    # channel rejection ---------------------------------------------------------
    if cfg['channel_reject']['run']:
        cfg, raw = channel_reject(cfg, raw, raw_emptyroom=raw_emptyroom)
    else:
        print("\nno channel rejection ------------------------------------\n")


    # harmonic field correction ---------------------------------------------------------
    if cfg['HFC']['run']:
        cfg, raw = hfc_proj(cfg, raw)
    else:
        print("\nno HFC ------------------------------------\n")


    # temporal filter ---------------------------------------------------------
    if cfg['temporal_filter']['run']:
        cfg, raw = temporal_filter(cfg, raw)
    else:
        print("\nno filter ------------------------------------\n")


    # reject segments ---------------------------------------------------------
    if cfg['segment_reject']['run']:
        cfg, raw = segment_reject(cfg, raw)
    else:
        print("\nno segment rejection ------------------------------------\n")


    # plot evoked ---------------------------------------------------------
    if cfg['eval_preproc']['run'][1]:
        cfg['eval_preproc']['function'](cfg, raw)
    else:
        print("\nno evaluation ------------------------------------\n")


    # ICA ----------------------------------------------------------------
    if cfg['ICA']['run']:
        cfg, ica = fit_ica(cfg, raw)
    else:
        print("\nno ICA ------------------------------------\n")
        ica = None


    
    # %% epoch  ==========================================================================================================
    
    # create epochs ---------------------------------------------------------
    cfg, epochs = create_epoch(cfg, raw, ica)


    # reject epochs ---------------------------------------------------------
    if cfg['epoch_reject']['run']:
        cfg, epochs = reject_epoch(cfg, epochs)
    else:
        print("\nno epoch rejection ------------------------------------\n")


    # evaluate preproc ---------------------------------------------------------
    if cfg['eval_preproc']['run'][2]:
        cfg['eval_preproc']['function'](cfg, raw)
    else:
        print("\nno evaluation ------------------------------------\n")



    # %% save ==========================================================================================================


    # save preproc data ---------------------------------------------------------
    if cfg['general']['save_raw'] or cfg['general']['save_epochs']:
        save_preproc(cfg, raw, epochs)
    else:
        print("\nno save ------------------------------------\n")


    # save parameters ---------------------------------------------------------
    if cfg['general']['save_param']:
        save_params(cfg)
    else:
        print("\nno save ------------------------------------\n")


    # save report ---------------------------------------------------------
    if cfg['general']['save_report']:
        save_report(cfg, raw, raw_emptyroom, epochs, ica)
    else:
        print("\nno save ------------------------------------\n")



    print("\n\n\nDONE ---------------------------------------------------\n")



if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = ""

    run_preproc(config_path)

