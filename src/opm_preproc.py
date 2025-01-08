# Preprocess OPM data using MNE-Python & OSL
# Harrison Ritz (2025)



# %% import packages ==========================================================================================================


# basic imports ---------------------------------------------------------
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import json
import gc
import psutil


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
from osl.osl_wrappers import (
    detect_badchannels, 
    detect_badsegments,
    drop_bad_epochs,
)

from plot.plot_ica_axis import plot_ica_axis






# add axis-selection methods to mne classes ==========================================================================================================

def pick_axis(self, axis='Z'):
    """Pick MEG sensors along specified axis.

    Selects MEG sensors oriented along a given axis using regular expressions matching
    on channel names. Returns a copy of the data with only the selected channels.

    Parameters
    ----------
    axis : str, default="Z"
        The axis to select sensors from. Must be one of: "X", "Y", or "Z".
        Case-sensitive.

    Returns
    -------
    instance
        A new instance containing only the channels matching the specified axis.
        Original instance remains unmodified.

    Notes
    -----
    - Channel names must follow the format: 'XX[axis]' where XX are any characters
      and axis is the orientation (X, Y, or Z)
    - Creates a copy to preserve the original data

    Harrison Ritz, 2025
    """
   
    axis_picks = mne.pick_channels_regexp(self.info["ch_names"], f'^..\\[{axis}]$')
    return self.copy().pick([self.ch_names[i] for i in axis_picks])



def misc_axis(self, axis='Z'):
    """Pick sensors axis by setting non-axis channels to 'misc'.

    Creates a copy of the data where channels not matching the specified axis 
    pattern are set to 'misc' type, while preserving the original axis channels.

    Parameters
    ----------
    axis : str
        Sensor axis to keep, must be 'X', 'Y' or 'Z' (default 'Z')

    Returns
    -------
    mne.io.Raw
        A copy of the data with non-axis channels set to 'misc' type

    Notes
    -----
    - Channel names must follow the pattern 'XX[Y]' where Y is the axis
    - Creates a copy to preserve the original data

    Harrison Ritz, 2025
    """
   

    axis_picks = mne.pick_channels_regexp(self.info["ch_names"], f'^..\\[{axis}]$')

    # Create a mask for non-matching columns
    non_axis_picks = np.ones(self.info['nchan'], dtype=bool)
    non_axis_picks[axis_picks] = False

    # Change channel type to 'misc' for non-axis channels
    self_copy = self.copy()
    for i, is_non_axis in enumerate(non_axis_picks):
        if is_non_axis:
            self_copy.info['chs'][i]['kind'] = mne.io.constants.FIFF.FIFFV_MISC_CH


    return self_copy


# Add methods to mne classes
mne.Epochs.get_axis = pick_axis
mne.io.Raw.get_axis = pick_axis
mne.Evoked.get_axis = pick_axis

mne.Epochs.misc_axis = misc_axis
mne.preprocessing.ICA.get_axis = misc_axis
mne.time_frequency.Spectrum.get_axis = misc_axis



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")






# %% functions ==========================================================================================================


def set_participant_params(param, args=None):

    # participant settings ---------------------------------------------------------
    param["participant"] = 1


    param["session"] = 1
    param["run"] = 1
    param["task"] = "task"
    param["datatype"] = "meg"
    param["device"] = 'opm'


    param["bids_root"] = 'MY_BIDS_ROOT'

    param["known_bads"] = [
        [],
        ]

    return param



def set_preproc_params(param, args=None):

    # general settings ---------------------------------------------------------
    param["save_name"] = "test-preproc"
    param["save_label"] = f'{param["save_name"]}_'

    param["save_preproc"] = True
    param["save_param"] = True
    param["save_report"] = True

    param["n_jobs"] = -1

    param["speed_run"] = False
    print("speed run: ", param["speed_run"])



    # intial plot ---------------------------------------------------------
    param["load_plot"] = False


    # channel rejection settings -----------------------------------------
    param["do_channel_reject"] = True
    param["channel_reject_method"] = "osl"
    param["channel_reject_plot"] = True
    param['channel_reject_sec'] = 5.0


    # HFC settings ---------------------------------------------------------
    param['do_hfc'] = True
    param['hfc_whiten'] = False
    param["hfc_order"] = 10
    param["hfc_plot"] = False


    # fitler settings -----------------------------------------
    param["do_filter"] = True
    param["filter_range"] = (1, 30) # Hz
    param["filter_window"] = "blackman"
    param['filter_notch_spectrum'] = True

    param["filter_plot"] = True
    param["filter_plot_bands"] = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
         'Gamma (30-45 Hz)': (30, 45), 'OPM Trouble (14-17 Hz)': (14, 17)}
    param['filter_plot_axis'] = ['Z']
    
    



    # segment rejection settings -----------------------------------------
    param["do_segment_reject"] = True
    param["segment_reject_thresh"] = .05
    param["segment_reject_sec"] = 5.0
    param["segment_reject_plot"] = False


    # SSP settings ---------------------------------------------------------
    param["do_ssp"] = False
    param["ssp_dims"] = 3


    # ICA settings ---------------------------------------------------------
    param["do_ica"] = True
    param["ica_tstep"] = param["segment_reject_sec"]
    param["ica_n_components"] = .99
    param["ica_method"] = "picard"
    param["ica_params"] = {"ortho":True, "extended":True}
    param["ica_decim"] = 4

    param['ica_auto_all'] = True
    param["ica_plot_axes"] = ['Z']

    param["ica_apply"] = True
    param["ica_save"] = False

    # epoch settings ---------------------------------------------------------
    param["epoch_tmin"] = -0.5
    param["epoch_tmax"] = 0.5
    param['epoch_plot'] = False
    param['epoch_decim'] = 2


    # epoch reject settings ---------------------------------------------------------
    param["do_epoch_reject"] = True
    param["epoch_reject_method"] = 'osl'
    param["epoch_reject_ar-interp"] = [0,1,2,3]
    param["epoch_reject_plot"] = True


    # speed run settings ---------------------------------------------------------
    if param["speed_run"]:
        
        print('\nSPEED RUN ========================================== \n')
        print("I want you to run as fast as you can")
        print("As fast as I can?")
        print("As fast as you can\n")

        param["load_plot"] = False
        param["hfc_plot"] = False
        param["filter_plot"] = False
        param["epoch_plot"] = False
        param["epoch_reject_plot"] = False

        param["ica_n_components"] = 8
        param["ica_save"] = False
        param["ica_apply"] = True
        param['ica_auto_all'] = False
        param['ica_decim'] = 6

    


    print("\n CONFIGURATION ========================================== \n")
    print(param)
    print(param["save_label"])

    return param



def make_paths(param):
    # make paths ==========================================================================================================
    print("\n\n\nMaking paths ---------------------------------------------------\n")


    param["bids_path"] = BIDSPath(
        subject = f"{param["participant"]:03}", 
        session = f"{param["session"]:02}", 
        task = param["task"],
        run = f"{param["run"]:02}",
        datatype = param["datatype"], 
        root = param["bids_root"]
    )

    param["emptyroom_path"] = BIDSPath(
        subject = f"{param["participant"]:03}", 
        session = f"{param["session"]:02}", 
        task = 'emptyroom',
        run = f"{param["run"]:02}",
        datatype = param["datatype"], 
        root = param["bids_root"]
    )


    # Create directories for ICA and preproc files
    ica_dir = os.path.join(param["bids_root"], "derivatives", "ICA")
    os.makedirs(ica_dir, exist_ok=True)
    param["ica_fname"] = os.path.join(ica_dir, f"{param["bids_path"].basename}-ica.fif")


    # Create directories for preproc & parameter files
    preproc_dir = os.path.join(param["bids_root"], "derivatives", "preproc")
    os.makedirs(preproc_dir, exist_ok=True)
    param["preproc_fname"] = os.path.join(preproc_dir, f"{param["bids_path"].basename}_{param["save_label"]}_preproc_epo.fif")
    param["param_fname"] = os.path.join(preproc_dir, f"{param["bids_path"].basename}_{param["save_label"]}_preproc_params.json")


    # Create directories for ICA and preproc files
    report_dir = os.path.join(param["bids_root"], "derivatives", "report")
    os.makedirs(report_dir, exist_ok=True)
    param["report_fname"] = os.path.join(report_dir, f"{param["bids_path"].basename}_{param["save_label"]}_preproc_report.html")



    print("\n---------------------------------------------------\n")
    return param



def read_data(param):
    # read-in data ==========================================================================================================
    print("\n\n\nReading in data ---------------------------------------------------\n")
    print(f"Participant: {param['participant']}, Session: {param['session']}, Run: {param['run']}, Task: {param['task']}, Datatype: {param['datatype']}")
    print(f"BIDS path: {param["bids_path"]}")


    # Read in raw file
    raw = read_raw_bids(
        bids_path=param["bids_path"], 
        extra_params=dict(preload=True))


    # Read in emptyroom file
    raw_emptyroom = read_raw_bids(
        bids_path=param["emptyroom_path"], 
        extra_params=dict(preload=True))



    # plot PSD
    if param["load_plot"]:
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
    param["sample_rate"] = raw.info["sfreq"]
    param["line_freq"] = raw.info["line_freq"]
    if not param["line_freq"]:
        param["line_freq"] = 60.0
    print(f"Sampling rate: {param["sample_rate"]} Hz")


    print("\n---------------------------------------------------\n")
    return param, raw, raw_emptyroom



def channel_reject(param,raw):
    # channel rejection ==========================================================================================================
    
    
    print("\n\n\nChannel rejection ---------------------------------------------------\n")


    # add known bad channels
    if len(param["known_bads"][param["participant"]-1]) > 0:
        print("Adding known bad channels...")
        raw.info["bads"].extend(param["known_bads"][param["participant"]-1])
        print("Known bads: ", raw.info["bads"])


    ransac = False
    match param["channel_reject_method"]:

        case "osl":
            print("Detecting bad channels using OSL")

            raw = detect_badchannels(raw, "mag", 
                                     ref_meg=None, 
                                     significance_level=0.05, 
                                     segment_len=round(raw.info["sfreq"]*param["channel_reject_sec"]),
                                     )
            

        case "manual":
            print("Manually rejecting channels")

            raw.plot(block=True)


        case "ransac":
            print("Detecting bad channels using RANSAC")

            # Create epochs for RANSAC
            epochs_ransac = mne.Epochs(raw, 
                                    events=None, 
                                    tmin=param["epoch_tmin"], tmax=param["epoch_tmax"], 
                                    baseline=None, 
                                    preload=True)

            # Fit RANSAC
            ransac = Ransac(verbose=True, picks='mag', n_jobs=param["n_jobs"], random_state=99)
            ransac = ransac.fit(epochs_ransac)

            # Apply RANSAC
            raw.info['bads'].extend(ransac.bad_chs_)
            del epochs_ransac
            del ransac


        case "None":
            print("No channel rejection method specified. Skipping channel rejection.")

        case _:
            raise Exception("channel reject not recognized")

    print(f"identified {len(raw.info["bads"])} bad channels...")
    print('bads: ', raw.info['bads'])


    if param["channel_reject_plot"]:

        raw_filt = raw.copy().pick('mag').filter(l_freq=.1, h_freq=150, method='iir') # plot filtered channels
        raw_filt.plot(block=True, scalings={"mag": 8e-12}, n_channels=32, duration=120)

        raw.info["bads"] = raw_filt.info["bads"] # transfer bads
        del raw_filt

    print("\n---------------------------------------------------\n")
    return param, raw



def hfc_proj(param,raw):
    # harmonic field correction ==========================================================================================================
    print("\n\n\nHarmonic Field Correction ---------------------------------------------------\n")


    # compute HFC
    hfc_proj = mne.preprocessing.compute_proj_hfc(raw.info, order=param["hfc_order"], picks="mag")
    raw.add_proj(hfc_proj)


    # plot HFC
    if param["hfc_plot"]:

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot PSD before HFC
        raw.compute_psd(fmin=0.1, fmax=param["line_freq"]+10,picks="mag").plot(
                                amplitude=True,
                                axes=ax1,
                                picks="mag",
                                show=False)
        ax1.set_title('PSD before HFC')

        
        # Plot PSD after HFC
        raw.copy().apply_proj(verbose="error").compute_psd(fmin=0.1, fmax=param["line_freq"]+10,picks="mag").plot(
                        amplitude=True,
                        axes=ax2,
                        picks="mag",
                        show=False)
        ax2.set_title('PSD after HFC')

        # Set ylims to be the same
        ylim1 = ax1.get_ylim()
        ylim2 = ax2.get_ylim()
        ax1.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))
        ax2.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))
        plt.show()


    # apply HFC    
    raw.apply_proj(verbose="error")
    print("applied HFC")


    print("\n---------------------------------------------------\n")
    return param, raw



def temporal_filter(param, raw):
    # resample & filter ==========================================================================================================
    print("\n\n\n Filter & Resample ---------------------------------------------------\n")


    # plot before filter
    if param["filter_plot"]:
        _, axs = plt.subplots(2, 1, figsize=(10, 8))
        raw.compute_psd(fmin=0.01, 
                        fmax=240, 
                        picks="mag",
                        ).plot(
                            picks="mag",
                            xscale='log',
                            axes = axs[0])



    # Notch Filter ---------------------------------------------------------
    if param['filter_notch_spectrum']:

        print('\n\nnotch filter: spectrum fit ----------\n')
        raw.notch_filter(freqs=None, 
                         method='spectrum_fit', 
                         filter_length='10s',
                         n_jobs=param['n_jobs'],
                         )
        
    else:

        print('\n\nnotch filter: traditional fit ----------\n')
        raw.notch_filter(param["line_freq"])
        if (2*param["line_freq"]) <  (param["filter_range"][1]+10):
            for ff in range(2, int(1+np.ceil((param["filter_range"][1] + 10) / param["line_freq"]))):
                print(f"\n\nnotch filter: {param["line_freq"]*ff} Hz ----------\n")
                raw.notch_filter(param["line_freq"]*ff)



    # high-pass filter then low-pass filter ---------------------------------------------------------
    raw.filter(l_freq=param["filter_range"][0], 
               h_freq=None, 
               fir_window=param["filter_window"],
               ).filter(l_freq=None, 
               h_freq=param["filter_range"][1], 
               fir_window=param["filter_window"],
               )



    # plot after filter
    if param["filter_plot"]:

        spec_filt = raw.compute_psd(fmin=0.01, 
                                    fmax=240, 
                                    picks="mag",
                                    n_jobs=param['n_jobs'])
        
        # plot PSD after filter
        spec_filt.plot(
            picks="mag",
            xscale='log',
            axes = axs[1])
        plt.show()

        # plot filtered topomaps
        for axis in param['filter_plot_axis']:
            spec_filt.get_axis(axis).plot_topomap(bands=param["filter_plot_bands"], 
                                                  ch_type='mag', 
                                                  normalize=True, 
                                                  show=True)
        
        del spec_filt

        

    # mop up memeory leak
    gc.collect()

    print("\n---------------------------------------------------\n")
    return param, raw



def segment_reject(param,raw):
    # reject continious segments ==========================================================================================================
    print('\n\nsegment rejection ---------------------------------------------------\n')


    raw = detect_badsegments(
            raw,
            picks="mag",
            ref_meg=False,
            metric="std",
            detect_zeros=False,
            channel_wise=True,
            segment_len=round(raw.info["sfreq"]*param["segment_reject_sec"]),
            channel_threshold=param["segment_reject_thresh"]
            )
    
    raw = detect_badsegments(
            raw,
            picks="mag",
            ref_meg=False,
            metric="std",
            mode="diff",
            detect_zeros=False,
            channel_wise=True,
            segment_len=round(raw.info["sfreq"]*param["segment_reject_sec"]),
            channel_threshold=param["segment_reject_thresh"]
            )
    
    if param["segment_reject_plot"]:
        raw.plot(block=True)


    print("\n---------------------------------------------------\n")
    return param, raw



def fit_emptyroom_SSP(param, raw, raw_emptyroom):
    # empty room SSP ==========================================================================================================
    print('\n\nempty room SSP ---------------------------------------------------\n')


    # fit SSP ---------------------------------------------------------
    raw_emptyroom.info['bads'] = raw.info['bads']
    empty_room_projs = mne.compute_proj_raw(raw_emptyroom.copy().apply_proj(), n_mag=param['ssp_dims'], n_jobs=param["n_jobs"])

    # plot SSP projections ---------------------------------------------------------
    _, axs = plt.subplots(3, 3, figsize=(9, 9))
    for aa, axis in enumerate(['X', 'Y', 'Z']):
        proj_copy = copy.deepcopy(empty_room_projs)
        axis_picks = np.array(
            mne.pick_channels_regexp(proj_copy[0]['data']['col_names'], f'^..\\[{axis}]$')
        )

        for pp in range(3):
            proj_data = proj_copy[pp]['data']
            proj_data['data'] = proj_data['data'][:, axis_picks]
            proj_data['col_names'] = [proj_data['col_names'][i] for i in axis_picks]
            proj_data['ncol'] = len(axis_picks)
            
        raw_copy = raw_emptyroom.copy().pick(picks="meg", exclude=raw_emptyroom.info["bads"])
        raw_copy.drop_channels([ch for i, ch in enumerate(raw_copy.ch_names) if i not in axis_picks])

        mne.viz.plot_projs_topomap(
            proj_copy,
            colorbar=True,
            vlim=(-0.2, 0.2),
            info=raw_copy.info,
            size=2,
            axes=axs[aa, :],
            show=False
        )
        axs[aa, 0].set_ylabel(f"{axis} (T)")

    plt.show()


    # plot SSP effects ---------------------------------------------------------
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
    # Plot PSD before SSP
    raw.compute_psd(fmin=0.1, fmax=param["filter_range"][1]+10,picks="mag").plot(
                            amplitude=True,
                            axes=ax1,
                            picks="mag",
                            show=False)
    ax1.set_title('PSD before SSP')

    
    # Plot PSD after SSP
    raw.copy().apply_proj(verbose="error").compute_psd(fmin=0.1, fmax=param["filter_range"][1]+10,picks="mag").plot(
                    amplitude=True,
                    axes=ax2,
                    picks="mag",
                    show=False)
    ax2.set_title('PSD after SSP')

    # Set ylims to be the same
    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()
    ax1.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))
    ax2.set_ylim(bottom=min(ylim1[0], ylim2[0]),top=max(ylim1[1], ylim2[1]))
    plt.show()



    # apply SSP ---------------------------------------------------------
    raw.add_proj(empty_room_projs)
    print(raw.info)


    print("\n---------------------------------------------------\n")

    return param, raw



def fit_ica(param, raw):
    # ICA ==========================================================================================================
    print("\n\n\nICA ---------------------------------------------------\n")


    if os.path.isfile(param["ica_fname"]):

        print(f"loading ICA from {param["ica_fname"]}")
        ica = mne.preprocessing.read_ica(param["ica_fname"])

    else:

        print("Fitting ICA...")
        if param["filter_range"][0] < 1.0:
            raw_ica = raw.copy().filter(l_freq=1, h_freq=None, fir_window=param["filter_window"]).pick(picks="meg", exclude=raw.info["bads"])
        else:
            raw_ica = raw.copy().pick(picks="meg", exclude=raw.info["bads"])
        

        ica = mne.preprocessing.ICA(n_components=param["ica_n_components"], 
                                    max_iter="auto", 
                                    random_state=99, 
                                    method=param["ica_method"],
                                    fit_params=param["ica_params"],
                                    )
        

        # fit ICA ---------------------------------------------------------
        ica.fit(raw_ica, 
                decim=param['ica_decim'],
                tstep=param["ica_tstep"],
                reject_by_annotation=True,
                )
        
        var_explained = ica.get_explained_variance_ratio(raw_ica, ch_type='mag')
        
        print('\nICA info ----------\n', ica, '\n', ica.info, '\n')
        print(f"varience expalined: {100*var_explained['mag']:0.2f}%")
        print(f"\nICA fit complete ----------\n\n")



    # auto-select bad components ---------------------------------------------------------
    if param['ica_auto_all']:
        print('\nfind bad muscles components ---- \n')
        ica.exclude.extend(ica.find_bads_muscle(raw_ica)[0])
        print('\nfind bad ECG components ---- \n')
        ica.exclude.extend(ica.find_bads_ecg(raw_ica)[0])


    print(f"\nICA exclude: {ica.exclude} ----------\n")


    # plot ICA ---------------------------------------------------------

    # plot ICA components
    for axis in param["ica_plot_axes"]:
        
        # plot ICA components spatially
        plot_ica_axis(ica, raw_ica, axis=axis)

       
    # plot ICA components temporally
    ica.plot_sources(raw_ica, block=True)
    del raw_ica

    if param["ica_save"]:
        print(f"saving ICA to {param["ica_fname"]}")
        ica.save(param["ica_fname"])
    else:
        print("not saving ICA")


    print("\n---------------------------------------------------\n")

    return param, ica



def create_epoch(param, raw, ica):
    # create standard & ICA epochs ==========================================================================================================
    print("\n\n\nEpoch ---------------------------------------------------\n")


    epochs = mne.Epochs(raw, 
                events=None, 
                tmin=param["epoch_tmin"], tmax=param["epoch_tmax"], 
                baseline=None, # don't baseline before ICA
                preload=True,
                decim=param['epoch_decim'],
                )
   
    if ica is not None:
        ica.plot_overlay(epochs.average(), exclude=ica.exclude, picks="mag")
        ica.apply(epochs, exclude=ica.exclude)


    print('\nEpoch info ----------\n', epochs, '\n', epochs.info, '\n')
    print("\n---------------------------------------------------\n")
    return param, epochs



def reject_epoch(param, epochs):

    # Epoch rejection 1 ==========================================================================================================
    print("\n\n\nEpoch rejection ---------------------------------------------------\n")


    # detect bad epochs
    match param["epoch_reject_method"]:

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

            ar = AutoReject(n_interpolate=param["epoch_reject_ar-interp"], random_state=99, 
                            picks="mag",
                            n_jobs=param['n_jobs'], verbose=True)
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
    return param, epochs



def save_preproc(param, epochs):
    # save preproc data ==========================================================================================================
    print("\n\n\nSaving preproc data ---------------------------------------------------\n")

    print(f"saving preproc data to {param["preproc_fname"]}")
    epochs.save(param["preproc_fname"], overwrite=True)


   
    print("\n---------------------------------------------------\n")



def save_params(param):
    # save preproc data ==========================================================================================================
    print("\n\n\nSaving fitting parameters ---------------------------------------------------\n")

    # Save parameters to json
    print(f"saving parameters to {param["param_fname"]}")

    # Convert any non-serializable objects to strings
    param_save = param.copy()
    for key, value in param_save.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            param_save[key] = str(value)

    # Save to json
    with open(param["param_fname"], 'w') as f:
        json.dump(param_save, f, indent=4)


    print("\n---------------------------------------------------\n")



def save_report(param, raw, raw_emptyroom, epochs, ica):
    # save report ==========================================================================================================
    print("\n\n\nSaving report ---------------------------------------------------\n")

    report = mne.Report(verbose=True,
                        info_fname=param["preproc_fname"],
                        subject=f"{param["participant"]:03}",
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
    report.add_ica(ica.get_axis("Z"),
                    title="ICA (Z)",
                    inst=epochs,
                    n_jobs=param["n_jobs"],
                    )
    
    # epochs
    report.add_epochs(epochs,
                      title="Epochs",
                      )


    # save report
    print(f"saving report to {param["report_fname"]}")
    report.save(param["report_fname"], overwrite=True)


                      



# %% run preproc ==========================================================================================================


def run_preproc():

    # %% init ==========================================================================================================

    # set params ---------------------------------------------------------
    param = dict()
    param = set_participant_params(param)
    param = set_preproc_params(param)
    param = make_paths(param)


    # load data ---------------------------------------------------------
    param, raw, raw_emptyroom = read_data(param)



    # %% artifact rejection ==========================================================================================================
    
    # channel rejection ---------------------------------------------------------
    if param['do_channel_reject']:
        param, raw = channel_reject(param,raw)
    else:
        print("\nno channel rejection ------------------------------------\n")


    # harmonic field correction ---------------------------------------------------------
    if param['do_hfc']:
        param, raw = hfc_proj(param, raw)
    else:
        print("\nno HFC ------------------------------------\n")


    # temporal filter ---------------------------------------------------------
    if param['do_filter']:
        param, raw = temporal_filter(param, raw)
    else:
        print("\nno filter ------------------------------------\n")


    # reject segments ---------------------------------------------------------
    if param['do_segment_reject']:
        param, raw = segment_reject(param,raw)
    else:
        print("\nno segment rejection ------------------------------------\n")


    # SSP ---------------------------------------------------------
    if param["do_ssp"]:
        param, raw = fit_emptyroom_SSP(param, raw, raw_emptyroom)
    else:
        print("\nno SSP ------------------------------------\n")

    
    # ICA ----------------------------------------------------------------
    if param["do_ica"]:
        param, ica = fit_ica(param, raw)
    else:
        print("\nno ICA ------------------------------------\n")
        ica = None


    
    # %% epoch  ==========================================================================================================
    
    # create epochs ---------------------------------------------------------
    param, epochs = create_epoch(param, raw, ica)


    # reject epochs ---------------------------------------------------------
    if param["do_epoch_reject"]:
        param, epochs = reject_epoch(param, epochs)
    else:
        print("\nno epoch rejection ------------------------------------\n")



    # %% save ==========================================================================================================


    # save preproc data ---------------------------------------------------------
    if param["save_preproc"]:
        save_preproc(param, epochs)
    else:
        print("\nno save ------------------------------------\n")


    # save parameters ---------------------------------------------------------
    if param["save_param"]:
        save_params(param)
    else:
        print("\nno save ------------------------------------\n")


    # save report ---------------------------------------------------------
    if param["save_report"]:
        save_report(param, raw, raw_emptyroom, epochs, ica)
    else:
        print("\nno save ------------------------------------\n")



    print("\n\n\nDONE ---------------------------------------------------\n")




if __name__ == "__main__":
    run_preproc()

    