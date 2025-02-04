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


# decoding ---------------------------------------------------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from sklearn.model_selection import ShuffleSplit, cross_val_score







# add axis methods to mne classes ==========================================================================================================

def rm_axis(self, axis='Z'):
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


def pick_axis(inst, axis='Z'): mne.pick_channels_regexp(inst.info['ch_names'], regexp=rf".*{axis}$")


# Add methods to mne classes
mne.Epochs.get_axis = rm_axis
mne.io.Raw.get_axis = rm_axis
mne.Evoked.get_axis = rm_axis

mne.Epochs.misc_axis = misc_axis
mne.preprocessing.ICA.get_axis = misc_axis
mne.time_frequency.Spectrum.get_axis = misc_axis



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")





# %% functions ==========================================================================================================



def set_participant_params(param, config=""):


    if config == "":

        config = """
            participant: 2
            session: 1
            run: 1
            task: "oddball"
            datatype: "meg"
            device: "opm"
            bids_root: "/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/oddball-pilot/bids"
            known_bads: ['2E[X]','2E[Y]','2E[Z]', '2Z[X]','2Z[Y]','2Z[Z]', '29[X]','29[Y]','29[Z]']
        """

    param = yaml.safe_load(config)

    return param



def set_preproc_params(param, config=""):

    # if config == "":
    #     config = """
    #     participant
    #         - id: 2
    #         - session: 1
    #         - run: 1
    #         - task: "oddball"
    #         - datatype: "meg"
    #         - bids_root: "/Users/hr0283/Brown Dropbox/Harrison Ritz/opm_data/data/oddball-pilot/bids"
    #         - known_bads: ['2E[X]','2E[Y]','2E[Z]', '2Z[X]','2Z[Y]','2Z[Z]', '29[X]','29[Y]','29[Z]']
    #     general:
    #         - save_name: "test-preproc"
    #         - save_label: "test-preproc_"
    #         - save_preproc: True
    #         - save_param: True
    #         - save_report: True
    #         - n_jobs: -1
    #         - speed_run: False
    #     preproc:
    #         - resample: {sfreq: 150}
    #         - filter: {l_freq: 4, h_freq: 40, method: iir, iir_params: {order: 5, ftype: butter}}
    #         - bad_segments: {segment_len: 300, picks: mag, significance_level: 0.25}
    #         - bad_channels: {picks: meg, significance_level: 0.4}        
    #     """

    #     param = yaml.safe_load(config)


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


    # assessment ---------------------------------------------------------
    param["do_assess"] = [False, True, True]
    param['assess_cv'] = 10
    param['assess_plot_axes'] = ['Z']
    param['assess_save'] = False


    # channel rejection settings -----------------------------------------
    param["do_channel_reject"] = True
    param["channel_reject_method"] = "osl"
    param['channel_reject_sec'] = 5.0
    param['channel_reject_filter'] = True
    param['chanel_reject_eSSS'] = False
    param["channel_reject_plot"] = False


    # HFC settings ---------------------------------------------------------
    param['do_hfc'] = True
    param["hfc_order"] = 10
    param['hfc_apply'] = True
    param["hfc_plot"] = False


    # fitler settings -----------------------------------------
    param["do_filter"] = True
    param["filter_range"] = (.1, 30) # Hz
    param["filter_window"] = "blackman"
    param['filter_notch_spectrum'] = True

    param["filter_plot"] = False
    param["filter_plot_bands_trouble"] = {'OPM Trouble (13-16 Hz)': (13, 16)}
    
    param["filter_plot_bands"] = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
         'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
         'Gamma (30-45 Hz)': (30, 45)}
    param['filter_plot_axis'] = ['X','Y','Z']
    
    
    # segment rejection settings -----------------------------------------
    param["do_segment_reject"] = True
    param["segment_reject_thresh"] = .05
    param["segment_reject_sec"] = 1.0
    param["segment_reject_plot"] = False


    # ICA settings ---------------------------------------------------------
    param["do_ica"] = True
    param["ica_tstep"] = param["segment_reject_sec"]
    param["ica_n_components"] = 64
    param["ica_method"] = "picard"
    param["ica_params"] = {"ortho":True, "extended":True}
    param["ica_decim"] = 4

    param['ica_auto_all'] = False
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



def assess_preproc(param, raw, epoch_in=None):
    # assess preproc ==========================================================================================================
    print("\n\n\nAssessing preproc ---------------------------------------------------\n")


    if epoch_in is None:
        epochs = mne.Epochs(raw, 
                    events=None, 
                    tmin=-.200, tmax=.400, 
                    baseline=(-.200, 0),
                    preload=True,
                    proj=True,
                    decim=param['epoch_decim'],
                    ).pick('mag', exclude='bads')
    else:
        epochs = epoch_in.copy().pick('mag', exclude='bads').crop(tmin=np.maximum(-.200, epoch_in.tmin),tmax=None).apply_baseline()

    classes_all = ['standard/left', 'standard/right', 'devient/left', 'devient/right']
    classes_decode = ["standard/right", "devient/right"]
    classes_title = "right"
    epochs = epochs.equalize_event_counts(event_ids=classes_all)[0]



    # plotting ---------------------------------------------------------
    _, axes = plt.subplot_mosaic([['topo', 'decode_time'],['topo', 'decode_time']],
                                constrained_layout=True, 
                                figsize=(12, 6))



    evk_dict = dict()
    for cond in epochs.event_id.keys():
        evk_dict[cond] = epochs[cond].copy().average()
    
    mne.viz.plot_compare_evokeds(
        evk_dict,
        picks='mag',
        ci=0.95,
        colors=dict(standard=0, devient=1),
        linestyles=dict(left="solid", right="dashed"),
        time_unit="ms",
        axes=axes['topo'],
        show_sensors=False,
        show=False,
    );
    del evk_dict


    X = np.concatenate([epochs.get_data(copy=False, item=item) for item in classes_decode])
    y = np.zeros(len(X))
    y[len(epochs[classes_decode[0]].get_data()):] = 1  # set classes[1] indices to 1    
    

    # time-resolved decoding ---------------------------------------------------------
    embeder = StandardScaler()
    decoder = LinearSVC(random_state=99, dual='auto', C=.0001, max_iter=100000)
    # decoder = LogisticRegression(solver="liblinear")

    clf = make_pipeline(embeder, decoder)
    time_decod = SlidingEstimator(clf, n_jobs=param['n_jobs'], scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=ShuffleSplit(param['assess_cv'], test_size=0.2, random_state=99), n_jobs=param['n_jobs'])
    
    # Plot
    axes['decode_time'].plot(epochs.times, np.mean(scores, axis=0), label="score")
    axes['decode_time'].axhline(0.5, color="k", linestyle="--", label="chance")
    axes['decode_time'].set_xlabel("Times")
    axes['decode_time'].set_ylabel("AUC")  # Area Under the Curve
    axes['decode_time'].legend()
    axes['decode_time'].axvline(0.0, color="k", linestyle="-")
    axes['decode_time'].set_title("Sensor space decoding")
    plt.show()

    # save
    if param['assess_save']:
        np.mean(scores, axis=0).tofile(f'decode{param["participant"]}_{param['hfc_order']}.csv', sep = ',')




    # CSP topo ---------------------------------------------------------
    # # TODO: IF PLOT CSP
    # csp = CSP(n_components=5, reg=.001, log=True, norm_trace=False)
    # csp.fit(X, y)

    # # for axis in [param['assess_plot_axes']]:
    # for axis in ['X', 'Y', 'Z']:
    #     csp.plot_patterns(epochs.misc_axis(axis).info, ch_type="mag", units=f"Patterns (AU) - {axis}", size=2);
    



    # time-varying patterns ---------------------------------------------------------
    embeder = StandardScaler()
    decoder = LinearModel(decoder)
    clf = make_pipeline(embeder, decoder)
    time_decod = SlidingEstimator(clf, n_jobs=param['n_jobs'], scoring="roc_auc", verbose=True)
    time_decod.fit(X, y)

    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))

    for axis in param['assess_plot_axes']:
        evoked_time_gen.get_axis(axis).plot_joint(
            times="peaks", title=f"decode pattern - {classes_title}", **joint_kwargs
            )



    # print overall decoding
    print(f"\n\n\n\n\n\n----------decoding: {100 * scores[:,epochs.time_as_index(0)[0]:].mean():0.4f}% ----------\n")

   



    del epochs



def channel_reject(param, raw, raw_emptyroom=None):
    # channel rejection ==========================================================================================================
    
    
    print("\n\n\nChannel rejection ---------------------------------------------------\n")


    # add known bad channels
    if len(param["known_bads"]) > 0:
        print("Adding known bad channels...")
        raw.info["bads"].extend(param["known_bads"])
        print("Known bads: ", raw.info["bads"])


    ransac = False
    match param["channel_reject_method"]:

        case "osl":

            print("Detecting bad channels using OSL")

            if param["channel_reject_filter"]:

                raw_filt = raw.copy().filter(l_freq=param["filter_range"][0], h_freq=param["filter_range"][1], method='iir')

                raw_filt = detect_badchannels(raw_filt, "mag", 
                                        ref_meg=None, 
                                        significance_level=0.05, 
                                        segment_len=round(raw.info["sfreq"]*param["channel_reject_sec"]),
                                        )
                
                raw.info['bads'] = raw_filt.info['bads']
                del raw_filt

            else:
                raw = detect_badchannels(raw, "mag", 
                                        ref_meg=None, 
                                        significance_level=0.05, 
                                        segment_len=round(raw.info["sfreq"]*param["channel_reject_sec"]),
                                        )
            

        case "maxwell":

            print("Detecting bad channels using maxwell")
            start_time = time.time()

            if param['chanel_reject_eSSS']:

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

            if param["channel_reject_plot"]: 
                # Plot noisy channel scores as heatmap
                plt.figure(figsize=(10, 6))
                plt.imshow(scores['scores_noisy'], aspect='auto')
                plt.yticks(range(len(scores['ch_names'])), scores['ch_names'], ha='right')
                plt.colorbar(label='Score')
                plt.set_cmap('Reds')
                plt.clim(np.nanmin(scores["limits_noisy"]), None)
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

        raw_filt = raw.copy().pick('mag').filter(l_freq=.1, h_freq=150, method='iir')
        raw_filt.plot(block=True, scalings={"mag": 8e-12}, n_channels=32, duration=120)

        raw.info["bads"] = raw_filt.info["bads"] # transfer bads
        del raw_filt

    print("\n---------------------------------------------------\n")
    return param, raw



def hfc_proj(param,raw):
    # harmonic field correction ==========================================================================================================
    print("\n\n\nHarmonic Field Correction ---------------------------------------------------\n")


    # compute HFC
    raw_pre = raw.copy()

    # HFC
    print(f"computing HFC order {param['hfc_order']}")
    hfc_proj = mne.preprocessing.compute_proj_hfc(raw.info, order=param['hfc_order'], picks="mag")
    raw.add_proj(hfc_proj)


    # apply HFC    
    if param['hfc_apply']:
        raw.apply_proj(verbose="error")
        print("applied HFC")
    else:
        print("HFC not applied")



    


    # plot HFC
    if param["hfc_plot"]:


        # raw.plot(block=True)


        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 16))
        
        # Plot PSD before HFC
        psd_orig = raw_pre.compute_psd(fmin=0, 
                                       fmax=2*param["line_freq"], 
                                       picks="mag",
                                       n_fft=2000)
        psd_orig.plot(
            axes=ax1,
            picks="mag",
            show=False)
        ax1.set_title('PSD before HFC')

        
        # Plot PSD after HFC
        psd_hfc = raw.copy().apply_proj(verbose="error").compute_psd(fmin=0, 
                                                                     fmax=2*param["line_freq"], 
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
    return param, raw



def temporal_filter(param, raw):
    # resample & filter ==========================================================================================================
    print("\n\n\nTemporal Filter ---------------------------------------------------\n")


    # plot before filter
    if param["filter_plot"]:
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
    if param['filter_notch_spectrum']:

        print('\n\nnotch filter: spectrum fit ----------\n')
        raw.notch_filter(freqs=None, 
                         method='spectrum_fit', 
                         filter_length='10s',
                         n_jobs=param['n_jobs'],
                         )
        
    else:

        print('\n\nnotch filter: traditional method ----------\n')
        raw.notch_filter(param["line_freq"])
        if (2*param["line_freq"]) <  (param["filter_range"][1]+10):
            for ff in range(2, int(1+np.ceil((param["filter_range"][1] + 10) / param["line_freq"]))):
                print(f"\n\nnotch filter: {param["line_freq"]*ff} Hz ----------\n")
                raw.notch_filter(param["line_freq"]*ff)


    # seperately high-pass filter then low-pass filter ---------------------------------------------------------
    raw.filter(l_freq=param["filter_range"][0], 
               h_freq=None, 
               fir_window=param["filter_window"],
               ).filter(l_freq=None, 
               h_freq=param["filter_range"][1], 
               fir_window=param["filter_window"],
               )


    # plot after filter
    if param["filter_plot"]:

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
        for axis in param['filter_plot_axis']:

            spec_filt.get_axis(axis).plot_topo(show=True)



            spec_filt.get_axis(axis).plot_topomap(bands=param["filter_plot_bands_trouble"], 
                                                            ch_type='mag', 
                                                            normalize=True, 
                                                            show_names=True,
                                                            show=True)


            spec_filt.get_axis(axis).plot_topomap(bands=param["filter_plot_bands"], 
                                                  ch_type='mag', 
                                                  normalize=True, 
                                                  show_names=True,
                                                  show=True)
        
        del spec_filt

        


    # mop up memeory leak
    gc.collect()

    print("\n---------------------------------------------------\n")
    return param, raw



def segment_reject(param,raw,metric='std'):
    # reject continious segments ==========================================================================================================
    print('\n\nsegment rejection ---------------------------------------------------\n')


    if metric=='kurtosis':

        raw = detect_badsegments(
            raw,
            picks='mag',
            detect_zeros=False,
            segment_len=round(raw.info["sfreq"]*1.0),
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
                segment_len=round(raw.info["sfreq"]*param["segment_reject_sec"]),
                channel_threshold=param["segment_reject_thresh"],
                significance_level=param["segment_reject_thresh"],
                )
        
        # raw = detect_badsegments(
        #         raw,
        #         picks="mag",
        #         ref_meg=False,
        #         metric="std",
        #         mode="diff",
        #         detect_zeros=False,
        #         channel_wise=False,
        #         segment_len=round(raw.info["sfreq"]*param["segment_reject_sec"]),
        #         channel_threshold=param["segment_reject_thresh"],
        #         significance_level=param["segment_reject_thresh"],
        #         )
    
    if param["segment_reject_plot"]:
        raw.plot(block=True)


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
                                    max_iter=1000,
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




            
    # auto-select
    if param['ica_auto_all']:
        print('\nfind bad muscles components ---- \n')
        ica.exclude.extend(ica.find_bads_muscle(raw_ica)[0])
        print('\nfind bad ECG components ---- \n')
        ica.exclude.extend(ica.find_bads_ecg(raw_ica)[0])


    print(f"\nICA exclude: {ica.exclude} ----------\n")


    # plot ICA ---------------------------------------------------------

    # plot ICA components
    for axis in param["ica_plot_axes"]:
        
        # plot all components
        plot_ica_axis(ica, raw_ica, axis=axis)

       

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


def run_preproc(participant_config="", preproc_config=""):

    # %% init ==========================================================================================================

    # set params ---------------------------------------------------------
    param = dict()
    param = set_participant_params(param, participant_config)
    param = set_preproc_params(param, preproc_config)
    param = make_paths(param)


    # load data ---------------------------------------------------------
    param, raw, raw_emptyroom = read_data(param)


    # initial fit ---------------------------------------------------------
    if param["do_assess"][0]:
        assess_preproc(param, raw)
    else:
        print("\nno assessment ------------------------------------\n")



    # %% artifact rejection ==========================================================================================================
    

    # reject segments ---------------------------------------------------------
    if param['do_segment_reject']:
        param, raw = segment_reject(param, raw, metric='kurtosis')
    else:
        print("\nno segment rejection ------------------------------------\n")



    # channel rejection ---------------------------------------------------------
    if param['do_channel_reject']:
        param, raw = channel_reject(param, raw, raw_emptyroom=raw_emptyroom)
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
        param, raw = segment_reject(param, raw)
    else:
        print("\nno segment rejection ------------------------------------\n")


    # plot evoked ---------------------------------------------------------
    if param["do_assess"][1]:
        assess_preproc(param, raw)
    else:
        print("\nno assessment ------------------------------------\n")


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


    # plot evoked ---------------------------------------------------------
    if param["do_assess"][2]:
        assess_preproc(param, raw, epoch_in=epochs)
    else:
        print("\nno assessment ------------------------------------\n")



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

    