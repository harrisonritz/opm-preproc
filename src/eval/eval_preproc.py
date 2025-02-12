## custom evaluation function


import numpy as np
import mne
import matplotlib.pyplot as plt



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



def eval_oddball(cfg, raw, epoch_in=None):
    
    
    
    
    # evaluate oddball ==========================================================================================================
    print("\n\n\nEvaluate oddball ---------------------------------------------------\n")


    if epoch_in is None:
        epochs = mne.Epochs(raw, 
                    events=None, 
                    tmin=-.200, tmax=.400, 
                    baseline=(-.200, 0),
                    preload=True,
                    proj=True,
                    decim=cfg['epoch']['decim'],
                    ).pick('mag', exclude='bads')
    else:
        epochs = epoch_in.copy().pick('mag', exclude='bads').crop(tmin=np.maximum(-.200, epoch_in.tmin),tmax=None).apply_baseline()

    classes_all = ['standard/left', 'standard/right', 'deviant/left', 'deviant/right']
    classes_decode = ['standard/right', 'deviant/right']
    classes_title = 'right'
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
        colors=dict(standard=0, deviant=1),
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
    time_decod = SlidingEstimator(clf, n_jobs=cfg['general']['n_jobs'], scoring="roc_auc", verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=ShuffleSplit(cfg['eval_preproc']['cv'], test_size=0.2, random_state=99), n_jobs=cfg['general']['n_jobs'])
    
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
    if cfg['eval_preproc']['save']:
        np.mean(scores, axis=0).tofile(f"decode{cfg['participant']['id']}_{cfg['HFC']['order']}.csv", sep = ',')


    # time-varying patterns ---------------------------------------------------------
    embeder = StandardScaler()
    decoder = LinearModel(decoder)
    clf = make_pipeline(embeder, decoder)
    time_decod = SlidingEstimator(clf, n_jobs=cfg['general']['n_jobs'], scoring="roc_auc", verbose=True)
    time_decod.fit(X, y)

    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))

    for axis in cfg['eval_preproc']['plot_axes']:
        evoked_time_gen.plot_joint(
            times="peaks", 
            title=f"decode pattern - {classes_title}", 
            picks=pick_axis(evoked_time_gen, axis, cfg['info']['sensor_wildcard']),
            **joint_kwargs,
            )



    # print overall decoding
    print(f"\n\n\n\n\n\n----------decoding: {100 * scores[:,epochs.time_as_index(0)[0]:].mean():0.4f}% ----------\n")

   



    del epochs