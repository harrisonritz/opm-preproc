## Convert Cerca OPM data to BIDS format.
# Harrison Ritz (2025)




# %% import -------------------------------------------------------------------

import mne
import mne_bids
import os
import numpy as np
import datetime
import yaml
import sys



# %% import parameters

def set_bids_params(config_path=""):

    # set-up configuration ==========================================================================================================
    print("\n\n\nloading configuration ---------------------------------------------------\n")

    # baseline configuration (set for oddball example)
    base_config = """
    dirs:
        
    session:

    trigger:

    recording_info:
        
    """

    # Load config file
    cfg = yaml.safe_load(base_config)
   
    print(f"\n\nloading config: {config_path}\n")
    if config_path:
        with open(config_path, 'r') as stream:
            proc = yaml.safe_load(stream)
        cfg.update(proc)

    return cfg




def bids_conversion(cfg):
    # %% convert to BIDS ---------------------------------------------------------

    # loop over subjects
    # can provide list of lists in the config file, or run per-subject
    # for sx, (subj, runs, emptyroom, task) in enumerate(zip(cfg["session"]["ids"], 
    #                                                       cfg["session"]["run_prefix"], 
    #                                                       cfg["session"]["emptyroom_prefix"],  
    #                                                       cfg["session"]["tasks"])):

    subj = cfg["session"]["ids"]
    runs = cfg["session"]["run_prefix"]
    emptyroom = cfg["session"]["emptyroom_prefix"]
    task = cfg["session"]["task"]

    raw_list = list()
    print("\nparticipant: ", subj, "--------\n")
    
    # Process empty room data ------------------------------------------------
    if emptyroom:

        if emptyroom:
            fn_empty_room = os.path.join(
                cfg['dirs']['emptyroom_dir'], 
                f"sub-{subj:03}", 
                f"{emptyroom}_cMEG_Data", 
                f"{emptyroom}_meg.fif"
                )
        else:
            fn_empty_room = os.path.join(
                cfg['dirs']['data_dir'], 
                f"sub-{subj:03}", 
                f"{emptyroom}_cMEG_Data", 
                f"{emptyroom}_meg.fif"
                )

        raw_empty_room = mne.io.read_raw_fif(fn_empty_room)
        raw_empty_room.info["line_freq"] = cfg["recording_info"]["line_freq"]
        
        emptyroom_bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=cfg["session"]["session"],
            task="emptyroom",
            root=cfg['dirs']['bids_dir'],
        )
        
        mne_bids.write_raw_bids(
            raw_empty_room,
            emptyroom_bids_path,
            allow_preload=True,
            overwrite=True,
            events=None,
            format="FIF",
        )
    
    # Loop over runs for this subject -----------------------------------------
    for rr, (run) in enumerate(runs):
        print("\nrun: ", run, "--------\n")
        
        # Construct file path
        fn = os.path.join(cfg['dirs']['data_dir'], f"sub-{subj:03}", f"{run}_cMEG_Data", f"{run}_meg.fif")
        raw = mne.io.read_raw_fif(fn)
        
        
        raw.info["line_freq"] = cfg["recording_info"]["line_freq"]
        raw.info["subject_info"] = {
            "id": int(subj),
            "his_id": f"{subj:03}",
            }
        
        
        # Add events using triggers from cfg["trigger"]
        if cfg["trigger"]["find_events"]:

            print("\n\n\nFINDING EVENTS\n\n\n")
            event_list = list()
            for stim, old_trigger, new_trigger in zip(cfg["trigger"]["stim_id"][rr],
                                                    cfg["trigger"]["old_trigger_id"][rr],
                                                    cfg["trigger"]["new_trigger_id"][rr]):
    
                event = mne.find_events(raw, stim_channel=stim, min_duration=0.001)
                event_list.append(mne.merge_events(event, [int(old_trigger)], new_trigger))
            
            events = np.concatenate(event_list, axis=0)
            print("\nevents: ", events)
            print("number of events: ", len(events), "\n")
            
            annot = mne.annotations_from_events(
                events=events,
                sfreq=raw.info["sfreq"],
                event_desc=cfg["trigger"]["event_desc"]
            )
            raw.set_annotations(annot)
 
        raw_list.append(raw)


    # Concatenate raws for all runs of this subject
    all_raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")

    # Rename annotations
    if cfg["trigger"]["rename_annot"]:
        all_raw.annotations.rename(cfg["trigger"]["event_desc"]) 
    
    bids_path = mne_bids.BIDSPath(
        subject=f"{subj:03}",
        session=cfg["session"]["session"],
        task=task,
        root=cfg['dirs']['bids_dir'],
    )


    mne_bids.write_raw_bids(
        all_raw,
        bids_path,
        allow_preload=True,
        overwrite=True,
        format="FIF",
    )


# %% main ---------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print('config path: ', config_path)
    else:
        config_path = ""

    print('config path: ', config_path)
    cfg = set_bids_params(config_path)
    bids_conversion(cfg)

    print("\n\n\nDONE!\n\n\n")
