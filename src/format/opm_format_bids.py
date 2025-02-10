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

def set_bids_params(S, config_path=""):

    # set-up configuration ==========================================================================================================
    print("\n\n\nloading configuration ---------------------------------------------------\n")

    # baseline configuration (set for oddball example)
    base_config = """
    dirs:
        data_dir: "/Users/hr0283/Projects/opm-preproc/examples/oddball/raw"         # UPDATE THIS PATH
        bids_dir: "/Users/hr0283/Projects/opm-preproc/examples/oddball/bids"        # UPDATE THIS PATH
        emptyroom_dir: "/Users/hr0283/Projects/opm-preproc/examples/oddball/raw"    # UPDATE THIS PATH
    
    session:
        ids: [2]                                         # list of subjects
        experiment: "oddball_pilot"
        session: "01"
        run_prefix: [["20241218_153301", "20241218_155254"]]
        emptyroom_prefix: ["20241218_160936"]  # corrected typo in key name
        tasks: ["oddball", "oddball"]

    trigger:
        stim_id: ["Trigger1[Z]", "Trigger2[Z]"]
        old_trigger_id: [3, 3]
        new_trigger_id: [1, 2]
        event_desc:
            1: "standard/left"
            2: "deviant/left"
            3: "standard/right"
            4: "deviant/right"
            900: "BAD boundary"
            901: "EDGE boundary"
        event_id:
            "standard/left": 1
            "deviant/left": 2
            "standard/right": 3
            "deviant/right": 4
            "BAD boundary": 900
            "EDGE boundary": 901

    recording_info:
        line_freq: 60.0
    """

    # Load config file
    S = yaml.safe_load(base_config)
    if config_path:
        print(f"\n\nloading config: {config_path}\n")
        with open(config_path, 'r') as stream:
            proc = yaml.safe_load(stream)
        S.update(proc)
        S['participant']['config_path'] = config_path

    return S




def bids_conversion(S):
    # %% convert to BIDS ---------------------------------------------------------

    # loop over subjects
    # can provide list of lists in the config file, or run per-subject
    for sx, (subj, runs, emptyroom, task) in enumerate(zip(S["session"]["ids"], 
                                                          S["session"]["run_prefix"], 
                                                          S["session"]["emptyroom_prefix"],  
                                                          S["session"]["tasks"])):
        
        raw_list = list()
        print("\nparticipant: ", subj, "--------\n")
        
        # Process empty room data ------------------------------------------------
        if emptyroom:

            if emptyroom:
                fn_empty_room = os.path.join(
                    S['dirs']['emptyroom_dir'], 
                    f"sub-{subj:03}", 
                    f"{emptyroom}_cMEG_Data", 
                    f"{emptyroom}_meg.fif"
                    )
            else:
                fn_empty_room = os.path.join(
                    S['dirs']['data_dir'], 
                    f"sub-{subj:03}", 
                    f"{emptyroom}_cMEG_Data", 
                    f"{emptyroom}_meg.fif"
                    )

            raw_empty_room = mne.io.read_raw_fif(fn_empty_room)
            raw_empty_room.info["line_freq"] = S["recording_info"]["line_freq"]
            
            emptyroom_bids_path = mne_bids.BIDSPath(
                subject=f"{subj:03}",
                session=S["session"]["session"],
                task="emptyroom",
                run="01",
                root=S['dirs']['bids_dir'],
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
        for run in runs:
            print("\nrun: ", run, "--------\n")
            
            # Construct file path
            fn = os.path.join(S['dirs']['data_dir'], f"sub-{subj:03}", f"{run}_cMEG_Data", f"{run}_meg.fif")
            raw = mne.io.read_raw_fif(fn)
            
            
            raw.info["line_freq"] = S["recording_info"]["line_freq"]
            raw.info["subject_info"] = {
                "id": int(subj),
                "his_id": f"{subj:03}",
                }
            
            
            # Add events using triggers from S["trigger"]
            event_list = list()
            for stim, old_trigger, new_trigger in zip(S["trigger"]["stim_id"],
                                                    S["trigger"]["old_trigger_id"],
                                                    S["trigger"]["new_trigger_id"]):
                event = mne.find_events(raw, stim_channel=stim, min_duration=0.001)
                event_list.append(mne.merge_events(event, [int(old_trigger)], new_trigger))
            
            events = np.concatenate(event_list, axis=0)
            print("\nevents: ", events)
            print("number of events: ", len(events), "\n")
            
            annot = mne.annotations_from_events(
                events=events,
                sfreq=raw.info["sfreq"],
                event_desc=S["trigger"]["event_desc"]
            )
            raw.set_annotations(annot)
            
            raw_list.append(raw)
        

        # Concatenate raws for all runs of this subject
        all_raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch="raise")
        
        bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=S["session"]["session"],
            task=task,
            run="01",
            root=S['dirs']['bids_dir'],
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
    else:
        config_path = ""

    S = set_bids_params(config_path)
    bids_conversion(S)
