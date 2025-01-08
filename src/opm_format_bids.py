## Convert Cerca OPM data to BIDS format.
# Harrison Ritz (2025)




# %% import -------------------------------------------------------------------

import mne
import mne_bids
import os
import numpy as np
import datetime


# %% set up parameters --------------------------------------------------------

root_dir = 'ROOT_DIR'
experiment = "EXPERIMENT"
data_dir = 'DATA_DIR'
line_frequency = 60.0


# subject information
subj_id = [
    1,
    2]

session=[
    '01',
    '01'
]

# run and task informatoin
run_id = [
    ["20241111_080000", "20241111_090000"],
    ["20241112_080000", "20241112_090000"],
]


empty_room_id = [
    "20241111_070000",
    "20241112_070000",
              ]

empty_room_date = [
    datetime.datetime.combine(datetime.date(2024, 11, 11), datetime.time(7,0)).replace(tzinfo=datetime.timezone.utc),
    datetime.datetime.combine(datetime.date(2024, 11, 12), datetime.time(7,0)).replace(tzinfo=datetime.timezone.utc) 
]


task_id = [
    "oddball",
    "oddball",
    ]

# event information
stim_id = [
    ["Trigger1[Z]", "Trigger2[Z]"],
    ["Trigger1[Z]", "Trigger2[Z]"],
    ]

old_trigger_id = [
    [3,3],
    [3,3],
    ]

new_trigger_id = [
    [1,2],
    [3,4],
    ]

event_desc = {    
    new_trigger_id[0][0]:"standard/left",
    new_trigger_id[0][1]:"devient/left",
    new_trigger_id[1][0]:"standard/right",
    new_trigger_id[1][1]:"devient/right",
    900:"BAD boundary",
    901:"EDGE boundary",
    }

event_id = {
    "standard/left": new_trigger_id[0][0],
    "devient/left": new_trigger_id[0][1],
    "standard/right": new_trigger_id[1][0],
    "devient/right": new_trigger_id[1][1],
    "BAD boundary": 900,
    "EDGE boundary": 901,
}





# %% convert to BIDS ---------------------------------------------------------

# set dir
orig_dir = os.path.join(data_dir, 'data', experiment, 'orig')
bids_dir = os.path.join(data_dir, 'data', experiment, 'bids')


# loop over subjects
for subj, session, runs, tasks, empty_room in zip(subj_id, session_id, run_id, task_id, empty_room_id):

    raw_list = list()
    bids_list = list()
    print("\participant: ",subj, '--------\n')

    # save empty room data
    fn_empty_room = os.path.join(orig_dir, f"Subject-{subj:03}", empty_room + '_cMEG_Data', empty_room +'_meg.fif')
    raw_empty_room = mne.io.read_raw_fif(fn_empty_room)
    raw_empty_room.info["line_freq"] = line_frequency  # specify power line frequency 
    
    emptyroom_bids_path = mne_bids.BIDSPath(
                subject=f"{subj:03}",
                session=session,
                task='emptyroom',
                run="01",
                root=bids_dir,
                )
    
    mne_bids.write_raw_bids(
        raw_empty_room, 
        emptyroom_bids_path, 
        allow_preload=True,
        overwrite=True,
        events=None,
        format='FIF',
        )


    
    # loop over runs ----------------------------------------------
    for run_count, (run, task, stims, old_triggers, new_triggers) in enumerate(zip(runs, tasks, stim_id, old_trigger_id, new_trigger_id)):

        print("\nrun: ",run, '--------\n')

        # set path
        fn = os.path.join(orig_dir, f"Subject-{subj:03}", run + '_cMEG_Data', run +'_meg.fif')

        # read raw data
        raw = mne.io.read_raw_fif(fn)
 
        # set info
        raw.info["line_freq"] = line_frequency  # specify power line frequency
        raw.info["subject_info"] = {"his_id": f"{subj:03}"}

        # add events
        event_list = list()
        for ee,(stim, old_trigger, new_trigger) in enumerate(zip(stims, old_triggers, new_triggers)): 
            event = mne.find_events(raw, stim_channel=stim, min_duration=0.001)
            event_list.append(mne.merge_events(event, [int(old_trigger)], new_trigger))

        events = np.concatenate(event_list, axis=0)
        print('events: ',events)
        print('number of events: ', len(events))
        annot = mne.annotations_from_events(events=events, 
                                            sfreq=raw.info["sfreq"], 
                                            event_desc=event_desc)
        raw.set_annotations(annot)

        # append raw to list        
        raw_list.append(raw)

        

    # concatenate raws
    all_raw = mne.concatenate_raws(raw_list, preload=True, on_mismatch='raise')

    # set path
    bids_path = mne_bids.BIDSPath(
            subject=f"{subj:03}",
            session=session,
            task=tasks,
            run=f"01",
            root=bids_dir,
            )

    # write to bids
    mne_bids.write_raw_bids(
        all_raw, 
        bids_path, 
        allow_preload=True,
        overwrite=True,
        format='FIF',
        )
    


# %%
