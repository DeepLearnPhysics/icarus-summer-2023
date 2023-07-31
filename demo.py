import sys
import numpy as np
from flashmatch_manager import FlashMatchManager
from photon_library import PhotonLibrary
from utils import print_match_result

def demo(cfg_file, det_file, out_file='', particleana=None,opflashana=None, start_entry=0, repeat=1, num_tracks=10, num_entries=-1):
    """
    Run function for ToyMC
    ---------
    Arguments
      cfg_file:   string for the location of a config file
      det_file:   string for the location of a detector spec file
      out_file:   string for an output analysis csv file path (optional)
      repeat:     int, number of times to run the toy MC simulation
      num_tracks: int for number of tracks to be generated each entry(optional)
      num_entries: number of entries(event) to run the matcher on
    """
    plib = PhotonLibrary()
    mgr = FlashMatchManager(det_file, cfg_file, particleana, opflashana, plib)

    entries = mgr.entries()

    toymc = False
    if len(entries) < 1:
        toymc = True
        entries = np.arange(repeat)
    else:
        if start_entry <0:
            return
        else:
            entries = np.arange(start_entry,max(entries)+1)
            if len(entries) < 1:
                return 

        if num_entries >0:
            entries = np.arange(start_entry,min(start_entry+num_entries,max(entries)+1))
            if len(entries)<1:
                return

    if out_file:
        import os
        if os.path.isfile(out_file):
            print('Output file',out_file,'already exists. Exiting...')
            return 

    np_result = None
    counter = 0
    for entry in entries:
        sys.stdout.write('Entry %d/%d\n' %(entry,len(entries)))
        sys.stdout.write('Event %d\n' % mgr.event_id(entry))
        sys.stdout.flush()
        # Generate samples
        generator_arg = entry if not toymc else num_tracks
        print(generator_arg)
        match_input = mgr.make_flashmatch_input(generator_arg)
        match_v = mgr.match(match_input)

        if not out_file:
            print_match_result(match_input, match_v)
            continue
        
        all_matches = []
        for idx, (flash_id, tpc_id) in enumerate(zip(match_v.flash_ids, match_v.tpc_ids)):
            qcluster, flash = match_input.qcluster_v[tpc_id], match_input.flash_v[flash_id]
            flash_idx, tpc_idx = match_input.flash_v[flash_id].idx, match_input.qcluster_v[tpc_id].idx
            raw_qcluster = match_input.raw_qcluster_v[tpc_id]
            loss, reco_dx, reco_pe, duration = match_v.loss_v[idx], match_v.reco_x_v[idx], match_v.reco_pe_v[idx], match_v.duration[idx]
            matched = (flash_idx, tpc_idx) in match_input.true_match
            store = np.array([[
                mgr.event_id(entry),
                entry,
                loss,
                flash.idx,
                qcluster.idx,
                raw_qcluster.xmin,
                raw_qcluster.xmax,
                qcluster.xmin,
                qcluster.xmax,
                qcluster.xmin + reco_dx,
                qcluster.xmax + reco_dx,
                int(matched),
                len(qcluster),
                qcluster.sum(),
                qcluster.length(),
                qcluster.time_true,
                reco_pe,
                flash.sum(),
                flash.time,
                flash.time_true,
                flash.dt_prev,
                flash.dt_next,
                duration
            ]])
            all_matches.append(store)
        if out_file and len(all_matches):
            np_event = np.concatenate(all_matches, axis=0)
            if np_result is None:
                np_result = np_event
            else:
                np_result = np.concatenate([np_result,np_event],axis=0)

    if not out_file:
        return

    np.savetxt(out_file, np_result, delimiter=',', header=','.join(attribute_names()))

def attribute_names():

    return [
        'event',
        'entry',
        'loss',
        'flash_idx',
        'track_idx',
        'true_min_x',
        'true_max_x',
        'qcluster_min_x',
        'qcluster_max_x',
        'reco_min_x',
        'reco_max_x',
        'matched',
        'qcluster_num_points',
        'qcluster_sum',
        'qcluster_length',
        'qcluster_time_true',
        'hypothesis_sum',  # Hypothesis flash sum
        'flash_sum', # OpFlash Sum
        'flash_time',
        'flash_time_true',
        'flash_dt_prev',
        'flash_dt_next',
        'duration'
    ]

if __name__ == '__main__':
    import sys, argparse

    parser = argparse.ArgumentParser(description='Run flash matching')

    parser.add_argument('--cfg', default='data/flashmatch.cfg')
    parser.add_argument('--det', default='data/detector_specs.yml')
    parser.add_argument('--outfile', '-o', default='')
    parser.add_argument('--particle', '-p')
    parser.add_argument('--opflash', '-op')
    parser.add_argument('--startentry', '-s', default = 0)
    parser.add_argument('--repeat', '-r', default = 1)
    parser.add_argument('--ntracks', '-nt', default = 10)
    parser.add_argument('--nentries', '-ne', default= -1)
    args = parser.parse_args()

    cfg_file = args.cfg
    det_file = args.det
    outfile = args.outfile
    particle = args.particle
    opflash = args.opflash
    start_entry = args.startentry
    repeat = args.repeat
    num_tracks = int(args.ntracks)
    num_entries = int(args.nentries)

    demo(cfg_file, det_file, outfile, particle, opflash, start_entry, repeat, num_tracks, num_entries)