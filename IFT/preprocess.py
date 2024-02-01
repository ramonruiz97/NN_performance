# preprocess
#
#

__all__ = []
__author__ = ["John Wendel"]
__email__ = ["john.wendel@cern.ch"]


import uproot
import numpy as np
import os
import argparse
import time
import pandas as pd
import pickle
import yaml
from xgboost import XGBClassifier


with open("IFT/preprocess_branches.yml") as branch:
    branches = yaml.load(branch, Loader=yaml.FullLoader)

def dot4v(v1, v2):
    '''     Perform n four-vector dot products on vectors of shape (n, 4)     (not a matrix product!)     '''
    metric = np.diag(np.array([-1, -1, -1, 1]))
    return np.einsum('ij,ij->i', v1, np.dot(v2, metric))

def preprocess(tree, DataMC, model, BDTmodel, nsplit, bkgcut):
    brn = list(branches["NNfeatures"].keys())
  
    bids = ["B_ID"]
    if(DataMC=="MC"):
        bids = bids+["B_TRUEID"]
    #doing a very lose perselection via B_M here since this process eats too much memory, might work on splits 
    data = tree.arrays(brn+bids, cut="(B_M>5000)&(B_M<6000)", library="np")
   
    print("Data contains {0} events".format(len(data["B_ID"])))
    features = list(branches["Trainfeatures"])
    additional_features = list(branches["calculation"])
    # For Per Evt Features (like nPV, B_len etc)
    per_evt_features = list(branches["perEvtFeatures"]) 

    print("creating additional features")
    df = pd.DataFrame(data)[additional_features+per_evt_features] #
    data["diff_z"]=df["Tr_T_zfirst"] - df["B_OWNPV_Z"]
    data["diff_eta"] = df["B_LOKI_ETA"] - df["Tr_T_Eta"]
    data["cos_diff_phi"] =  np.array(list(map(lambda x : np.cos(x), df["B_LOKI_PHI"] - df["Tr_T_Phi"])))
    # Construct B 'four-vectors'
    b4v = df[['B_PX', 'B_PY', 'B_PZ', 'B_PE']].values
    proj_array = []
    # For each event, get the arrays of track momenta
    for i, (_, (px, py, pz, pe)) in enumerate(df[['Tr_T_PX', 'Tr_T_PY', 'Tr_T_PZ', 'Tr_T_E']].iterrows()):
        # Construct an array 'four-vectors' of track mometa for this event (shape (nTracks, 4))
        t4v = np.vstack((px, py, pz, pe)).T
        # Broadcast the B four-vector to the same shape as the tracks, so they can be multiplied
        b = np.broadcast_to(b4v[i], (len(t4v), 4))
        # Perform the dot product on the vector of length nTracks
        proj = dot4v(b, t4v)
        proj_array.append(proj)
    data["P_proj"] = np.array(proj_array)
    for f in per_evt_features:
        data[f]= df[f]-df["Tr_T_Eta"]+df["Tr_T_Eta"] #ugly, but fast way to ensure the correct shape
    features=features+["diff_z", "diff_eta", "cos_diff_phi", "P_proj"]+per_evt_features

    sizes = features[0]
    evt_sizes = data[sizes]

    ntracks_total = np.sum(evt_sizes)
    print("nEvents total: ", len(data["B_ID"]))
    print("nTracks total: ", ntracks_total)

    evt_idx = np.cumsum(np.concatenate(([0], evt_sizes)), dtype=np.int32)
    borders = np.array(list(zip(evt_idx[:-1], evt_idx[1:])))
    print(f"Used Model {model}")
    if(model=="minimal"):
        out_data = np.zeros((ntracks_total, len(features[1:])), dtype=np.float32) 
    elif(model=="TrackTypes" or model=="SSIFT" or model=="OSIFT"):
        out_data = np.zeros((ntracks_total, len(features[1:])+4), dtype=np.float32) # +4 for BDT decs    tr_2_evt_idx = np.zeros(ntracks_total, dtype=np.int64)
    tr_2_evt_idx = np.zeros(ntracks_total, dtype=np.int64)

         # skip first column
    for idx, name in enumerate(features[1:]):
        out_data[:, idx] = np.concatenate(data[name])

    if(model=="TrackTypes" or model=="SSIFT" or model=="OSIFT"):
        print(f"Predicting TrackTypes in {nsplit} Splits")
        df = pd.DataFrame()
        for i,f in enumerate(features[1:-len(per_evt_features)]):
            df[f] = out_data[:,i]
        # for i,f in enumerate(features[1:]):
        #     df[f] = out_data[:,i]
        BDT = pickle.load(open(BDTmodel, "rb"))

        splitsize = int(ntracks_total/nsplit)
        for split in range(nsplit):
            if(split<nsplit-1):
                print("Predicting tracks ", split*splitsize, " - ", (split+1)*splitsize-1)
                dec = BDT.predict_proba(df[split*splitsize:(split+1)*splitsize-1])
                for i in range(4):
                    out_data[:,i-4][split*splitsize:(split+1)*splitsize-1] = dec[:,i]
            elif(split==nsplit-1):
                print("Predicting tracks ", split*splitsize, " - ", ntracks_total)
                dec = BDT.predict_proba(df[split*splitsize:-1])
                for i in range(4):
                    out_data[:,i-4][split*splitsize:-1] = dec[:,i]
        # BDT_dec = BDT.predict_proba(df)
        # for i in range(4):
        #     out_data[:,i-4] = BDT_dec[:,i]

            
        print("TrackTypes Done") 

    
    for i, (b, e) in enumerate(borders):
        tr_2_evt_idx[b:e] = i
    # tuple tool tagging seems to flag some tracks as "broken" by filling this fake value for all features
    mask = out_data[:, 0] != -99999
    print(f"Filtering out {np.sum(~mask)} tracks")
    out_data = out_data[mask]
    tr_2_evt_idx = tr_2_evt_idx[mask]

    #Keep track of all events, unwanted from here on are classified as DEC=0
    kept_evts_master = np.unique(tr_2_evt_idx, return_counts=False)

    # Doing individual per track cuts
    if(bkgcut):
        print(f"Cutting backgrounds by value {bkgcut}")
        mask = out_data[:, -1] <= bkgcut
        print(f"Filtering out {np.sum(~mask)} BKG BDT tracks with cutpoint {bkgcut}")
        out_data = out_data[mask]
        tr_2_evt_idx = tr_2_evt_idx[mask]

         # just the borders of each event
    
    if(model == "SSIFT"):
        print("APPLYING SS CUTS")
        mask = out_data[:, -4] >= out_data[:, -3]
        print(f"Filtering out {np.sum(~mask)} relative BDT tracks")
        out_data = out_data[mask]
        tr_2_evt_idx = tr_2_evt_idx[mask]


        mask = out_data[:, -4] >= out_data[:, -2]
        print(f"Filtering out {np.sum(~mask)} Frag BDT tracks")
        out_data = out_data[mask]
        tr_2_evt_idx = tr_2_evt_idx[mask]


    if(model == "OSIFT"):
        print("APPLYING OS CUTS")
        mask = out_data[:, -3] >= out_data[:, -4]
        print(f"Filtering out {np.sum(~mask)} relative BDT tracks")
        out_data = out_data[mask]
        tr_2_evt_idx = tr_2_evt_idx[mask]


        mask = out_data[:, -3] >= out_data[:, -2]
        print(f"Filtering out {np.sum(~mask)} Frag BDT tracks")
        out_data = out_data[mask]
        tr_2_evt_idx = tr_2_evt_idx[mask]
        ###### Maybe check on OSFrag cuts here as well ######
    
    kept_evts, counts = np.unique(tr_2_evt_idx, return_counts=True)

    evt_borders = np.cumsum(np.concatenate(([0], counts)))

    extra_out = dict()

    print(f"Filtered out {len(evt_sizes) - len(kept_evts)} / {len(evt_sizes)} Events, after track filtering ")

    #need B_ID here for training
    if(DataMC=="MC"):
        extra_out["B_TRUEID"] = data["B_TRUEID"][kept_evts]######HERE######
    extra_out["B_ID"] = data["B_ID"][kept_evts]######HERE######


    print("ALL EVENTS: ", len(kept_evts_master))
    print("KEPT EVENTS: ", len(kept_evts))
    print("KEPT TRACKS: ", len(out_data[:,0]))

    out = {"features": out_data, "evt_borders": evt_borders, **extra_out, "Kept_events": kept_evts, "Kept_events_master": kept_evts_master}
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess (MC-)data for Flavour Tagging.")
    parser.add_argument("--input", help="File to be processed. (ROOT,(L)DST file expected)")
    parser.add_argument("--mode", help="What decay channel is processed")
    parser.add_argument("--tree", default="DecayTree", help="Full path of TTree to process. Default: 'DecayTree'")
    parser.add_argument("--out", help="Specify name for output file.")
    parser.add_argument("--model", default="minimal", help="What IFT model should be trained")
    parser.add_argument("--BDTmodel", 
												help="path to BDT model (required if model = TrackTypes, SSIFT or OSIFT)", 
												default=None, nargs='?')
    parser.add_argument("--bkgcut", default=None, help="Cut value (times 100, so 0.5 = 50) of BDT output probBKG")
    parser.add_argument("--split", default=1, type=int, help="How Many splits are being created for BDT classification (important for large filesizes)")


    args = parser.parse_args()

    tree = args.tree.replace("MC_", "").replace("Bu2JpsiKplus","Bu2JpsiK").replace("Bs2DsPi", "Bs2Dspi")#.replace("Bd2JpsiKstar", "Bd2JpsiKst")
    mode = args.mode
    model = args.model
    BDTmodel = args.BDTmodel
    bkgcut = args.bkgcut.replace("bkg", "")
    nsplit = args.split


    #TODO: Improve this for the moment is a shit
	  #But better to discuss w/ John first
    try: 
        bkgcut=float(bkgcut)/100
    except:
	    bkgcut = False
    print("Background cut: ", bkgcut)

    if("MC" in mode):
        DataMC = "MC"
    else:
        DataMC = "Data"

    brn = list(branches["NNfeaturesTrain"].keys())
    

    print("\n Preprocessing File {0}, Decay Channel = {1}, {2}".format(args.input, mode, DataMC), "\n" )
    start = time.time()

    tree = uproot.open(args.input)[tree]
    print(f"\n File {args.input} opened.")

    out = preprocess(tree, DataMC, model, BDTmodel, nsplit, bkgcut)

    print("Saving Data...", end="\r")
    np.savez(args.out, **out)
    print("Data processed and saved as {0} in {1:.0f}s".format(args.out, time.time()-start), "\n")

    print("##########Programm Succeeded##########")



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
