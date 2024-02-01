# ApplyIFT
#
#

__all__ = []
__author__ = ["John Wendel"]


import uproot
import numpy as np
import os
import argparse
import time
import pandas as pd
import pickle
import torch
import joblib
from model import BaselineModel
import uproot3
import yaml

with open("IFT/preprocess_branches.yml") as branch:
    branches = yaml.load(branch, Loader=yaml.FullLoader)


   

def dot4v(v1, v2):
    '''     Perform n four-vector dot products on vectors of shape (n, 4)     (not a matrix product!)     '''
    metric = np.diag(np.array([-1, -1, -1, 1]))
    return np.einsum('ij,ij->i', v1, np.dot(v2, metric))


def apply_to_data(data, model_fname, scaler_fname):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = data["features"]
    feat_dim = len(features[1])


    evt_borders = data["evt_borders"]
    assert evt_borders[-1] == len(features)

    #features[features[:, 3] == -1, 3] = 0

    scaler = joblib.load(f"{scaler_fname}")
    features = scaler.transform(features)

    borders = np.array(list(zip(evt_borders[:-1], evt_borders[1:])))
    idx_vec = np.zeros(len(features), dtype=np.int64)
    for i, (b, e) in enumerate(borders):
        idx_vec[b:e] = i

    feat = torch.tensor(features).to(device)
    idx = torch.tensor(idx_vec).to(device)

    batch_size = 1000
    batch_borders = [
        (x[0, 0], x[-1, 1]) for x in np.array_split(borders, len(borders) // batch_size)
    ]

    model = BaselineModel(model_fname, lat_space_dim=feat_dim, in_feature_dim=feat_dim).to(device)

    model.eval()

    mypreds = np.zeros((len(borders), 1))

    for (beg, end) in batch_borders:
        tmp_data = feat[beg:end]
        # indices for the index_add inside the forward()
        tmp_idx = idx[beg:end] - idx[beg]

        e_beg, e_end = idx[[beg, end - 1]]
        # one past the last event is the boundary
        e_end += 1

        with torch.no_grad():
            output = model(tmp_data, tmp_idx)

        mypreds[e_beg:e_end] = torch.sigmoid(output).detach().cpu().numpy()

    pred_tags = mypreds.squeeze()
    eta = np.where(pred_tags > 0.5, 1 - pred_tags, pred_tags)
    calib_tags = np.where(pred_tags > 0.5, 1, -1).astype(np.int32)

    Export = {}
    Export["B_ID"] = data["B_ID"]

    Export["Standard_IFT_TAGDEC"] = np.zeros(len(Export["B_ID"]))
    Export["Standard_IFT_TAGETA"] = np.ones(len(Export["B_ID"]), dtype=int)*0.5
    kept_evts_master = data["Kept_events_master"]
    kept_evts_IFT = data["Kept_events"]
    assert len(kept_evts_IFT)==len(calib_tags)
    print(f"{ len(kept_evts_IFT)}/{len(kept_evts_master)} have a decision after cuts")
    mask = np.isin(kept_evts_master, kept_evts_IFT)
    Export["Standard_IFT_TAGDEC"][mask] = calib_tags
    Export["Standard_IFT_TAGETA"][mask] = eta
    print("predictions done")
    print(Export["Standard_IFT_TAGDEC"][:10])
    print(Export["Standard_IFT_TAGETA"][:10])



    return Export, kept_evts_master 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applying the Inclusive Flavour Tagger (IFT and Split IFT(D)) from a root file while creating a new one.")
    parser.add_argument( "filename", help="File to apply the inclusive flavour tagging to (*.npz file).")
    parser.add_argument( "-rootfile", help="File to apply the inclusive flavour tagging to (*.npz file).")
    parser.add_argument( "-NNmodel",help="File name to load NN model weights from. (.pt file)")
    parser.add_argument( "-scaler", default=None, help="File name to load scaler from. Default is MODELNAME_scaler.bin" )
    parser.add_argument( "-out", default=None, help="Filename of root file output" )
    parser.add_argument( "-mode", help="What decay channel is processed")
    parser.add_argument( "-model", help="(minimal, TrackTypes)")
    parser.add_argument( "-DataMC", help="Data or MC")
    args = parser.parse_args()

    filename = args.filename
    NNmodel = args.NNmodel 
    scaler = args.scaler
    model = args.model
    mode = args.mode
    DataMC = args.DataMC
    rootfile = args.rootfile

    #TODO: Improve this for the moment is a shit
	  #But better to discuss w/ John first
    try: 
        bkgcut=float(bkgcut)/100
    except:
	    bkgcut = False
    print("Background cut: ", bkgcut)

    
    print(f"\n File {filename} opened.")
     
    out = np.load(filename, allow_pickle=True)

    print("Starting to Apply IFT...")

    print(f" Applying model {NNmodel[:-3]} to {args.filename}")

    if scaler == None:
        scaler = NNmodel[:-3] + "_scaler.bin"

    Export, kept_evts_mstr = apply_to_data(out, NNmodel, scaler)

    tree = f"{mode}Detached/DecayTree".replace("Bu2JpsiKplus","Bu2JpsiK").replace("Bs2DsPi", "Bs2Dspi")
    tree = uproot.open(rootfile)[tree]

    with open("IFT/saveBranches.yml") as branch:
        b = yaml.load(branch, Loader=yaml.FullLoader) 
        savedBranches = list(b[mode].keys())
    
    
    if(DataMC=="MC"):
        savedBranches = savedBranches+list(b["MC"].keys())
    for b in savedBranches:
        if(b == "B_ConstJpsi_M"):
            Export[b] = tree.arrays(["B_M"],  library="np")["B_M"][kept_evts_mstr]
            constjpsi = tree.arrays([b],  library="np")[b][kept_evts_mstr]
            for i in range(len(Export["B_ConstJpsi_M"])):
                Export["B_ConstJpsi_M"][i] = constjpsi[i][0]
        else:
            Export[b]=tree.arrays([b],  library="np")[b][kept_evts_mstr]

    print(len(Export["Standard_IFT_TAGDEC"]), len(kept_evts_mstr),len(Export["B_M"]))
    savedBranches = savedBranches+["Standard_IFT_TAGDEC", "Standard_IFT_TAGETA"]

    with uproot3.recreate(f"{args.out}", compression=None) as file:
        _branches = {}

        for b in savedBranches:
            if("B_ID" in b or "B_TRUEID" in b or "Dec" in b or "TAGDEC" in b or b =="nPVs" or b == "nTracks" or b == "nLongTracks"):
                _v = np.int32
            else:
                _v = np.float64
            _branches[b] = _v
        mylist = list(dict.fromkeys(_branches.values()))
        file["DecayTree"] = uproot3.newtree(_branches)
        t = file["DecayTree"]
        for b in savedBranches:
            if("B_ID" in b or "B_TRUEID" in b or "Dec" in b or "TAGDEC" in b or b =="nPVs" or b == "nTracks" or b == "nLongTracks"):
                print("Adding {0} as int".format(b))
                print(Export[b])
                t[b].newbasket(Export[b].astype(np.int32))
            else:
                print("Adding {0} as float".format(b))
                print(Export[b])

                t[b].newbasket(Export[b].astype(np.float64))

    print("##########Programm Succeeded##########")



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
