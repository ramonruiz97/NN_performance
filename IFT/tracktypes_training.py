# tracktypes_training
#
#

__all__ = []
__author__ = ["John Wendel"]


import numpy as np 
import uproot  
import yaml
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import os

with open("IFT/preprocess_branches.yml") as branch:
    branches = yaml.load(branch, Loader=yaml.FullLoader)

def compare_train_test(clf, X_train, y_train, X_test,  y_test, bins=30, file_path=None):
    path_pdf = f"output/figures/TrackTypesBDT/{mode}"
    os.makedirs(path_pdf, exist_ok=True)
    pdf = PdfPages(f"{path_pdf}/{mode}_TrackTypes.pdf")

    for probcat in [0,1,2,3]:
        for true in [0,1,2,3]:
            if(true==probcat):
                continue

            if(true==0):
                truecat = "SS"
            if(true==1):
                truecat = "OS Decay"
            if(true==2):
                truecat = "OS Fragmentation"
            if(true==3):
                truecat = "Background"
            if(probcat==0):
                deccat = "SS"
            if(probcat==1):
                deccat = "OS Decay"
            if(probcat==2):
                deccat = "OS Fragmentation"
            if(probcat==3):
                deccat = "Background"

            decisions = []
            for X,y in ((X_train, y_train), (X_test, y_test)):
                X0=X[y==probcat]
                X1=X[y==true]
                d0 = clf.predict_proba(X0)
                d0 = d0[:,probcat]
                d1 = clf.predict_proba(X1)
                d1 = d1[:,probcat]
                decisions += [d0, d1]

            low = min(np.min(d) for d in decisions)
            high = max(np.max(d) for d in decisions)
            low_high = (low,high)
            plt.clf()
            fig = plt.figure(figsize=(8., 8.), dpi=100)
            plt.hist(decisions[0],
                     color='r', alpha=0.5, range=low_high, bins=bins,
                     histtype='stepfilled', density=True,
                     label='{0} (Training)'.format(deccat))
            plt.hist(decisions[1],
                     color='b', alpha=0.5, range=low_high, bins=bins,
                     histtype='stepfilled', density=True,
                     label='{0} (Training)'.format(truecat))

            #print decisions
            hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, density=True)
            scale = len(decisions[0]) / sum(hist)
            err = np.sqrt(hist * scale) / scale
            width = (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='{0} (Test)'.format(deccat))

            hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, density=True)
            scale = len(decisions[2]) / sum(hist)
            err = np.sqrt(hist * scale) / scale
            plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='{0} (Test)'.format(truecat))


            if(deccat=="Background"):
                dc = "BKG"
            elif(deccat=="OS Fragmentation"):
                dc = "Frag"
            elif(deccat == "OS Decay"):
                dc = "OS"
            elif(deccat == "SS"):
                dc = "SS"
            plt.xlabel("prob{}".format(dc), ha='right', x=1, fontsize = 20)
            plt.ylabel("any unit", ha='right', y=1, fontsize = 20)
            plt.title("{0} vs. {1}".format(deccat, truecat), fontsize = 20)
            if(deccat == "Background"):
                plt.legend(loc='upper left', fontsize=18)
            else:
                plt.legend(loc='upper center', fontsize=18)
            plt.minorticks_on()
            plt.grid()
            pdf.savefig()
            plt.close()
    pdf.close()


def dot4v(v1, v2):
    '''Perform n four-vector dot products on vectors of shape (n, 4)     (not a matrix product!)     '''
    metric = np.diag(np.array([-1, -1, -1, 1]))
    return np.einsum('ij,ij->i', v1, np.dot(v2, metric))

def prep_dataframe(input, tree):
    brn = list(branches["TTBDTfeaturesTrain"].keys())

    tree = uproot.open(input)[tree]
    print(f"\n File {input} Opened")

    features = list(branches["Trainfeatures"])
    additional_features = list(branches["calculation"])

    data = tree.arrays(brn, library="np")
    nEvts = len(data["B_len"])
    nTrks = np.sum(data["B_len"])
    print(f"Data contains {nEvts} events and {nTrks} tracks")

    df = pd.DataFrame(data)[additional_features]
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

    featBDT = list(branches["tracktypes"].keys()) + list(branches["Trainfeatures"].keys())[1:] + ["diff_z", "diff_eta", "cos_diff_phi", "P_proj"]

    for key in featBDT:
        data[key] = np.concatenate(data[key])
    
    data_tmp = {key: data[key] for key in featBDT}

    df = pd.DataFrame(data_tmp)
    #this is for training on less events
    # df = df[:10000000]

    #using origin flags to label properly
    labels = df["Tr_ORIG_FLAGS"] 
    df = df.drop("Tr_ORIG_FLAGS", axis=1) 

    #some background
    labels[labels==0] = -99 
    #Same Side
    labels[labels==1] = 0
    #Opposite Side
    labels[labels==2] = 1
    #OS Fragmentation
    labels[labels==3] = 2
    labels[labels==4] = 2
    #backgrounds 
    labels[labels>=5] = 3
    labels[labels<=-1] = 3


    return df, labels

def trainBDT(data, labels, mode):
    trainX, testX, trainY, testY = train_test_split(data, labels, random_state=42)
    trainWeights = trainX["weights"]
    trainX=trainX.drop("weights", axis=1) 
    testX = testX.drop("weights", axis=1) 
    print("Start Training")
    nestimators = 150
    learningrate =0.1
    maxdepth=5
    if "cuda" in os.environ["IPANEMA_BACKEND"]:
      BDT = XGBClassifier(nthread = 20,  n_estimators = nestimators, 
												learning_rate= learningrate, max_depth=maxdepth , 
												device = "gpu",
												tree_method="gpu_hist", predictor = "gpu_predictor", n_gpus=2)
    else:
      BDT = XGBClassifier(nthread = 20,  n_estimators = nestimators, 
												learning_rate= learningrate, max_depth=maxdepth, device="cpu")
    results = BDT.fit(trainX, trainY, sample_weight=trainWeights)
    compare_train_test(results, trainX, trainY, testX, testY)

    return BDT





    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a track category BDT to make predictions if a trck is SS, OS, OSFrag or BKG.")
    parser.add_argument("input", nargs="+", help="File to be processed. (ROOT,(L)DST file expected)")
    parser.add_argument("--mode", help="What decay channel is processed")
    parser.add_argument("--tree", default="DecayTree", help="Full path of TTree to process. Default: 'DecayTree'")
    parser.add_argument("--out", default=None, help="Specify name for output file.")
    args = parser.parse_args()

    tree = args.tree.replace("MC_", "").replace("Bu2JpsiKplus","Bu2JpsiK").replace("Bs2DsPi", "Bs2Dspi")#.replace("Bd2JpsiKstar", "Bd2JpsiKst")
    mode = args.mode


    print(f"\n Preprocessing Files for TrackType BDT, Decay Channel = {mode}[")
    data16, labels16 = prep_dataframe(args.input[0], tree)
    data17, labels17 = prep_dataframe(args.input[1], tree)
    data18, labels18 = prep_dataframe(args.input[2], tree)

    data = pd.concat([data16, data17, data18], ignore_index=True)
    labels = pd.concat([labels16, labels17, labels18], ignore_index=True)

     

    print(f"For Training {len(data)} tracks are used")

    for l,t in enumerate(["SS","OS", "OS Frag", "BKG"]):
        print(f"Number of {t}: {np.sum(labels==l)}")
    print("\n The following features are used for the BDT:")
    for f in data.keys():
        print(f"     {f}")

    #indroduce weights due to different number in each category (especially 80% bkg)

    nSS = np.sum(labels==0)
    nOS = np.sum(labels==1)
    nFrag = np.sum(labels==2)
    nBkg = np.sum(labels==3)

    nAll = nSS+nOS+nFrag+nBkg 

    wSS = nAll*0.25/nSS
    wOS = nAll*0.25/nOS
    wFrag = nAll*0.25/nFrag
    wBkg = nAll*0.25/nBkg 

    weights = np.ones(len(labels)) 
    weights[labels==0] = wSS 
    weights[labels==1] = wOS
    weights[labels==2] = wFrag
    weights[labels==3] = wBkg

    print(f"Training weights for each category: {wSS}(SS), {wOS}(OS), {wFrag}(Frag), {wBkg}(Bkg)")
    data["weights"] = weights

    #train and save BDT model
    BDT = trainBDT(data, labels,  mode)
    pickle.dump(BDT, open(args.out, "wb"))
    print(f"TrackTypes BDT trained and saved as {args.out}")

    print("#############Program Succeeded############")




# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
