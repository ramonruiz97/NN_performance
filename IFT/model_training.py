# model_training
#
#

__all__ = []
__author__ = ["John Wendel"]
__email__ = ["john.wendel@cern.ch"]


#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import RobustScaler
from model import BaselineModel
import matplotlib
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
import os



def plot_roc_curve(fpr, tpr, aucs, out):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.5f'%(aucs))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate', ha='right', x=1, fontsize = 20)
    plt.ylabel('True Positive Rate', ha='right', y=1, fontsize = 20)
    plt.title('')
    plt.legend(loc='best', fontsize = 17)
    pdf.savefig()
    plt.close()

def compare_train_test( train_1, train_0, test_1, test_0, out , bins=40, file_path=None):
    low = np.min(train_0)
    high = np.max(train_1)
    low_high = (low,high)
    fig = plt.figure(figsize=(8., 8.), dpi=100)
    plt.hist(train_1,
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B0 (training)')
    plt.hist(train_0,
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', density=True,
             label='B0 bar (training)')
    hist, bins = np.histogram(test_1, bins=bins, range=low_high, density=True)
    scale = len(test_1) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='B0 (test)')

    hist, bins = np.histogram(test_0, bins=bins, range=low_high, density=True)
    scale = len(test_1) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B0 bar (test)')

    plt.xlabel("IFT Classification", ha='right', x=1, fontsize = 20)
    plt.ylabel("any unit", ha='right', y=1, fontsize = 20)
    plt.legend(loc='best', fontsize=17)
    plt.minorticks_on()
    plt.grid()
    pdf.savefig()
    plt.close()


def train_model(files, model_out_name, scaler_out_name, n_epochs, train_frac, batch_size):

    start = time.time()
    #setup gpu if available
    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("Running on cuda")
    else:
        device = torch.device("cpu")
        print("Running on cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    print("Preparing Data")
    start=time.time()
    features = np.concatenate([f["features"] for f in files]) # =input of the NN
    NNsize = len(features[0])
    tags = np.concatenate([f["B_TRUEID"] for f in files]).reshape((-1, 1)) # Labels
    if ("Bd" in args.mode):
        print("Channel = Bd")
        tags = np.where(tags == 511, 1, 0).astype(np.int32) # if tags== 511 -> becomes 1, else 0
    elif("Bs" in args.mode):
        print("Channel = Bs")
        tags = np.where(tags == 531, 1, 0).astype(np.int32) # if tags== 531 -> becomes 1, else 0
    elif("Bu" in args.mode):
        print("Channel = Bu")
        tags = np.where(tags == 521, 1, 0).astype(np.int32)

    evt_borders = files[0]["evt_borders"]
    for f in files[1:]:
        evt_borders = np.concatenate((evt_borders, f["evt_borders"][1:] + evt_borders[-1])) # borders of event

    assert evt_borders[-1] == len(features) #check if last event_border is equal to number of tracks

    # scale data, and safe scaler for later use
    scaler = RobustScaler()
    features = scaler.fit_transform(features)
    joblib.dump(scaler, scaler_out_name)

    borders = np.array(list(zip(evt_borders[:-1], evt_borders[1:]))) # formating evt borders
    idx_vec = np.zeros(len(features), dtype=np.int64)
    for i, (b, e) in enumerate(borders):
        idx_vec[b:e] = i #every track gets an index according to which event it belongs to



    evt_split = int(len(borders) * train_frac) #how many events are used for training

    track_split = evt_borders[evt_split] #what is the last track in the event we want to split?
    #What happens here is basically just splitting tags, feats and events at a point in the dataset for training and testing
    train_tags_np = tags[:evt_split]
    train_tags = torch.tensor(tags[:evt_split], dtype=torch.float32).to(device) #tags for training
    train_feat = torch.tensor(features[:track_split]).to(device) #features for training
    train_idx = torch.tensor(idx_vec[:track_split]).to(device)

    #same stuff but for test sample
    test_tags_np = tags[evt_split:]
    test_tags = torch.tensor(test_tags_np, dtype=torch.float32).to(device)
    test_feat = torch.tensor(features[track_split:]).to(device)
    test_idx = torch.tensor(idx_vec[track_split:]).to(device)
    print("Data contains {0} train Events and {1} test Events".format(len(tags)-len(test_tags_np),len(test_tags_np)))
    #do borders for training and testing
    train_borders = [(x[0, 0], x[-1, 1]) for x in np.array_split(borders[:evt_split], len(borders[:evt_split]) // batch_size)]#also doing batches
    test_borders = [
        (x[0, 0], x[-1, 1])
        for x in np.array_split(borders[evt_split:] - borders[evt_split][0], len(borders[evt_split:]) // batch_size)
    ]

    model = BaselineModel(lat_space_dim=NNsize, in_feature_dim=NNsize).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)#, , lr = 1e-2 weight_decay=1e-4
    print("ADAM: ", optimizer)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=5) #lr = learning rate

    all_tl = [] #trainloss function
    all_vl = [] #validation loss function
    all_ac = [] #accuracy

    mypreds = np.zeros((len(test_tags), 1))
    mypreds_train = np.zeros((len(train_tags), 1))

    minlosscheck=1
    minlosscounter=0
    trained_epochs=0
    print("Data successfully prepared in {0:.3f}s".format(time.time()-start))
    print("Start Training.")
    start=time.time()
    for epoch in range(n_epochs):
        model.train()
        trainloss = 0
        for batch_idx, (beg, end) in enumerate(train_borders):
            optimizer.zero_grad() #get gradient for every event

            data = train_feat[beg:end] #training data in this
            idx = train_idx[beg:end] - train_idx[beg]
            e_beg, e_end = train_idx[[beg, end - 1]]
            # one past the last event is the boundary
            e_end += 1
            target = train_tags[e_beg:e_end]

            output = model(data, idx)
            loss = nn.functional.binary_cross_entropy_with_logits(output, target)

            loss.backward()

            optimizer.step()

            trainloss += loss.detach().cpu().numpy()
            mypreds_train[e_beg:e_end] = torch.sigmoid(output.detach()).cpu().numpy()


        train1 = mypreds_train[train_tags_np>0.5]#==1
        train0 = mypreds_train[train_tags_np<0.5]#==0
        # averaged trainloss of epoch
        all_tl.append(trainloss / (batch_idx + 1))
        trainloss = 0

        model.eval()
        valloss = 0
        for batch_idx, (beg, end) in enumerate(test_borders):

            data = test_feat[beg:end]
            # indices for the index_add inside the forward()
            idx = test_idx[beg:end] - test_idx[beg]

            # minus to make the test_idx start at 0 since we are indexing into
            # the split off test_tags array
            e_beg, e_end = test_idx[[beg, end - 1]] - test_idx[0]
            # one past the last event is the boundary
            e_end += 1
            target = test_tags[e_beg:e_end]

            with torch.no_grad():
                output = model(data, idx)

            mypreds[e_beg:e_end] = torch.sigmoid(output.detach()).cpu().numpy()

            valloss += nn.functional.binary_cross_entropy_with_logits(output, target).detach().cpu().numpy()

        acc = np.mean((mypreds > 0.5) == test_tags_np)
        all_vl.append(valloss / (batch_idx + 1))
        all_ac.append(acc)
        test0 = mypreds[test_tags_np<0.5]#==0
        test1 = mypreds[test_tags_np>0.5]#==1
        scheduler.step(valloss / (batch_idx + 1))
        #check if no gain happened in 20 epochs. Abort if it didnt
        trained_epochs+=1
        if(min(all_vl)<minlosscheck):
            minlosscounter=0
            minlosscheck = min(all_vl)
            fpr, tpr, thresholds = roc_curve(test_tags_np, mypreds)
            aucs = auc(fpr, tpr) #calculate roc curve of lowest valloss
            train1_best = train1
            train0_best = train0
            test1_best=test1
            test0_best=test0
        else:
            minlosscounter+=1
        # if(minlosscounter>20):
        #     print("\n")
        #     print("NO GAIN IN 20 EPOCHS. ABORTING")
        #     break
        print(
            f"Epoch: {epoch}/{n_epochs} | Val loss {valloss/(batch_idx+1):.5f} | AUC: {roc_auc_score(test_tags_np, mypreds):.5f} | ACC: {acc:.5f} | Difference train/test loss: {abs(all_vl[-1]-all_tl[-1]):.5f} | No gain since {minlosscounter} Epochs | Duration: {time.time()-start:.0f}s",
            end="\r",
        )


    print("\n")
    print("Training complete in {0:.3f}s after {1} epochs".format(time.time()-start,trained_epochs))
    print(f"Minimum training loss: {min(all_vl):.5f} in epoch: {np.argmin(all_vl)}")
    print(f"Maximum training ACC:  {max(all_ac):.5f} in epoch: {np.argmax(all_ac)}")

    # done training so let's set it to eval
    model.eval()

    torch.save(model.state_dict(), model_out_name)

    print("Making plots.")
    plot_roc_curve(fpr, tpr, aucs, out = model_out_name)

    matplotlib.rcParams.update({"font.size": 22})

    plt.figure(figsize=(16, 9))
    plt.plot(all_tl, label="Train Loss", linewidth=3)
    plt.plot(all_vl, label="Validation Loss", linewidth=3)
    plt.legend(fontsize=17)
    plt.xlabel("Epoch", fontsize = 17)
    plt.ylim(np.min(all_tl+all_vl)-0.01, np.max(all_tl+all_vl))
    plt.grid()
    pdf.savefig()
    plt.close()

    compare_train_test(train1_best, train0_best, test1_best, test0_best, out = model_out_name)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range (0.0, 1.0]")

    return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Model for Flavour Tagging.")
    parser.add_argument("filenames", nargs="+", help="Files that contain training data. *.npz files expected)")
    parser.add_argument("-out", default="model.pt", help="File name to save weights into. Default is model.pt")
    parser.add_argument("-out-plots", help="Directory with plots")
    parser.add_argument("-scaler-out-name", default=None, help="File name to save scaler into. Default is MODELNAME_scaler.bin")
    parser.add_argument("-mode", help="Bs, Bu or Bd")
    parser.add_argument("-model", help="TracksTypes or minimal")
    parser.add_argument("-cut", help="Cut to tracktypes")
    parser.add_argument("-epochs", dest="n_epochs", default=100, type=int, help="Batch size")
    parser.add_argument("-train-frac", default=0.75, type=restricted_float, help="Fraction of data to use for training")
    parser.add_argument("-batch-size", default=1000, type=int, help="Batch size")

    args = parser.parse_args()
    print("Loading Data...")
    start = time.time()
    mode = args.mode
    model = args.model
    cut = args.cut
    filenames = args.filenames
    path_pdf = args.out_plots
    os.makedirs(path_pdf, exist_ok=True)
    pdf = PdfPages(f"{path_pdf}/{mode}_{model}_{cut}.pdf")

    if args.scaler_out_name == None:
            scaler_out_name = args.out.replace(".pt", "_scaler.bin")
    files = [np.load(f, allow_pickle=True) for f in args.filenames]
    train_model(
        files, args.out, scaler_out_name, args.n_epochs, args.train_frac, args.batch_size
    )

    pdf.close()

    print("#######Code SUCCEEDED#######")



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
