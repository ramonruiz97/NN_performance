# IFT_performance



## Getting started


```
git clone ssh://git@gitlab.cern.ch:7999/rruizfer/ift_performance.git
cd ift_performance
conda env create -f .ci/environment.yml
conda activate ift_perf
pip install -r .ci/requirements
```

## Tasks

- [x] Create a rectangular selection for decays
- [x] Mass Fits for Bu2JpsiKplus
- [x] Mass Fits for BsDsPi
- [x] Mass Fits for BsDsPi Prompt
- [ ] Reweighter hepml
- [x] Combination of OS taggers
- [x] Combination of SS taggers
- [x] Linear calibration of Bu2JpsiKplus
- [x] Decay-time Resolution Bs2DsPi
- [x] Decay-time Fit for calibration of Bs2DsPi
- [x] Performance of tagging algorithms comparison 
- [x] Add training script
- [x] Add script for translating cuts from John to calibration structure
- [x] Do check in bins of variables: pTb, nTracks, ETA...
- [ ] Include per-event features (nPV, nTracks, ...?) as NN training feature

