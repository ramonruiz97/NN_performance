# eq_bin
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]

import uproot3 as uproot
import numpy as np
import pandas as pd


def equibins1d(x, nbin):
  n = len(x)
  return np.interp(np.linspace(0, n, nbin + 1), np.arange(n), np.sort(x))

branch = "nTracks"
# tp = "/scratch47/ramon.ruiz/tuples_ift/2017/Bs2DsPi/v1r0_full_sweighted.root"
# tp = "/scratch47/ramon.ruiz/tuples_ift/2017/Bd2JpsiKstar/v0r1_full_sweighted.root"
nbins = 4
weight = "sigBsSW"

# all_data = uproot.open(tp)["DecayTree"].pandas.df(branches=[branch, weight])
all_tps = [f"/scratch47/ramon.ruiz/tuples_ift/2017/Bs2DsPi/v0r1_nTracks{i}_sweighted.root" for i in range(4)]
all_tps += [f"/scratch47/ramon.ruiz/tuples_ift/2016/Bs2DsPi/v0r1_nTracks{i}_sweighted.root" for i in range(4)]
all_tps += [f"/scratch47/ramon.ruiz/tuples_ift/2018/Bs2DsPi/v0r1_nTracks{i}_sweighted.root" for i in range(4)]

all_data = pd.concat([uproot.open(tp)["DecayTree"].pandas.df(branches=[branch, weight]) for tp in all_tps])

all_data = all_data.sort_values(branch).reset_index(drop=True)
t = np.array(all_data[branch])
sw = np.array(all_data[weight])
# sw *= np.sum(sw) / np.sum(sw**2)

# create a good starting point
edges = equibins1d(np.array(all_data[branch]), nbins)
edges[0] = np.min(all_data[branch])
edges[-1] = np.max(all_data[branch])
counts, edges = np.histogram(all_data[branch], weights=sw, bins=edges)

# starting step size from data range
step = np.max(all_data[branch]) - np.min(all_data[branch])
step /= 1000

# start bisecting
all_ok = False
iter = 0
old_edges = []
while not all_ok:
  iter += 1
  theWeight = np.zeros(all_data.shape[0])
  old_edges.append(np.copy(edges))
  print(counts, edges)
  for i in range(0, len(counts) - 1):
    # print(i)
    if counts[i] - counts[i + 1] > 2:
      edges[i + 1] -= step
    elif counts[i + 1] - counts[i] > 2:
      edges[i + 1] += step
    else:
      all_ok = True
  for low, high in zip(edges[0:-1], edges[1:]):
    _df = all_data.query(f"{branch}>={low} & {branch}<={high}")
    # print(f"{branch}>{low} & {branch}<{high}")
    sw = np.array(_df[weight])
    sw *= np.sum(sw) / np.sum(sw**2)
    theWeight[list(_df.index)] = sw

  # update histogram with new edges
  counts, edges = np.histogram(t, weights=theWeight, bins=edges)

  # after 5 iterations, start adaptative procedure
  if iter > 5:
    if np.max(edges - old_edges[-2]) < 1e-14:
      step /= 10

  # check/ skip for convergence
  if iter > 1000:
    raise ValueError("1000 iterations and not converged")
  # if np.max(np.abs(np.diff(counts))) < 10:
  if step < 1e-10:
    all_ok = True



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
