# model
#
#

__all__ = []
__author__ = ["John Wendel"]
__email__ = ["john.wendel@cern.ch"]


import torch
import torch.nn as nn




def run_ds(x, idx, l1, l2, l3, l4, ls_dim):
    x = l1(x)
    x = nn.functional.relu(x)
    x = l2(x)
    tmp = nn.functional.relu(x)

    # the last entry of idx is the index for the last entry
    # so +1 is equal to number of events
    x = torch.zeros(idx[-1] + 1, ls_dim, device=x.device)
    x.index_add_(0, idx, tmp)

    x = l3(x)
    x = nn.functional.relu(x)
    x = l4(x)
    return x


class BaselineModel(nn.Module):
    def __init__(
        self, load_weights_path=None, lat_space_dim=23, in_feature_dim=23, cpu=False
    ):
        super().__init__()
        self.latent_space_dim = lat_space_dim
        self.in_feature_dim = in_feature_dim

        self.l1 = nn.Linear(self.in_feature_dim, self.in_feature_dim)
        self.l2 = nn.Linear(self.in_feature_dim, self.latent_space_dim)

        self.l3 = nn.Linear(self.latent_space_dim, lat_space_dim+2)
        self.l4 = nn.Linear(lat_space_dim+2, 1)

        if load_weights_path != None:
            self.load_state_dict(
                torch.load(
                    load_weights_path, map_location=torch.device("cpu") if cpu else None
                )
            )

    def forward(self, x, idx):
        return run_ds(x, idx, self.l1, self.l2, self.l3, self.l4, self.latent_space_dim)



# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
