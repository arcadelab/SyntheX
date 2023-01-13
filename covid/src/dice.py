# Differentiable dice loss
#
# Copyright (C) 2019-2020 Robert Grupp (grupp@jhu.edu)
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch

import torch.nn.modules.loss

class DiceLoss2D(torch.nn.modules.loss._Loss):
    def __init__(self, skip_bg=True):
        super(DiceLoss2D, self).__init__()
        
        self.skip_bg = skip_bg

    def forward(self, input, target):
        # Add this to numerator and denominator to avoid divide by zero when nothing is segmented
        # and ground truth is also empty (denominator term).
        # Also allow a Dice of 1 (-1) for this case (both terms).
        eps = 1.0e-4
        
        if self.skip_bg:
            # numerator of Dice, for each class except class 0 (background)
            # multiply by -2 (usually +2), since we are minimizing the objective function and we want to maximize Dice
            numerators = -2 * torch.sum(torch.sum(target[:,1:,:,:] * input[:,1:,:,:], dim=3), dim=2) + eps

            # denominator of Dice, for each class except class 0 (background)
            denominators = torch.sum(torch.sum(target[:,1:,:,:] * target[:,1:,:,:], dim=3), dim=2) + \
                             torch.sum(torch.sum(input[:,1:,:,:] * input[:,1:,:,:], dim=3), dim=2) + eps

            # minus one to exclude the background class
            num_classes = input.shape[1] - 1
        else:
            # numerator of Dice, for each class
            # multiply by -2 (usually +2), since we are minimizing the objective function and we want to maximize Dice
            numerators = -2 * torch.sum(torch.sum(target[:,:,:,:] * input[:,:,:,:], dim=3), dim=2) + eps

            # denominator of Dice, for each class
            denominators = torch.sum(torch.sum(target[:,:,:,:] * target[:,:,:,:], dim=3), dim=2) + \
                             torch.sum(torch.sum(input[:,:,:,:] * input[:,:,:,:], dim=3), dim=2) + eps
            
            num_classes = input.shape[1]

        # Dice coefficients for each image in the batch, for each class
        dices = numerators / denominators

        # compute average Dice score for each image in the batch
        avg_dices = torch.sum(dices, dim=1) / num_classes
        
        # compute average over the batch
        return torch.mean(avg_dices)


