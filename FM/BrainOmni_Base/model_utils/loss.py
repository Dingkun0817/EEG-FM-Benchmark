import torch
import torch.nn as nn
from einops import rearrange


def get_pcc(rec: torch.Tensor, raw: torch.Tensor):
    # B C W
    B, C, W, D = rec.shape
    x = rearrange(rec, "B C W D->(B C W) 1 D")
    y = rearrange(raw, "B C W D -> (B C W) 1 D")
    c = (
        (x - x.mean(dim=-1, keepdim=True))
        @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
        * (1.0 / (D - 1))
    ).squeeze()
    sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
    return (c / sigma).mean()


def compute_l1_loss(rec, raw):
    """
    rec  B C W D
    raw  B C W D
    """
    l1_distance = torch.abs(rec - raw)
    return torch.mean(l1_distance)


def get_time_loss(predicted, target):
    """
    rec  B C W D
    raw  B C W D
    """
    return compute_l1_loss(predicted, target)


def get_frequency_domain_loss(predicted, target):
    window = torch.hamming_window(target.shape[-1], device=predicted.device)
    predicted = window * predicted
    target = window * target

    pred_fft = torch.fft.rfft(predicted, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target, dim=-1, norm="ortho")

    pred_magnitude = torch.abs(pred_fft)
    target_magnitude = torch.abs(target_fft)

    pred_phase = torch.angle(pred_fft)
    target_phase = torch.angle(target_fft)

    magnitude_loss = compute_l1_loss(pred_magnitude, target_magnitude)
    phase_loss = compute_l1_loss(pred_phase, target_phase)
    return magnitude_loss, phase_loss


# class RBF(nn.Module):

#     def __init__(self, n_kernels=4, mul_factor=2.0, bandwidth=None):
#         super().__init__()
#         self.bandwidth_multipliers = mul_factor ** (
#             torch.arange(n_kernels) - n_kernels // 2
#         )
#         self.bandwidth = bandwidth

#     def get_bandwidth(self, L2_distances):
#         if self.bandwidth is None:
#             n_samples = L2_distances.shape[0]
#             return L2_distances.data.sum() / (n_samples**2 - n_samples)

#         return self.bandwidth

#     def forward(self, X):
#         L2_distances = torch.cdist(X, X) ** 2
#         return torch.exp(
#             -L2_distances[None, ...]
#             / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[
#                 :, None, None
#             ]
#         ).sum(dim=0)


# class MMDLoss(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.kernel = RBF()

#     def forward(self, X, Y):
#         K = self.kernel(torch.vstack([X, Y]))

#         X_size = X.shape[0]
#         XX = K[:X_size, :X_size].mean()
#         XY = K[:X_size, X_size:].mean()
#         YY = K[X_size:, X_size:].mean()
#         return XX - 2 * XY + YY


# class ClipLoss(nn.Module):
#     """CLIP (See Open AI CLIP) constrastive loss."""

#     def __init__(
#         self,
#         linear=None,
#         twin=True,
#     ):
#         super().__init__()
#         self.linear = None
#         if linear is not None:
#             self.linear_est = torch.nn.LazyLinear(linear)
#             if twin:
#                 self.linear_gt = self.linear_est
#             else:
#                 self.linear_gt = torch.nn.LazyLinear(linear)

#     def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
#         """
#         B*W C T
#         return B*W*C C
#         """
#         if self.linear:
#             estimates = self.linear_est(estimates)
#             candidates = self.linear_gt(candidates)
#         inv_norms = 1 / (1e-8 + candidates.norm(dim=2, p=2))
#         scores = torch.einsum("bct,bot,bo->bco", estimates, candidates, inv_norms)
#         return rearrange(scores, "B C O -> (B C) O")

#     def forward(self, estimate, candidate, mask):
#         """
#         estimate    B C W T
#         candidate   B C W T
#         """
#         assert mask.all()
#         assert estimate.shape == candidate.shape
#         B, C, W, T = estimate.shape
#         estimate = rearrange(estimate, "B C W T->(B W) C T")
#         candidate = rearrange(candidate, "B C W T->(B W) C T")
#         # (B W C) C
#         scores = self.get_scores(estimate, candidate)
#         target = torch.arange(C, device=estimate.device).repeat(B * W)
#         return F.cross_entropy(scores, target)


# class ClipLoss(nn.Module):
#     """CLIP (See Open AI CLIP) constrastive loss."""

#     def __init__(
#         self,
#         linear=None,
#         twin=True,
#         pool=False,
#         tmin=None,
#         tmax=None,
#         tmin_train=None,
#         tmax_train=None,
#         center=False,
#     ):
#         super().__init__()
#         self.linear = None
#         self.pool = pool
#         self.center = center
#         if linear is not None:
#             self.linear_est = torch.nn.LazyLinear(linear)
#             if twin:
#                 self.linear_gt = self.linear_est
#             else:
#                 self.linear_gt = torch.nn.LazyLinear(linear)
#         self.tmin = tmin
#         self.tmax = tmax
#         self.tmin_train = tmin_train
#         self.tmax_train = tmax_train

#     def trim_samples(self, estimates, candidates):
#         """Given estimates that is [B1, C, T] and candidates
#         which is [B2, C, T], return estimates_trim of size [B1, C, T']
#         and candidates_trim of size [B2, C, T'], such that T'
#         corresponds to the samples between [self.tmin, self.tmax]
#         """
#         trim_min, trim_max = self.tmin, self.tmax
#         if trim_min is None:
#             trim_min = 0
#         if trim_max is None:
#             trim_max = estimates.shape[-1]
#         estimates_trim = estimates[..., trim_min:trim_max]
#         candidates_trim = candidates[..., trim_min:trim_max]
#         return estimates_trim, candidates_trim

#     def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
#         """Given estimates that is [B, C, T] and candidates
#         which is [B', C, T], return a [B, B'] matrix of scores of matching.
#         """
#         estimates, candidates = self.trim_samples(estimates, candidates)
#         if self.linear:
#             estimates = self.linear_est(estimates)
#             candidates = self.linear_gt(candidates)
#         if self.pool:
#             estimates = estimates.mean(dim=2, keepdim=True)
#             candidates = candidates.mean(dim=2, keepdim=True)
#         if self.center:
#             estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
#             candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
#         inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
#         # We normalize inside the einsum, to avoid creating a copy
#         # of candidates, which can be pretty big.
#         scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
#         return scores

#     def get_probabilities(self, estimates, candidates):
#         """Given estimates that is [B, C, T] and candidates
#         which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
#         """
#         scores = self.get_scores(estimates, candidates)
#         return F.softmax(scores, dim=1)

#     def forward(self, estimate, candidate, mask=None):
#         """Warning: estimate and candidate are not symmetrical.
#         If estimate of shape [B, C, T] and candidate of size [B', C, T]
#         with B'>=B, the first B samples of candidate are targets, while
#         the remaining B'-B samples of candidate are only used as negatives.
#         """
#         #assert mask.all(), "mask is not supported for now"
#         assert estimate.size(0) <= candidate.size(
#             0
#         ), "need at least as many targets as estimates"
#         scores = self.get_scores(estimate, candidate)
#         target = torch.arange(len(scores), device=estimate.device)
#         return F.cross_entropy(scores, target)
