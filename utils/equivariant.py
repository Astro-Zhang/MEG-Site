"""Utilities for rigid-motion equivariant graph networks."""

import torch


def invariant_edge_features(edge_attr: torch.Tensor) -> torch.Tensor:
    """Return scalar edge features that are invariant under E(3) transforms.

    Legacy binary graphs contain 20 features in the order
    ``[inverse_distance, direction_xyz, rbf_0, ..., rbf_15]``.  Cartesian
    direction components are vectors, not invariant scalars, and feeding them
    to a scalar MLP breaks rotation equivariance.  Drop those three components
    while retaining compatibility with the existing precomputed graph files.

    Other layouts are assumed to already contain scalar invariant features.
    """
    if edge_attr.dim() == 1:
        edge_attr = edge_attr.unsqueeze(-1)
    if edge_attr.size(-1) == 20:
        edge_attr = torch.cat((edge_attr[..., :1], edge_attr[..., 4:]), dim=-1)
    return edge_attr
