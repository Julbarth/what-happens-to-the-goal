"""PCA and representational-geometry metrics (effective rank, DoM/LDA directions, alignment)."""

from gm_geometry.pca.metrics import (
    cosine_gram,
    difference_of_means_direction,
    effective_rank_spectral_entropy,
    fit_dom_directions,
    fit_lda_directions,
    huh_effective_rank_from_activations,
    participation_ratio,
    participation_ratio_from_activations,
    pca_two_components,
    probe_pc_alignment,
    whitened_difference_of_means_direction,
)

__all__ = [
    "cosine_gram",
    "difference_of_means_direction",
    "effective_rank_spectral_entropy",
    "fit_dom_directions",
    "fit_lda_directions",
    "huh_effective_rank_from_activations",
    "participation_ratio",
    "participation_ratio_from_activations",
    "pca_two_components",
    "probe_pc_alignment",
    "whitened_difference_of_means_direction",
]
