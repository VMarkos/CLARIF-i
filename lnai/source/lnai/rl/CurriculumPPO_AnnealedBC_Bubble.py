# CurriculumPPO_AnnealedBC_Bubble
#
# Alias: after the CPAB refactor both the generic and bubble-masked envs use the
# same MaskablePPO + annealed-BC trainer; adjacency is enforced in the env.

from lnai.rl.CurriculumPPO_AnnealedBC import CurriculumPPO_AnnealedBC


class CurriculumPPO_AnnealedBC_Bubble(CurriculumPPO_AnnealedBC):
    """Identical trainer; bubble constraints live in CurriculumSortEnv_Bubble."""
    pass
