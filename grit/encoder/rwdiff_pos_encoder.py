import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder

@register_node_encoder('RWDIFF')
class RWDIFFNodeEncoder(torch.nn.Module):
    """ Geometric Diffusion with Random Walk Probabilities

    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, dim_emb, expand_x = True):
        super().__init__()
        pecfg = cfg.posenc_RWDIFF
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(cfg.posenc_RWDIFF.pos_enc_dim)
        else:
            self.raw_norm = None

        self.expand_x = expand_x
        self.dim_emb = pecfg.dim_pe
        self.linear_encoder = nn.Linear(cfg.posenc_RWDIFF.pos_enc_dim, self.dim_emb)

    def forward(self, batch):


        pos_enc = batch.pos_enc.to("cuda:0")
        # pos_enc = torch.tensor(batch.pos_enc).to("cuda:0")
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder(pos_enc)
        h = batch.x
        # if self.expand_x:
        #     h = self.linear_encoder(pos_enc)
        # else:
        #     h = batch.x

        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE separate in a variable
        batch.pe_EquivStableLapPE = pos_enc

        return batch





