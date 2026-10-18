"""M3Site seven-class head used by the MEG-Site paper."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

from utils.equivariant import invariant_edge_features


class EGNNLayer(MessagePassing):
    """Sparse GAEN layer that updates residue features and 3D coordinates."""

    def __init__(self, in_features, hidden_features, edge_dim, dropout=0.1):
        super().__init__(aggr="add")
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features + 1 + edge_dim, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
        )
        self.coord_net = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, x, pos, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        coord_diff = pos_i - pos_j
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        msg_input = torch.cat([x_i, x_j, radial, edge_attr], dim=1)
        msg = self.message_net(msg_input)
        trans = coord_diff * torch.tanh(self.coord_net(msg)) * 0.1
        return msg, trans

    def aggregate(self, inputs, index, dim_size=None):
        msg, trans = inputs
        msg_aggr = scatter(msg, index, dim=0, dim_size=dim_size, reduce="add")
        trans_aggr = scatter(trans, index, dim=0, dim_size=dim_size, reduce="mean")
        return msg_aggr, trans_aggr

    def update(self, aggr_out, x, pos):
        msg_aggr, trans_aggr = aggr_out
        pos = pos + trans_aggr
        x = self.update_net(torch.cat([x, msg_aggr], dim=1)) + x
        return x, pos


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class MulticlassActiveSiteModel(nn.Module):
    """Residue-level 7-class graph classifier for M3Site ``esm3_abs`` files.

    Graph construction follows the idea the user requested:
      * one residue = one node
      * node feature = amino-acid id embedding + ESM sequence embedding + residue properties
      * graph structure = edge_index / edge_attr / 3D coordinates
      * optional protein-level func embedding is injected after graph message passing

    The model returns raw logits with shape [num_residues, num_classes]. Input
    dimensions match M3Site preprocessing: ESM 1536, properties 62, function
    features 768, and scalar edge attributes. ``gate`` is the DSAF-style fusion.
    """

    def __init__(
        self,
        num_classes=7,
        hidden_features=512,
        num_layers=4,
        dropout=0.3,
        aa_vocab_size=4096,
        aa_emb_dim=32,
        esm_dim=1536,
        prop_dim=62,
        func_dim=768,
        edge_dim=1,
        use_func=True,
        fusion_mode="gate",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.func_dim = func_dim
        self.use_func = use_func
        self.fusion_mode = fusion_mode

        self.aa_embedding = nn.Embedding(aa_vocab_size, aa_emb_dim)
        self.esm_input_norm = nn.LayerNorm(esm_dim)
        self.prop_input_norm = nn.LayerNorm(prop_dim)

        node_input_dim = aa_emb_dim + esm_dim + prop_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_features, dropout=dropout),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        self.layers = nn.ModuleList(
            [
                EGNNLayer(
                    hidden_features,
                    hidden_features,
                    edge_dim=hidden_features,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_features) for _ in range(num_layers)]
        )
        self.refine = nn.Sequential(
            ResidualBlock(hidden_features, dropout=dropout),
            ResidualBlock(hidden_features, dropout=dropout),
        )

        classifier_dim = hidden_features
        if use_func:
            if fusion_mode not in {"gate", "macross"}:
                raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
            self.func_encoder = nn.Sequential(
                nn.Linear(func_dim, hidden_features),
                nn.LayerNorm(hidden_features),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            if fusion_mode == "gate":
                self.func_gate = nn.Sequential(
                    nn.Linear(hidden_features * 2, hidden_features),
                    nn.SiLU(),
                    nn.Linear(hidden_features, hidden_features),
                    nn.Sigmoid(),
                )
            else:
                self.func_cross = nn.MultiheadAttention(
                    hidden_features,
                    num_heads=4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.func_cross_norm = nn.LayerNorm(hidden_features)
            classifier_dim += hidden_features

        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.LayerNorm(hidden_features // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, num_classes),
        )

    def _get_batch(self, data):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = data.x.new_zeros(data.x.size(0), dtype=torch.long)
        return batch

    def _get_func(self, data, batch):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        func = data.func.float()
        if func.dim() == 1:
            if func.numel() == num_graphs * self.func_dim:
                func = func.view(num_graphs, self.func_dim)
            elif num_graphs == 1:
                func = func.unsqueeze(0)
            else:
                raise ValueError(
                    f"Invalid flat func length {func.numel()} for {num_graphs} graphs."
                )
        return func

    def forward(self, data):
        batch = self._get_batch(data)

        aa_idx = data.x.view(-1).long().clamp_min(0)
        aa_idx = aa_idx.clamp_max(self.aa_embedding.num_embeddings - 1)
        aa_feat = self.aa_embedding(aa_idx)

        esm_rep = self.esm_input_norm(data.esm_rep.float())
        prop = self.prop_input_norm(data.prop.float())
        node_input = torch.cat([aa_feat, esm_rep, prop], dim=1)
        h = self.node_encoder(node_input)
        pos = data.pos.float()

        edge_attr = invariant_edge_features(data.edge_attr.float())
        edge_attr = self.edge_encoder(edge_attr)

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h, pos = layer(h, pos, data.edge_index, edge_attr)
            h = norm(h + residual)
            h = torch.nn.functional.silu(h)

        h = self.refine(h)

        if self.use_func:
            func_graph = self.func_encoder(self._get_func(data, batch))
            if self.fusion_mode == "gate":
                func_node = func_graph[batch]
                gate = self.func_gate(torch.cat([h, func_node], dim=1))
                h = torch.cat([h * gate, func_node], dim=1)
            else:
                query = h.unsqueeze(1)
                memory = func_graph[batch].unsqueeze(1)
                cross_out, _ = self.func_cross(query, memory, memory)
                cross_out = self.func_cross_norm(cross_out.squeeze(1) + h)
                h = torch.cat([h, cross_out], dim=1)

        token_logits = self.classifier(h)
        return {"token_logits": token_logits}
