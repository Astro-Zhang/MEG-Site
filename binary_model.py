"""Binary MEG-Site model components.

``EGNN`` is the paper's Geometry-Aware Equivariant Network (GAEN), while
``CombinedModel`` implements the text-informed Dynamic Site-Aware Fusion head.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import scatter

from utils.equivariant import invariant_edge_features
from utils.util_functions import cos_sim, kl_loss


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


class EGNNLayer(MessagePassing):
    """One GAEN message-passing layer with equivariant coordinate updates."""

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


class EGNN(nn.Module):
    """Geometry-Aware Equivariant Network (GAEN) residue encoder."""

    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        input_feature = model_cfg.in_features
        hidden_features = model_cfg.hidden_features
        num_layers = model_cfg.num_layers
        edge_dim = model_cfg.edge_dim
        dropout = model_cfg.dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(input_feature, hidden_features),
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
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, data):
        return self.fc_out(self.extract_features(data))

    def extract_features(self, data):
        h = self.node_encoder(data.x.float())
        pos = data.pos.float()

        edge_attr = invariant_edge_features(data.edge_attr.float())
        edge_attr = self.edge_encoder(edge_attr)

        for layer, norm in zip(self.layers, self.norms):
            residual = h
            h, pos = layer(h, pos, data.edge_index, edge_attr)
            h = norm(h + residual)
            h = torch.nn.functional.silu(h)

        return self.refine(h)


class CombinedModel(nn.Module):
    """MEG-Site binary predictor with alignment and DSAF fusion."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tau = config.tau
        self.struct_dim = config.model.hidden_features
        self.text_dim = config.model.text_dim
        dropout = config.model.dropout

        # GAEN: multimodal residue graph encoder guided by 3D geometry.
        self.structure_model = EGNN(config)

        # Text Encoders
        self.num_attr = 17
        self.texts_encoder = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.text_dim, nhead=4, dropout=dropout, batch_first=True
                )
                for _ in range(self.num_attr)
            ]
        )
        self.text_suffix_encoder = nn.TransformerEncoderLayer(
            d_model=self.text_dim, nhead=4, dropout=dropout, batch_first=True
        )
        self.text_suffix_transformer = nn.TransformerEncoder(
            self.text_suffix_encoder, num_layers=2
        )
        self.text_crosses = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.text_dim, num_heads=4, dropout=dropout, batch_first=True
                )
                for _ in range(4)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.text_dim) for _ in range(4)])

        # Fusion & Alignment
        # Project structure features to match text dimension for alignment/fusion
        self.project_struct = nn.Linear(self.struct_dim, self.text_dim)

        # Shared Transformer for Alignment
        self.share_encoder = nn.TransformerEncoderLayer(
            d_model=self.text_dim, nhead=8, dropout=dropout, batch_first=True
        )
        self.share_transformer = nn.TransformerEncoder(self.share_encoder, num_layers=2)

        # DSAF: residue-specific semantic gate for protein-level functional text.
        self.func_encoder = nn.Sequential(
            nn.Linear(self.text_dim, self.struct_dim),
            nn.LayerNorm(self.struct_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.func_gate = nn.Sequential(
            nn.Linear(self.struct_dim * 2, self.struct_dim),
            nn.SiLU(),
            nn.Linear(self.struct_dim, self.struct_dim),
            nn.Sigmoid(),
        )

        # Final Classifier
        # Input: gated structure features + protein-level text features
        self.classifier_input_dim = self.struct_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_dim, self.struct_dim),
            nn.LayerNorm(self.struct_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.struct_dim, self.struct_dim // 2),
            nn.LayerNorm(self.struct_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.struct_dim // 2, 1),
        )

    def forward(self, data):
        # 1. Extract Structure Features
        # struct_feats: (N_total, D_struct)
        struct_feats = self.structure_model.extract_features(data)

        # 2. Extract Text Features
        # anchor_text_input_ids is a list of tensors, one for each attribute
        if isinstance(data.texts, list):
            # Process multiple attributes -> (B, 16, 768)
            text_branch_output = self._get_text_branch_output(data.texts)
        else:
            # Use full sequence output -> (B, L, 768)
            text_branch_output = data.texts

        # 3. Alignment (Contrastive Loss)
        # Project structure features: (N_total, 768)
        struct_proj = self.project_struct(struct_feats)

        # Pool structure features per protein: (B, 768)
        struct_pool = global_mean_pool(struct_proj, data.batch)

        # Apply shared transformer
        # struct_pool: (B, 1, 768)
        struct_pool_trans = self.share_transformer(struct_pool.unsqueeze(1))
        # text_branch_output: (B, 16, 768) -> (B, 16, 768)
        text_trans = self.share_transformer(text_branch_output)

        # Calculate Contrastive Loss
        cross_modal_loss = self._softlabel_loss_3d(
            struct_pool_trans, text_trans, tau=self.tau
        )

        # 4. Fusion (Residue Level)
        # encode one protein-level text vector,
        # expand it to residues, and use it to gate structure features.
        func_graph = self.func_encoder(text_trans.mean(dim=1))
        func_node = func_graph[data.batch]
        gate = self.func_gate(torch.cat([struct_feats, func_node], dim=1))

        # 5. Final Prediction
        final_input = torch.cat([struct_feats * gate, func_node], dim=1)
        logits = self.classifier(final_input).squeeze(-1)  # (N_total,)

        return {
            "token_logits": logits,
            "contrastive_loss": cross_modal_loss,
        }

    def _get_text_branch_output(self, text_outputs):
        text_outputs = [
            self.texts_encoder[i](text_outputs[i]) for i in range(self.num_attr)
        ]
        texts_output_cls = [
            text_outputs[idx][:, 0, :].unsqueeze(1)
            for idx in range(len(text_outputs))
            if idx != 3
        ]
        texts_output_cls = torch.cat(texts_output_cls, dim=1)
        texts_output_cls = self.text_suffix_transformer(texts_output_cls)
        text_func = text_outputs[3]
        x = texts_output_cls
        for i in range(4):
            _x = x
            x = self.text_crosses[i](x, text_func, text_func)
            x = self.norms[i](x[0] + _x)
        text_branch_output = x
        return text_branch_output

    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        seq_features = seq_features.mean(dim=1)
        text_features = text_features.mean(dim=1)
        seq_sim, text_sim = (
            cos_sim(seq_features, seq_features),
            cos_sim(text_features, text_features),
        )
        logits_per_seq, logits_per_text = self._get_similarity(
            seq_features, text_features
        )
        cross_modal_loss = (
            kl_loss(logits_per_seq, seq_sim, tau=tau)
            + kl_loss(logits_per_text, text_sim, tau=tau)
        ) / 2.0
        return cross_modal_loss

    def _get_similarity(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
