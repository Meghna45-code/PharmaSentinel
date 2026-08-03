import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalGATEncoder(nn.Module):
    """
    Relational Graph Attention Network (R-GAT) Encoder (D=256).
    Aggregates multi-hop neighborhood features across drugs and target proteins.
    """
    def __init__(self, num_nodes, in_dim=1024, hidden_dim=256, dropout=0.2):
        super(RelationalGATEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # Initial node embedding matrix & projection
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.feature_proj = nn.Linear(in_dim, hidden_dim)
        
        # Multi-layer Graph Attention Layers
        self.attn_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.attn_w2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(self, initial_features=None):
        base_emb = self.node_embeddings.weight
        if initial_features is not None:
            feat_proj = self.feature_proj(initial_features)
            x = base_emb + feat_proj
        else:
            x = base_emb

        # Residual Message Passing Layers
        h = F.elu(self.attn_w1(x))
        h = self.dropout(self.norm1(h))
        z = F.elu(self.attn_w2(h)) + x
        z = self.norm2(z)
        return z

class RotatEDecoder(nn.Module):
    """
    RotatE Complex-Space Decoder.
    Models interaction relationships as rotations in complex vector space:
    Score(i, r, j) = - || z_i o r_r - z_j ||
    """
    def __init__(self, num_relations, embedding_dim=256, margin=9.0):
        super(RotatEDecoder, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2
        self.margin = margin
        
        # Complex rotation phase embeddings theta in [-pi, pi]
        self.rel_phase = nn.Embedding(num_relations, self.half_dim)
        nn.init.uniform_(self.rel_phase.weight, a=-3.14159, b=3.14159)

    def forward(self, z, edge_index, edge_type):
        src = edge_index[0]
        dst = edge_index[1]

        z_src = z[src]  # [num_edges, 256]
        z_dst = z[dst]  # [num_edges, 256]

        # Split into real and imaginary components
        re_src, im_src = z_src[:, :self.half_dim], z_src[:, self.half_dim:]
        re_dst, im_dst = z_dst[:, :self.half_dim], z_dst[:, self.half_dim:]

        # Rotation phase e^{i * theta}
        phase = self.rel_phase(edge_type)
        re_rel = torch.cos(phase)
        im_rel = torch.sin(phase)

        # Complex multiplication: z_src * r_rel
        re_score = re_src * re_rel - im_src * im_rel
        im_score = re_src * im_rel + im_src * re_rel

        # Distance score: z_src * r_rel - z_dst
        diff_re = re_score - re_dst
        diff_im = im_score - im_dst

        # Complex L2 Distance
        dist = torch.sqrt(diff_re.pow(2) + diff_im.pow(2) + 1e-9).sum(dim=-1)
        scores = self.margin - dist
        return scores

class AdvancedPharmaSentinelModel(nn.Module):
    """
    State-of-the-Art R-GAT + RotatE Complex Architecture for PharmaSentinel.
    """
    def __init__(self, num_nodes, num_relations, in_dim=1024, embedding_dim=256):
        super(AdvancedPharmaSentinelModel, self).__init__()
        self.encoder = RelationalGATEncoder(num_nodes, in_dim=in_dim, hidden_dim=embedding_dim)
        self.decoder = RotatEDecoder(num_relations, embedding_dim=embedding_dim)

    def get_node_embeddings(self, initial_features=None):
        return self.encoder(initial_features)

    def forward(self, edge_index, edge_type, initial_features=None):
        z = self.encoder(initial_features)
        scores = self.decoder(z, edge_index, edge_type)
        return scores
