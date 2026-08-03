import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

class AdvancedTrainer:
    def __init__(self, model, num_nodes, num_drugs, lr=0.008, weight_decay=1e-5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_nodes = num_nodes
        self.num_drugs = num_drugs
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def generate_hard_negative_edges(self, pos_edge_index, pos_edge_type):
        """
        Generates Hard Negative Edges by sampling nodes within the drug/protein node range
        to teach fine-grained decision boundaries.
        """
        num_pos = pos_edge_index.shape[1]
        neg_src = pos_edge_index[0].clone()
        
        # Hard negative sampling: perturb destination within drug range for drug edges
        neg_dst = torch.randint(0, self.num_drugs, (num_pos,), device=pos_edge_index.device)
        neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)
        neg_edge_type = pos_edge_type.clone()
        return neg_edge_index, neg_edge_type

    def train_epoch(self, train_edge_index, train_edge_type, initial_features=None, batch_size=131072):
        self.model.train()
        total_loss = 0.0
        num_edges = train_edge_index.shape[1]
        perm = torch.randperm(num_edges)

        train_edge_index = train_edge_index.to(self.device)
        train_edge_type = train_edge_type.to(self.device)
        if initial_features is not None:
            initial_features = initial_features.to(self.device)

        for i in range(0, num_edges, batch_size):
            batch_indices = perm[i:i + batch_size]
            pos_edge = train_edge_index[:, batch_indices]
            pos_type = train_edge_type[batch_indices]

            neg_edge, neg_type = self.generate_hard_negative_edges(pos_edge, pos_type)

            all_edges = torch.cat([pos_edge, neg_edge], dim=1)
            all_types = torch.cat([pos_type, neg_type], dim=0)
            
            labels = torch.cat([
                torch.ones(pos_edge.shape[1], device=self.device),
                torch.zeros(neg_edge.shape[1], device=self.device)
            ])

            self.optimizer.zero_grad()
            logits = self.model(all_edges, all_types, initial_features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(batch_indices)

        return total_loss / num_edges

    @torch.no_grad()
    def evaluate(self, val_edge_index, val_edge_type, initial_features=None, max_samples=100000):
        self.model.eval()
        val_edge_index = val_edge_index.to(self.device)
        val_edge_type = val_edge_type.to(self.device)
        if initial_features is not None:
            initial_features = initial_features.to(self.device)

        if val_edge_index.shape[1] > max_samples:
            perm = torch.randperm(val_edge_index.shape[1])[:max_samples]
            val_edge_index = val_edge_index[:, perm]
            val_edge_type = val_edge_type[perm]

        neg_edge_index, neg_edge_type = self.generate_hard_negative_edges(val_edge_index, val_edge_type)

        pos_logits = self.model(val_edge_index, val_edge_type, initial_features)
        neg_logits = self.model(neg_edge_index, neg_edge_type, initial_features)

        pos_probs = torch.sigmoid(pos_logits).cpu().numpy()
        neg_probs = torch.sigmoid(neg_logits).cpu().numpy()

        y_true = np.concatenate([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
        y_scores = np.concatenate([pos_probs, neg_probs])

        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

        return auroc, auprc, y_true, y_scores

class EnsembleClassifier:
    """
    Model Ensembling: Combines R-GAT + RotatE GNN predictions with Gradient Boosted Trees (XGBoost/HistGBDT).
    """
    def __init__(self, gnn_weight=0.70):
        self.gnn_weight = gnn_weight
        self.gbdt = HistGradientBoostingClassifier(max_iter=50, random_state=42)

    def fit_and_predict(self, gnn_train_probs, train_labels, gnn_val_probs, val_labels):
        # Train GBDT auxiliary classifier
        X_train = gnn_train_probs.reshape(-1, 1)
        self.gbdt.fit(X_train, train_labels)

        # Get GBDT prediction probabilities
        X_val = gnn_val_probs.reshape(-1, 1)
        gbdt_probs = self.gbdt.predict_proba(X_val)[:, 1]

        # Weighted Ensemble Probability
        ensemble_probs = self.gnn_weight * gnn_val_probs + (1.0 - self.gnn_weight) * gbdt_probs

        ensemble_auroc = roc_auc_score(val_labels, ensemble_probs)
        ensemble_auprc = average_precision_score(val_labels, ensemble_probs)

        return ensemble_auroc, ensemble_auprc
