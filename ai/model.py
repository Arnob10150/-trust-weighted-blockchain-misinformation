"""Hybrid BERT-BiLSTM content risk scorer."""
import torch
import torch.nn as nn
from transformers import BertModel


class BertBiLSTM(nn.Module):
    """Frozen BERT encoder -> BiLSTM -> dropout -> sigmoid risk score in [0,1]."""

    def __init__(self, hidden=64, dropout=0.3, proj=128, name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(name)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.bert.eval()
        self.proj = nn.Linear(768, proj)          # embedding size 128 (Table II)
        self.lstm = nn.LSTM(proj, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def logits(self, ids, mask):
        with torch.no_grad():
            emb = self.bert(ids, attention_mask=mask).last_hidden_state
        x = torch.relu(self.proj(emb))
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)      # forward ; backward
        return self.fc(self.drop(h)).squeeze(-1)

    def forward(self, ids, mask):
        return torch.sigmoid(self.logits(ids, mask))


class TemperatureScaler(nn.Module):
    """Post-hoc calibration: s = sigmoid(z / T), T fit on a held-out split."""

    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    @property
    def T(self):
        return self.log_t.exp()

    def forward(self, logits):
        return logits / self.T

    def fit(self, logits, labels, iters=300):
        logits = logits.detach()
        labels = labels.float()
        opt = torch.optim.LBFGS([self.log_t], lr=0.05, max_iter=iters)
        lossf = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = lossf(self(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.T.item())
