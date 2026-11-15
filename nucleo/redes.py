"""
redes.py
--------
Arquitecturas de redes neuronales (Paso 3 del flujo), en PyTorch.

Las tres reciben exactamente las mismas entradas (cat, est, seq, mask) y
entregan logits para las clases [Deserción, Matriculado, Graduado]:

1. MLPProfundo      Perceptrón multicapa con embeddings: aplana todo y aprende
                    interacciones con capas densas.
2. RedRecurrente    Arquitectura híbrida: el contexto de ingreso (embeddings +
                    socioeconómicas) inicializa el estado oculto de una celda
                    GRU o LSTM que recorre la trayectoria semestre a semestre.
                    El estado que se usa para clasificar es el del último
                    semestre conocido, así un solo modelo sirve para alertar al
                    ingreso, tras el 1.er semestre o tras el 1.er año.
3. FTTransformer    Transformer tabular (Gorishniy et al., 2021): cada variable
                    se convierte en un token y la autoatención aprende qué
                    variables mirar en conjunto. Los tokens de semestres que
                    aún no ocurren se enmascaran en la atención.

ACA 2 - Aprendizaje Automático | Especialización en IA - CUN
"""
from __future__ import annotations

import torch
from torch import nn

N_CLASES = 3


def dim_embedding(cardinalidad: int, maximo: int = 8) -> int:
    """Regla práctica (fast.ai): crece con la raíz de la cardinalidad."""
    return int(min(maximo, max(2, round(1.6 * cardinalidad ** 0.56))))


class BloqueEmbeddings(nn.Module):
    def __init__(self, cardinalidades: list[int], maximo: int = 8, dropout: float = 0.0):
        super().__init__()
        self.tablas = nn.ModuleList(nn.Embedding(c, dim_embedding(c, maximo)) for c in cardinalidades)
        self.dim = sum(t.embedding_dim for t in self.tablas)
        self.drop = nn.Dropout(dropout)

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.cat([t(cat[:, j]) for j, t in enumerate(self.tablas)], dim=1))


def _densa(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout))


# --------------------------------------------------------------------------- 1
class MLPProfundo(nn.Module):
    def __init__(self, cardinalidades, d_est, d_sem, capas=(256, 128, 64), dropout=0.3):
        super().__init__()
        self.emb = BloqueEmbeddings(cardinalidades, dropout=dropout / 2)
        d = self.emb.dim + d_est + 2 * d_sem + 2
        bloques = []
        for h in capas:
            bloques.append(_densa(d, h, dropout))
            d = h
        self.cuerpo = nn.Sequential(*bloques)
        self.salida = nn.Linear(d, N_CLASES)

    def forward(self, cat, est, seq, mask):
        x = torch.cat([self.emb(cat), est, seq.flatten(1), mask], dim=1)
        return self.salida(self.cuerpo(x))


# --------------------------------------------------------------------------- 2
class RedRecurrente(nn.Module):
    def __init__(self, cardinalidades, d_est, d_sem, oculto=64, dropout=0.3, celda="gru"):
        super().__init__()
        self.celda_tipo = celda
        self.emb = BloqueEmbeddings(cardinalidades, dropout=dropout / 2)
        self.contexto = nn.Sequential(_densa(self.emb.dim + d_est, 2 * oculto, dropout),
                                      _densa(2 * oculto, oculto, dropout))
        self.h0 = nn.Linear(oculto, oculto)
        self.proy = nn.Sequential(nn.Linear(d_sem, oculto), nn.GELU())
        if celda == "gru":
            self.celda = nn.GRUCell(oculto, oculto)
        elif celda == "lstm":
            self.celda = nn.LSTMCell(oculto, oculto)
            self.c0 = nn.Linear(oculto, oculto)
        else:
            raise ValueError(celda)
        self.drop = nn.Dropout(dropout)
        self.cabeza = nn.Sequential(_densa(2 * oculto + 2, oculto, dropout), nn.Linear(oculto, N_CLASES))

    def forward(self, cat, est, seq, mask):
        c = self.contexto(torch.cat([self.emb(cat), est], dim=1))
        h = torch.tanh(self.h0(c))
        estados = [h]
        if self.celda_tipo == "lstm":
            m = torch.tanh(self.c0(c))
        for t in range(seq.shape[1]):          # S1 -> S2
            x = self.proy(seq[:, t, :])
            if self.celda_tipo == "gru":
                h = self.celda(x, h)
            else:
                h, m = self.celda(x, (h, m))
            estados.append(h)
        m1, m2 = mask[:, 0:1], mask[:, 1:2]
        # estado del último semestre conocido (etapa 0 -> h0, 1 -> h1, 2 -> h2)
        h_final = estados[0] * (1 - m1) + estados[1] * (m1 - m2) + estados[2] * m2
        return self.cabeza(torch.cat([self.drop(h_final), c, mask], dim=1))


# --------------------------------------------------------------------------- 3
class _AtencionMultiCabeza(nn.Module):
    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.c, self.dk = cabezas, d // cabezas
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sesgo):               # sesgo: [n, 1, 1, T] con 0 o -1e4
        n, T, d = x.shape
        q, k, v = self.qkv(x).reshape(n, T, 3, self.c, self.dk).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1)) / self.dk ** 0.5 + sesgo
        a = self.drop(torch.softmax(a, dim=-1))
        return self.out((a @ v).transpose(1, 2).reshape(n, T, d))


class _BloqueTransformer(nn.Module):
    def __init__(self, d, cabezas, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = _AtencionMultiCabeza(d, cabezas, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sesgo):
        x = x + self.drop(self.att(self.n1(x), sesgo))
        return x + self.drop(self.ff(self.n2(x)))


class FTTransformer(nn.Module):
    def __init__(self, cardinalidades, d_est, d_sem, d=32, capas=2, cabezas=4, dropout=0.2):
        super().__init__()
        self.emb_cat = nn.ModuleList(nn.Embedding(c, d) for c in cardinalidades)
        self.w_est = nn.Parameter(torch.randn(d_est, d) * 0.1)
        self.b_est = nn.Parameter(torch.zeros(d_est, d))
        self.w_sem = nn.Parameter(torch.randn(d_sem, d) * 0.1)
        self.b_sem = nn.Parameter(torch.zeros(d_sem, d))
        self.pos_sem = nn.Parameter(torch.zeros(2, 1, d))      # identifica S1 y S2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.bloques = nn.ModuleList(_BloqueTransformer(d, cabezas, dropout) for _ in range(capas))
        self.norma = nn.LayerNorm(d)
        self.salida = nn.Linear(d, N_CLASES)
        self.n_fijos = 1 + len(cardinalidades) + d_est
        self.d_sem = d_sem

    def forward(self, cat, est, seq, mask):
        n = cat.shape[0]
        t_cat = torch.stack([e(cat[:, j]) for j, e in enumerate(self.emb_cat)], dim=1)
        t_est = est.unsqueeze(-1) * self.w_est + self.b_est
        t_sem = torch.cat([seq[:, k, :].unsqueeze(-1) * self.w_sem + self.b_sem + self.pos_sem[k]
                           for k in range(2)], dim=1)
        x = torch.cat([self.cls.expand(n, -1, -1), t_cat, t_est, t_sem], dim=1)
        visible = torch.cat([torch.ones(n, self.n_fijos, device=x.device),
                             mask[:, 0:1].expand(n, self.d_sem), mask[:, 1:2].expand(n, self.d_sem)], dim=1)
        sesgo = ((1 - visible) * -1e4).view(n, 1, 1, -1)
        for b in self.bloques:
            x = b(x, sesgo)
        return self.salida(self.norma(x[:, 0]))


class Ensamble(nn.Module):
    """Promedia las probabilidades de varios modelos (uno por pliegue de la
    validación cruzada). Es lo que se exporta a ONNX para la app."""

    def __init__(self, modelos):
        super().__init__()
        self.modelos = nn.ModuleList(modelos)

    def forward(self, cat, est, seq, mask):
        p = torch.stack([torch.softmax(m(cat, est, seq, mask), dim=1) for m in self.modelos])
        return p.mean(0)


def construir(nombre: str, cardinalidades, d_est, d_sem, **hp) -> nn.Module:
    if nombre == "mlp":
        return MLPProfundo(cardinalidades, d_est, d_sem, **hp)
    if nombre == "rnn":
        return RedRecurrente(cardinalidades, d_est, d_sem, **hp)
    if nombre == "ftt":
        return FTTransformer(cardinalidades, d_est, d_sem, **hp)
    raise ValueError(nombre)


def contar_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
