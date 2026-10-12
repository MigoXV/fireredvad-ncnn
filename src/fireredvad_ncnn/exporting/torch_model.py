from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FireRedVadStreamModule(nn.Module):
    def __init__(
        self,
        idim: int,
        odim: int,
        R: int,
        M: int,
        H: int,
        P: int,
        N1: int,
        S1: int,
        N2: int = 0,
        S2: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.dfsmn = DFSMN(idim, R, M, H, P, N1, S1, N2, S2, dropout)
        self.out = nn.Linear(H, odim)

    @classmethod
    def from_config(cls, config) -> "FireRedVadStreamModule":
        return cls(
            idim=config.idim,
            odim=config.odim,
            R=config.R,
            M=config.M,
            H=config.H,
            P=config.P,
            N1=config.N1,
            S1=config.S1,
            N2=config.N2,
            S2=config.S2,
            dropout=config.dropout,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        caches: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x, new_caches = self.dfsmn(input_features, caches=caches)
        logits = self.out(x)
        probs = torch.sigmoid(logits)
        return probs, new_caches


class DFSMN(nn.Module):
    def __init__(
        self,
        D: int,
        R: int,
        M: int,
        H: int,
        P: int,
        N1: int,
        S1: int,
        N2: int = 0,
        S2: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(H, P), nn.ReLU(), nn.Dropout(dropout))
        self.fsmn1 = FSMN(P, N1, S1, N2, S2)
        self.fsmns = nn.ModuleList(
            [DFSMNBlock(H, P, N1, S1, N2, S2, dropout) for _ in range(R - 1)]
        )
        dnn: list[nn.Module] = [nn.Linear(P, H), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(M - 1):
            dnn += [nn.Linear(H, H), nn.ReLU(), nn.Dropout(dropout)]
        self.dnns = nn.Sequential(*dnn)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
        caches: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        mask = None if input_lengths is None else get_mask_from_lengths(input_lengths)

        h = self.fc1(inputs)
        p = self.fc2(h)
        new_caches = []

        cache = None if caches is None else caches[0]
        memory, new_cache = self.fsmn1(p, mask=mask, cache=cache)
        new_caches.append(new_cache)

        for i, fsmn in enumerate(self.fsmns, start=1):
            cache = None if caches is None else caches[i]
            memory, new_cache = fsmn(memory, mask=mask, cache=cache)
            new_caches.append(new_cache)

        output = self.dnns(memory)
        return output, new_caches


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_length = torch.max(lengths).item()
    mask = torch.zeros(batch_size, max_length, device=lengths.device)
    for i in range(batch_size):
        mask[i, lengths[i] :] = 1
    return mask.to(torch.uint8)


class DFSMNBlock(nn.Module):
    def __init__(
        self,
        H: int,
        P: int,
        N1: int,
        S1: int,
        N2: int = 0,
        S2: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(P, H), nn.ReLU(), nn.Dropout(dropout))
        self.fc2 = nn.Linear(H, P, bias=False)
        self.fsmn = FSMN(P, N1, S1, N2, S2)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = inputs
        h = self.fc1(inputs)
        p = self.fc2(h)
        memory, new_cache = self.fsmn(p, mask=mask, cache=cache)
        output = memory + residual
        return output, new_cache


class FSMN(nn.Module):
    def __init__(
        self,
        P: int,
        N1: int,
        S1: int,
        N2: int = 0,
        S2: int = 1,
    ) -> None:
        super().__init__()
        if N1 < 1:
            raise ValueError("N1 must be greater than or equal to 1")
        self.N1, self.S1, self.N2, self.S2 = N1, S1, N2, S2
        self.lookback_padding = (N1 - 1) * S1
        self.lookback_filter = nn.Conv1d(
            in_channels=P,
            out_channels=P,
            kernel_size=N1,
            stride=1,
            padding=self.lookback_padding,
            dilation=S1,
            groups=P,
            bias=False,
        )
        if self.N2 > 0:
            self.lookahead_filter = nn.Conv1d(
                in_channels=P,
                out_channels=P,
                kernel_size=N2,
                stride=1,
                padding=(N2 - 1) * S2,
                dilation=S2,
                groups=P,
                bias=False,
            )
        else:
            self.lookahead_filter = nn.Identity()

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = inputs.size(1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            inputs = inputs.masked_fill(mask, 0.0)

        inputs = inputs.permute((0, 2, 1)).contiguous()
        residual = inputs

        if cache is not None:
            inputs = torch.cat((cache, inputs), dim=2)
        new_cache = inputs[:, :, -self.lookback_padding :]

        lookback = self.lookback_filter(inputs)
        if self.N1 > 1:
            lookback = lookback[:, :, : -(self.N1 - 1) * self.S1]
            if cache is not None:
                lookback = lookback[:, :, cache.size(2) :]
        memory = residual + lookback

        if self.N2 > 0 and sequence_length > 1:
            lookahead = self.lookahead_filter(inputs)
            memory += F.pad(lookahead[:, :, self.N2 * self.S2 :], (0, self.S2))
        memory = memory.permute((0, 2, 1)).contiguous()

        if mask is not None:
            memory = memory.masked_fill(mask, 0.0)
        return memory, new_cache
