"""Compact serialization and summaries for attention-browser artifacts."""

from __future__ import annotations

import base64
import csv
import gzip
from pathlib import Path
from typing import Sequence

import torch


def encode_gzip_base64(data: bytes) -> str:
    compressed = gzip.compress(data, compresslevel=9, mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def quantize_attention(tensor: torch.Tensor) -> bytes:
    quantized = torch.round(tensor.clamp(0.0, 1.0) * 65535.0)
    return quantized.to(torch.uint16).numpy().tobytes(order="C")


def quantize_effect(effect: torch.Tensor) -> tuple[bytes, float]:
    max_abs = max(float(effect.abs().max()), 1e-12)
    quantized = torch.round(effect.clamp(-max_abs, max_abs) / max_abs * 32767.0)
    return quantized.to(torch.int16).numpy().tobytes(order="C"), max_abs


def write_compact_payload(path: Path, key: str, raw_bytes: bytes) -> None:
    encoded = encode_gzip_base64(raw_bytes)
    path.write_text(
        "window.__ATTENTION_COMPACT_PAYLOADS__ = "
        "window.__ATTENTION_COMPACT_PAYLOADS__ || {};\n"
        f'window.__ATTENTION_COMPACT_PAYLOADS__["{key}"] = "{encoded}";\n',
        encoding="utf-8",
    )


def write_attention_payloads(
    output_dir: Path,
    *,
    clean: torch.Tensor,
    injected_by_position: dict[int, torch.Tensor],
    artifact_stem: str = "attention",
) -> dict:
    """Write one shared clean tensor and one signed effect per position."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_file = f"{artifact_stem}.clean.uint16.gz.b64.js"
    write_compact_payload(
        output_dir / clean_file,
        "clean",
        quantize_attention(clean),
    )
    effect_files: dict[str, str] = {}
    effect_scales: dict[str, float] = {}
    for position, injected in injected_by_position.items():
        filename = (
            f"{artifact_stem}.effect_position_{position:02d}.int16.gz.b64.js"
        )
        effect_bytes, scale = quantize_effect(injected - clean)
        write_compact_payload(
            output_dir / filename,
            f"effect_{position}",
            effect_bytes,
        )
        effect_files[str(position)] = filename
        effect_scales[str(position)] = scale
    return {
        "compact_effect_payloads": True,
        "clean_payload_file": clean_file,
        "effect_payload_files": effect_files,
        "effect_scales": effect_scales,
        "payload_compression": "gzip",
        "quantization": "clean:uint16_0_to_1,effect:int16_scaled",
        "quantization_scale": 1.0,
    }


def head_summary_rows(
    clean: torch.Tensor,
    injected: torch.Tensor,
    *,
    layers: Sequence[int],
    target_token_index: int,
    item_token_indices: Sequence[int],
) -> list[dict]:
    """Compute the same per-head diagnostics as the original browser run."""
    delta = injected - clean
    item_index = torch.tensor(list(item_token_indices), dtype=torch.long)
    rows: list[dict] = []
    for layer_offset, layer in enumerate(layers):
        for head in range(int(clean.shape[1])):
            clean_head = clean[layer_offset, head]
            injected_head = injected[layer_offset, head]
            delta_head = delta[layer_offset, head]
            clean_item_mass = (
                clean_head.index_select(0, item_index)
                .index_select(1, item_index)
                .sum()
            )
            injected_item_mass = (
                injected_head.index_select(0, item_index)
                .index_select(1, item_index)
                .sum()
            )
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "mean_query_total_variation": float(
                        0.5 * torch.abs(delta_head).sum(dim=-1).mean()
                    ),
                    "target_query_total_variation": float(
                        0.5
                        * torch.abs(delta_head[target_token_index]).sum()
                    ),
                    "final_query_total_variation": float(
                        0.5 * torch.abs(delta_head[-1]).sum()
                    ),
                    "clean_choice_to_choice_mass": float(clean_item_mass),
                    "injected_choice_to_choice_mass": float(
                        injected_item_mass
                    ),
                    "delta_choice_to_choice_mass": float(
                        injected_item_mass - clean_item_mass
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
