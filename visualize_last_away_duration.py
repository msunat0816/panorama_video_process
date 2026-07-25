#!/usr/bin/env python3
"""Visualize the distribution of the final away-hold duration from metadata JSONL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "codex-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt


TITLE_FONTSIZE = 18
LABEL_FONTSIZE = 16
TICK_FONTSIZE = 11


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return records


def condition_label(record: dict[str, Any]) -> str:
    mode = record.get("mode", "unknown")
    if mode == "away_target_away_target":
        return f"{mode}:{record.get('away_pair_mode', 'unknown')}"
    return str(mode)


def display_mode(value: Any) -> str:
    return str(value).replace("_", "-")


def mode_title_line(records: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    seen: set[str] = set()
    for record in records:
        mode = record.get("mode", "unknown")
        label = display_mode(mode)
        if mode == "away_target_away_target":
            label = f"{label} [{record.get('away_pair_mode', 'unknown')}]"
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return f"Mode: {', '.join(labels) if labels else 'unknown'}"


def final_away_duration_sec(record: dict[str, Any]) -> float | None:
    candidates = (
        "exit_away_hold_sec",
        "exit_away_hold_duration_sec",
        "away_hold_sec",
        "away_hold_duration_sec",
    )
    for key in candidates:
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def extract_durations(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_condition: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    for record in records:
        duration = final_away_duration_sec(record)
        if duration is None:
            skipped += 1
            continue
        by_condition[condition_label(record)].append(duration)

    if not by_condition:
        raise ValueError(
            "No final-away duration fields found. Expected one of: "
            "exit_away_hold_sec, exit_away_hold_duration_sec, away_hold_sec, "
            "away_hold_duration_sec."
        )
    if skipped:
        print(f"Skipped {skipped} record(s) without a final-away duration field.")
    return dict(sorted(by_condition.items()))


def summary_rows(by_condition: dict[str, list[float]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for label, values in by_condition.items():
        std = pstdev(values) if len(values) > 1 else 0.0
        rows.append(
            [
                table_condition_label(label),
                str(len(values)),
                f"{mean(values):.2f}",
                f"{median(values):.2f}",
                f"{std:.2f}",
                f"{min(values):.2f}",
                f"{max(values):.2f}",
            ]
        )
    return rows


def table_condition_label(label: str) -> str:
    if ":" in label:
        mode, pair_mode = label.split(":", 1)
        display_label = f"{display_mode(mode)} [{pair_mode}]"
    else:
        display_label = display_mode(label)
    return "\n".join(
        textwrap.wrap(
            display_label,
            width=16,
            break_long_words=False,
            break_on_hyphens=True,
        )
    )


def plot_distribution(
    by_condition: dict[str, list[float]],
    output_path: Path,
    dpi: int,
    mode_line: str,
) -> None:
    labels = list(by_condition)
    colors = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
    fig.suptitle(
        f"Final Away-Hold Duration Distribution\n{mode_line}",
        fontsize=TITLE_FONTSIZE + 2,
    )

    ax = axes[0, 0]
    for index, label in enumerate(labels):
        values = by_condition[label]
        x_values = list(range(1, len(values) + 1))
        ax.plot(x_values, values, marker="o", linewidth=1.8, label=label, color=colors[index % len(colors)])
    ax.set_title("Duration by Sample Order", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Sample Number", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Final Away Duration (sec)", fontsize=LABEL_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[0, 1]
    for index, label in enumerate(labels):
        values = sorted(by_condition[label])
        x_values = list(range(1, len(values) + 1))
        ax.plot(x_values, values, marker="o", linewidth=1.8, label=label, color=colors[index % len(colors)])
    ax.set_title("Sorted Duration by Sample Rank", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Sorted Sample Number", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Final Away Duration (sec)", fontsize=LABEL_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[0, 2]
    all_values = [value for values in by_condition.values() for value in values]
    bin_count = min(12, max(4, int(len(all_values) ** 0.5) + 1))
    for index, label in enumerate(labels):
        ax.hist(
            by_condition[label],
            bins=bin_count,
            alpha=0.45,
            label=label,
            color=colors[index % len(colors)],
            edgecolor="white",
        )
    ax.set_title("Histogram", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Final Away Duration (sec)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Sample Count", fontsize=LABEL_FONTSIZE)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[1, 0]
    box_values = [by_condition[label] for label in labels]
    positions = list(range(1, len(labels) + 1))
    ax.boxplot(box_values, positions=positions, widths=0.5, showmeans=True)
    for index, values in enumerate(box_values, start=1):
        jitter = [index + ((point_index % 5) - 2) * 0.035 for point_index in range(len(values))]
        ax.scatter(jitter, values, alpha=0.75, s=36, color=colors[(index - 1) % len(colors)])
    ax.set_title("Box Plot with Raw Samples", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Condition", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Final Away Duration (sec)", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    for index, label in enumerate(labels):
        values = sorted(by_condition[label])
        cumulative_counts = list(range(1, len(values) + 1))
        ax.step(cumulative_counts, values, where="post", label=label, color=colors[index % len(colors)])
        ax.scatter(cumulative_counts, values, s=28, color=colors[index % len(colors)])
    ax.set_title("Cumulative Count Curve", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Cumulative Sample Count", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Final Away Duration (sec)", fontsize=LABEL_FONTSIZE)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)

    ax = axes[1, 2]
    ax.axis("off")
    table = ax.table(
        cellText=summary_rows(by_condition),
        colLabels=["Condition", "n", "Mean", "Median", "Std", "Min", "Max"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.2)
    for (row, _column), cell in table.get_celld().items():
        if row > 0:
            cell.set_height(cell.get_height() * 1.35)
    ax.set_title("Summary Statistics", fontsize=TITLE_FONTSIZE)

    for axis in axes.flat:
        axis.tick_params(labelsize=TICK_FONTSIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot final away-hold duration distribution from panorama metadata JSONL."
    )
    parser.add_argument("jsonl", type=Path, help="Input metadata or extracted_metadata JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <jsonl stem>_last_away_duration_distribution.png.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output or args.jsonl.with_name(
        f"{args.jsonl.stem}_last_away_duration_distribution.png"
    )
    records = read_jsonl(args.jsonl)
    by_condition = extract_durations(records)
    plot_distribution(by_condition, output_path, args.dpi, mode_title_line(records))
    print(f"Wrote plot: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
