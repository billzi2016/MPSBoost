"""把 MPSBoost benchmark 原始 JSON 转为诚实的 Markdown 摘要。

转换器不重新计算或筛选样本，只展示 runner 已记录的中位数、加速比和完整场景。原始
JSON 必须与摘要一同保存，避免只公开有利数字。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(payload: dict) -> str:
    """将完整 benchmark payload 渲染为紧凑 Markdown 表格。"""

    lines = [
        "# MPSBoost histogram benchmark",
        "",
        "Wall time includes host preparation, buffer transfer, command submission, and synchronization.",
        "",
        "| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup | Pool reuse |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["scenarios"]:
        lines.append(
            f"| {item['name']} | {item['rows']} | {item['features']} | "
            f"{item['cpu_median_seconds']:.6f} | {item['gpu_median_seconds']:.6f} | "
            f"{item['wall_speedup']:.3f}x | "
            f"{item.get('pooled_buffer_reuse_count', 0)} |"
        )
    if payload.get("regressor_scenarios"):
        lines.extend(
            [
                "",
                "# MPSBoost end-to-end regressor benchmark",
                "",
                "Wall time includes Python input adaptation, quantization, training, model assembly, and synchronization.",
                "",
                "| Scenario | Rows | Features | CPU median (s) | MPS median (s) | Speedup | Max prediction diff |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in payload["regressor_scenarios"]:
            lines.append(
                f"| {item['name']} | {item['rows']} | {item['features']} | "
                f"{item['cpu_median_seconds']:.6f} | "
                f"{item['mps_median_seconds']:.6f} | "
                f"{item['wall_speedup']:.3f}x | "
                f"{item['max_prediction_difference']:.6g} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    """读取指定原始 JSON，并将 Markdown 写到标准输出。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    arguments = parser.parse_args()
    print(render(json.loads(arguments.input.read_text(encoding="utf-8"))), end="")


if __name__ == "__main__":
    main()
