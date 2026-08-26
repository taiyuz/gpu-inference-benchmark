"""gpu-bench command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_bench.report import write_json, write_markdown
from gpu_bench.runner import expand_jobs, parse_csv, run_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-bench",
        description="Compare PyTorch, ONNX Runtime, and TensorRT 10 inference runtimes.",
    )
    p.add_argument(
        "--backends",
        action="append",
        default=None,
        help="Comma-separated or repeatable: pytorch,onnx,tensorrt. Default: all available.",
    )
    p.add_argument(
        "--precision",
        default="fp32,fp16",
        help="Comma-separated precisions (default: fp32,fp16).",
    )
    p.add_argument(
        "--batch",
        default="1",
        help="Comma-separated batch sizes (default: 1).",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--graph", action="store_true", help="Capture CUDA Graphs when supported.")
    p.add_argument(
        "--include-transfer",
        action="store_true",
        help="Time H2D + compute + D2H instead of compute-only after the input is on device.",
    )
    p.add_argument(
        "--pinned",
        action="store_true",
        help="Pin host memory for H2D copies.",
    )
    p.add_argument("--model", choices=["resnet50", "tiny"], default="resnet50")
    p.add_argument("--out", default="report.json", help="JSON report path.")
    p.add_argument("--md", default="report.md", help="Markdown report path.")
    p.add_argument(
        "--require-cuda-events",
        action="store_true",
        help="Fail if CUDA event timing is unavailable (do not fall back to wall-clock).",
    )
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument(
        "--suite",
        choices=["full"],
        default=None,
        help="Recruiting comparison: pytorch/onnx/tensorrt, fp32/fp16, "
        "batches 1/8/16/32, graphs at batch 1 and 8.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backends = parse_csv(args.backends) or None
    precisions = parse_csv(args.precision)
    batches = [int(x) for x in parse_csv(args.batch)]
    jobs = expand_jobs(
        backends=backends,
        precisions=precisions,
        batches=batches,
        graph=bool(args.graph),
        suite=args.suite,
        warmup=args.warmup,
        iters=args.iters,
        include_transfer=bool(args.include_transfer),
        pinned=bool(args.pinned),
        model=args.model,
        require_cuda_events=bool(args.require_cuda_events),
        artifacts_dir=Path(args.artifacts_dir),
    )
    suite = run_suite(jobs)
    for skip in suite.skips:
        graph = "graph" if skip.graph else "eager"
        print(
            f"skip {skip.backend} {skip.precision} batch={skip.batch_size} "
            f"({graph}): {skip.reason}",
            file=sys.stderr,
        )
    for result in suite.results:
        print(
            f"{result.backend} {result.precision} batch={result.batch_size} "
            f"graph={result.graph} timing={result.timing_backend} "
            f"mean_ms={result.mean_ms:.3f} p99_ms={result.p99_ms:.3f} "
            f"ips={result.throughput_ips:.1f} mem={result.gpu_mem_bytes}"
        )
    write_json(Path(args.out), suite.results, suite.skips)
    write_markdown(Path(args.md), suite.results, suite.skips)
    print(f"wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
