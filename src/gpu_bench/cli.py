"""gpu-bench CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu_bench.config import BenchConfig
from gpu_bench.report import markdown_table, write_json, write_markdown
from gpu_bench.runner import available_backends, run_suite


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpu-bench",
        description="Compare PyTorch, ONNX Runtime, and TensorRT inference on ResNet-50.",
    )
    p.add_argument("--backends", default="pytorch", help="comma list: pytorch,onnx,tensorrt")
    p.add_argument("--precision", default="fp32", help="comma list: fp32,fp16,bf16")
    p.add_argument("--batch", default="1", help="comma list of batch sizes")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--graph", action="store_true", help="CUDA Graphs on pytorch/tensorrt")
    p.add_argument("--include-transfer", action="store_true", help="time H2D+compute+D2H")
    p.add_argument("--no-pinned", action="store_true")
    p.add_argument("--model", choices=("resnet50", "tiny"), default="resnet50")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--suite", choices=("default", "full"), default="default")
    p.add_argument("--out", type=Path, default=None, help="JSON report path")
    p.add_argument("--md", type=Path, default=None, help="Markdown report path")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--require-cuda-events", action="store_true")
    p.add_argument(
        "--default-stream",
        action="store_true",
        help="use the CUDA default stream (graph capture needs a non-default stream)",
    )
    p.add_argument("--list", action="store_true", help="print backend availability and exit")
    return p


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for name, (ok, reason) in available_backends().items():
            status = "ready" if ok else f"skip ({reason})"
            print(f"{name:10} {status}")
        return 0

    cfg = BenchConfig(
        model=args.model,
        precision=_csv(args.precision)[0],
        batch_size=int(_csv(args.batch)[0]),
        warmup=args.warmup,
        iters=args.iters,
        graph=args.graph,
        include_transfer=args.include_transfer,
        pinned=not args.no_pinned,
        pretrained=args.pretrained,
        artifacts_dir=args.artifacts_dir,
        require_cuda_events=args.require_cuda_events,
        use_nondefault_stream=not args.default_stream,
    )
    results = run_suite(
        backends=_csv(args.backends),
        precisions=_csv(args.precision),
        batches=[int(x) for x in _csv(args.batch)],
        graphs=args.graph,
        suite=args.suite,
        base=cfg,
    )
    print(markdown_table(results))
    if args.out:
        write_json(args.out, results)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.md:
        write_markdown(args.md, results)
        print(f"wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
