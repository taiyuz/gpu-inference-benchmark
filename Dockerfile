# 24.08 is a TensorRT 10.x-era NGC image. Confirm with:
#   python -c "import tensorrt as trt; print(trt.__version__)"
# Use TensorRT 10.x as shipped by this tag; do not assume a patch number.
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /workspace

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml LICENSE README.md ./
COPY src ./src

RUN uv sync --extra torch --extra onnx

COPY . .

ENV PATH="/workspace/.venv/bin:${PATH}"

ENTRYPOINT ["gpu-bench"]
CMD ["--suite", "full"]
