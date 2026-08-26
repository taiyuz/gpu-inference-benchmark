# NGC TensorRT 24.08 is a TRT 10.x-era image. Confirm in the container:
#   python -c "import tensorrt as trt; print(trt.__version__)"
# Do not assume a patch version beyond what that print reports.
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[torch,onnx,dev]"

ENTRYPOINT ["gpu-bench"]
CMD ["--suite", "full", "--require-cuda-events"]
