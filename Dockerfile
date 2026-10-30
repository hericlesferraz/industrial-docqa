ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337

FROM ${PYTHON_IMAGE} AS model
RUN pip install --no-cache-dir huggingface_hub==2.4.0 onnx==1.23.2 onnxruntime==1.31.1
COPY docker/build_model.py /build/build_model.py
RUN python /build/build_model.py --repo BAAI/bge-m3 \
      --revision 5617a9f61b028005a4858fdac845db406aefb181 --out /models/bge-m3-int8 \
 && rm -rf /root/.cache

FROM ${PYTHON_IMAGE} AS runtime
RUN pip install --no-cache-dir uv==0.9.18
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH PYTHONPATH=/app PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY --from=model /models /models
COPY docqa ./docqa
RUN useradd --uid 10001 --create-home app && mkdir -p /data/blobs && chown -R app:app /data
USER app
EXPOSE 8000
