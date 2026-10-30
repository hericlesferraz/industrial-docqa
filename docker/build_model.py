"""Download the BGE-M3 ONNX export at a pinned revision and quantize it to int8, at image build time only."""

import argparse
import contextlib
import hashlib
import importlib
import json
import resource
import shutil
import time
from pathlib import Path
from unittest import mock

import onnx
import onnxruntime
from huggingface_hub import hf_hub_download
from onnx import external_data_helper, shape_inference
from onnxruntime.quantization import QuantType, onnx_model, onnx_quantizer, quant_utils, quantize_dynamic

FILES = ("onnx/model.onnx", "onnx/model.onnx_data", "onnx/Constant_7_attr__value", "onnx/tokenizer.json")
ORT_QUANTIZE = importlib.import_module("onnxruntime.quantization.quantize")
ORT_SAVE = onnx_model.ONNXModel.save_model_to_file


def sha256_of(path: Path) -> str:
    """Hash a file in 1 MiB blocks.

    Args:
        path: File to hash.

    Returns:
        The hex sha256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    """Return the peak resident memory of this process so far.

    Returns:
        Peak RSS in MiB (Linux reports `ru_maxrss` in KiB).
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def load_graph_with_shapes(model_path: Path) -> onnx.ModelProto:
    """Stand in for ORT's loader: infer shapes on disk and load the graph without the fp32 weights.

    ORT's loader reads all 2.3 GB of weights, and the quantizer then copies and reloads them for a second shape
    inference: three copies alive at the 8.5 GB peak. Shape inference only needs the dims, which the graph has.

    Args:
        model_path: The fp32 `model.onnx`; its weights stay in the external data file next to it.

    Returns:
        The graph with inferred shapes; the initializers still point to the external data file.
    """
    inferred = quant_utils.generate_identified_filename(model_path, "-inferred")
    shape_inference.infer_shapes_path(str(model_path), str(inferred))
    model = onnx.load(inferred.as_posix(), load_external_data=False)
    quant_utils.add_infer_metadata(model)
    inferred.unlink()
    return model


def infer_shapes_in_memory(model: onnx.ModelProto) -> onnx.ModelProto:
    """Stand in for ORT's save-and-reload: run the second shape inference on the weightless graph, in memory.

    Args:
        model: Graph whose initializers still point to the external data file.

    Returns:
        A new graph with the shapes inferred again; the initializers still point to the external data file.
    """
    inferred = shape_inference.infer_shapes(model)
    quant_utils.add_infer_metadata(inferred)
    return inferred


def save_with_kept_weights(model: onnx_model.ONNXModel, output_path: str, use_external_data_format: bool) -> None:
    """Load the fp32 tensors the quantizer kept (biases, norms) into the model, then save it as ORT does.

    Args:
        model: The quantized model; runs with the source folder as working directory.
        output_path: Output file.
        use_external_data_format: Passed through to ORT.
    """
    external_data_helper.load_external_data_for_model(model.model, ".")
    ORT_SAVE(model, output_path, use_external_data_format)


def quantize_int8(source: Path, target: Path) -> None:
    """Quantize the weights to dynamic int8 with ORT, reading each fp32 weight from disk only when it is quantized.

    Same bytes as a plain `quantize_dynamic(source, target, weight_type=QuantType.QInt8)` (sha256 checked with
    onnxruntime 1.31.1 and onnx 1.23.2) at ~4 GB of peak RAM instead of 8.5 GB. ORT resolves external weights
    against the working directory, hence the `chdir`; a renamed ORT internal fails the build at `patch.object`.

    Args:
        source: The fp32 `model.onnx`, with its external data file in the same folder.
        target: Output path of the single-file int8 model.
    """
    source, target = source.resolve(), target.resolve()
    with (
        contextlib.chdir(source.parent),
        mock.patch.object(ORT_QUANTIZE, "load_model_with_shape_infer", load_graph_with_shapes),
        mock.patch.object(onnx_quantizer, "save_and_reload_model_with_shape_infer", infer_shapes_in_memory),
        mock.patch.object(onnx_model.ONNXModel, "save_model_to_file", save_with_kept_weights),
    ):
        quantize_dynamic(source, target, weight_type=QuantType.QInt8, use_external_data_format=False)


def build(repo: str, revision: str, out: Path) -> dict[str, object]:
    """Download the fp32 export, quantize its weights to int8 and keep only what the runtime reads.

    Args:
        repo: Hugging Face repository id.
        revision: Commit hash of the repository.
        out: Output folder; receives `model.onnx`, `tokenizer.json` and `MODEL_INFO.json`.

    Returns:
        The build measurements: download and quantization seconds, peak RSS and model size.
    """
    staging = out.parent / f"{out.name}-staging"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name in FILES:
        hf_hub_download(repo, name, revision=revision, local_dir=staging)
    downloaded = time.perf_counter()
    rss_after_download = peak_rss_mb()
    model = out / "model.onnx"
    quantize_int8(staging / "onnx" / "model.onnx", model)
    quantized = time.perf_counter()
    shutil.copyfile(staging / "onnx" / "tokenizer.json", out / "tokenizer.json")
    info = {
        "repo": repo,
        "revision": revision,
        "onnxruntime": onnxruntime.__version__,
        "quantization": "dynamic int8 weights (QInt8)",
        "model_sha256": sha256_of(model),
    }
    (out / "MODEL_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    shutil.rmtree(staging)
    return {
        "download_s": round(downloaded - started, 1),
        "quantize_s": round(quantized - downloaded, 1),
        "peak_rss_mb_after_download": round(rss_after_download),
        "peak_rss_mb": round(peak_rss_mb()),
        "model_mb": round(model.stat().st_size / 2**20),
    }


def main() -> None:
    """Parse the command line, build the model and print the measurements as one JSON line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"build_model": build(args.repo, args.revision, args.out)}), flush=True)


if __name__ == "__main__":
    main()
