"""Small, dependency-free helpers shared by the portable entry points."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "584ed56008754fde7997d904983607def8328322"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    paths = []
    for folder in ["scripts", "vlfm", "third_party"]:
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "configs").glob("*.json"))
    paths.extend((ROOT / "reference").glob("*-episodes.json"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def runtime_paths():
    runtime = Path(os.environ.get("VLFM_RUNTIME", ROOT / "runtime")).resolve()
    return {
        "root": ROOT, "runtime": runtime,
        "models": Path(os.environ.get("VLFM_MODELS", runtime / "models")).resolve(),
        "vlfm_python": os.environ.get("VLFM_PYTHON", str(runtime / "envs/vlfm/bin/python")),
        "sim_python": os.environ.get("LANGMAP_PYTHON", str(runtime / "envs/sim/bin/python")),
        "sam_python": os.environ.get("SAM2_PYTHON", str(runtime / "envs/sam2/bin/python")),
        "dino_source": Path(os.environ.get("VLFM_DINO_SOURCE", ROOT / "external/GroundingDINO")).resolve(),
        "yolo_source": Path(os.environ.get("VLFM_YOLO_SOURCE", ROOT / "external/yolov7")).resolve(),
        "sam_source": Path(os.environ.get("VLFM_SAM2_SOURCE", ROOT / "external/sam2")).resolve(),
    }


def base_environment(paths, gpu="0"):
    env = os.environ.copy()
    env.update(VLFM_ROOT=str(ROOT), VLFM_RUNTIME=str(paths["runtime"]),
               VLFM_PYTHON=paths["vlfm_python"], CUDA_VISIBLE_DEVICES=str(gpu),
               VLFM_DINO_SOURCE=str(paths["dino_source"]), VLFM_YOLO_SOURCE=str(paths["yolo_source"]),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HABITAT_SIM_LOG="quiet", MAGNUM_LOG="quiet", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "third_party/EmbodiedScore-envs"),
                                       env.get("PYTHONPATH", "")])
    for name, default in {
        "HF_HOME": paths["runtime"] / "cache/huggingface",
        "TORCH_HOME": paths["runtime"] / "cache/torch",
        "XDG_CACHE_HOME": paths["runtime"] / "cache",
        "EMBODIEDSCORE_DATA_ROOT": paths["runtime"] / "datasets",
        "EMBODIEDSCORE_SCENE_ROOT": paths["runtime"] / "scenes",
    }.items():
        env.setdefault(name, str(default))
    env["NO_PROXY"] = "localhost,127.0.0.1,::1," + env.get("NO_PROXY", "")
    env["no_proxy"] = env["NO_PROXY"]
    return env
