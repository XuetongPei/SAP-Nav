"""Serve the unchanged SwinB detector with an explicit CPU thread budget.

Separate from the frozen evaluation launcher so the original source record
remains verifiable. The deployment records this launcher and its checksum.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from vlfm.vlm.grounding_dino import GroundingDINO
from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cpu-threads", required=True, type=int, choices=(1, 2, 4, 8))
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    class Server(ServerMixin, GroundingDINO):
        def process_payload(self, payload):
            return self.predict(str_to_image(payload["image"]), caption=payload["caption"]).to_json()

    config = str(Path(os.environ["VLFM_DINO_SOURCE"]) /
                 "groundingdino/config/GroundingDINO_SwinB_cfg.py")
    server = Server(config_path=config, weights_path=str(args.checkpoint))
    count = sum(p.numel() for p in server.model.parameters())
    assert count == 232903808, count
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"model": "dino_swinb", "pid": os.getpid(), "port": args.port,
                "route": "gdino", "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": digest.hexdigest(), "config": config,
                "parameter_count": count, "dtype": str(next(server.model.parameters()).dtype),
                "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                "cpu_threads": torch.get_num_threads(), "loaded_at": time.time(),
                "launcher": str(Path(__file__).resolve()),
                "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(args.metadata)
    print(json.dumps(metadata), flush=True)
    host_model(server, name="gdino", port=args.port)


if __name__ == "__main__":
    main()
