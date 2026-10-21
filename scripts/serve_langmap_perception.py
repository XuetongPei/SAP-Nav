"""Serve the selected ActiveSpin perception model using VLFM's existing API."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from vlfm.vlm.server_wrapper import ServerMixin, bool_arr_to_str, host_model, str_to_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("dino_swinb", "sam21_large"), required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.model == "dino_swinb":
        from vlfm.vlm.grounding_dino import GroundingDINO

        class Server(ServerMixin, GroundingDINO):
            def process_payload(self, payload):
                return self.predict(str_to_image(payload["image"]), caption=payload["caption"]).to_json()

        config = str(Path(os.environ["VLFM_DINO_SOURCE"]) /
                     "groundingdino/config/GroundingDINO_SwinB_cfg.py")
        server = Server(config_path=config, weights_path=str(args.checkpoint))
        network, route, expected_parameters = server.model, "gdino", 232903808
    else:
        from vlfm.vlm.sam2_hiera import SAM2Hiera

        class Server(ServerMixin, SAM2Hiera):
            def process_payload(self, payload):
                mask = self.segment_bbox(str_to_image(payload["image"]), payload["bbox"])
                return {"cropped_mask": bool_arr_to_str(mask)}

        config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        server = Server(sam_checkpoint=str(args.checkpoint), sam_config=config)
        network, route, expected_parameters = server.predictor.model, "mobile_sam", 224446642
    parameters = sum(p.numel() for p in network.parameters())
    if parameters != expected_parameters:
        raise RuntimeError(f"Unexpected model size: {parameters}, expected {expected_parameters}")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"model": args.model, "pid": os.getpid(), "port": args.port, "route": route,
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
                "config": config, "parameter_count": parameters,
                "dtype": str(next(network.parameters()).dtype), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "loaded_at": time.time()}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(args.metadata)
    print(json.dumps(metadata), flush=True)
    host_model(server, name=route, port=args.port)


if __name__ == "__main__":
    main()
