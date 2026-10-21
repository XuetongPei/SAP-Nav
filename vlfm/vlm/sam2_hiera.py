"""SAM 2.1 Hiera-Large segmentation server, drop-in replacement for the
MobileSAM server (vlfm/vlm/sam.py).

Serves the same "/mobile_sam" route and payload schema as MobileSAMServer, so
MobileSAMClient (and thus every policy) works unchanged — only the launcher
picks this module instead of vlfm.vlm.sam.

Loads the same GroundingDINO-SwinB-companion setup as the ActiveSpin agent's
modules/object_detector.py: config "configs/sam2.1/sam2.1_hiera_l.yaml" +
checkpoint sam2_hiera_large.pt.
"""

import os
from typing import Any, List, Optional

import numpy as np
import torch

from .server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    str_to_bool_arr,
    str_to_image,
)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ModuleNotFoundError:
    print("Could not import sam2. This is OK if you are only using the client.")


class SAM2Hiera:
    def __init__(
        self,
        sam_checkpoint: str,
        sam_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        sam2_model = build_sam2(sam_config, sam_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Same contract as MobileSAM.segment_bbox: full-image boolean mask of
        the object inside bbox ([x1, y1, x2, y2])."""
        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, _, _ = self.predictor.predict(
                box=np.array(bbox), multimask_output=False
            )

        return masks[0].astype(bool)


# Re-export the client so imports keep working if anyone points here.
from vlfm.vlm.sam import MobileSAMClient  # noqa: E402,F401

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12183)
    args = parser.parse_args()

    print("Loading model...")

    class SAM2HieraServer(ServerMixin, SAM2Hiera):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            cropped_mask = self.segment_bbox(image, payload["bbox"])
            cropped_mask_str = bool_arr_to_str(cropped_mask)
            return {"cropped_mask": cropped_mask_str}

    sam2_server = SAM2HieraServer(
        sam_checkpoint=os.environ.get("SAM2_CHECKPOINT", "data/sam2.1_hiera_large.pt"),
        sam_config=os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    )
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    # Keep the route name "mobile_sam" so MobileSAMClient works unchanged.
    host_model(sam2_server, name="mobile_sam", port=args.port)
