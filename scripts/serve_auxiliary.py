"""Start an independent BLIP-2 or YOLO service, without earlier service state."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['blip2itm', 'yolov7'])
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--models', required=True, type=Path)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    from vlfm.vlm.server_wrapper import ServerMixin, host_model, str_to_image
    if args.model == 'blip2itm':
        # LAVIS/timm looks up these exact filenames in the torch hub cache.
        cache = Path(torch.hub.get_dir()) / 'checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        for name in ['blip2_pretrained.pth', 'eva_vit_g.pth']:
            target = cache / name
            source = (args.models / name).resolve()
            if target.exists() and target.resolve() != source:
                from scripts.common import ROOT, read_json, sha256
                specs = read_json(ROOT / 'configs/models.json')['models']
                expected = next(v['sha256'] for v in specs.values() if v['filename'] == name)
                if sha256(target) != expected:
                    raise ValueError(f'Conflicting BLIP-2 cache: {target}')
            elif not target.exists():
                target.symlink_to(source)
        from vlfm.vlm.blip2itm import BLIP2ITM
        class Server(ServerMixin, BLIP2ITM):
            def process_payload(self, payload):
                return {'response': self.cosine(str_to_image(payload['image']), payload['txt'])}
        model = Server()
    else:
        from vlfm.vlm.yolov7 import YOLOv7
        class Server(ServerMixin, YOLOv7):
            def process_payload(self, payload):
                return self.predict(str_to_image(payload['image'])).to_json()
        model = Server(weights=str(args.models / 'yolov7-e6e.pt'))
    host_model(model, name=args.model, port=args.port)


if __name__ == '__main__':
    main()
