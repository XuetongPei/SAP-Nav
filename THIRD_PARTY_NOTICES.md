# Third-party sources

The repository retains the upstream VLFM MIT license and file headers. The root
license does not replace the separate licenses or terms of external models and datasets.

| Component | Source / revision | Included |
|---|---|---|
| VLFM | [rai-opensource/vlfm](https://github.com/rai-opensource/vlfm), `584ed56008754fde7997d904983607def8328322` | Core code, original MIT license |
| ActiveSpin text adapter | [XuetongPei/ActiveSpin](https://github.com/XuetongPei/ActiveSpin), `vlfm`, `2a4ca1fe68f15581b1706bce8690d3fe0c500df8` | Converter and source license in `third_party/activespin/` |
| SAM2 service wrapper | Same ActiveSpin revision, `VLFM/vlfm/vlm/sam2_hiera.py` | Wrapper retained; model source fetched separately |
| LangMap navmesh | [bo-miao/LangMap](https://github.com/bo-miao/LangMap), `3f2cd6e37a7f06818f515b480b0e81293a8269af` | `navmesh_utils.py` and source license in `third_party/langmap/` |
| EmbodiedScore-envs | Frozen workspace revision `77e5ce28a0e4df70ccbc66f048f2b6f872edd696`, `MIP/thirdparty/EmbodiedScore-envs` | Python package, MIT license and provenance; no access to its original repository is needed |

External GroundingDINO, YOLOv7, SAM2, frontier_exploration and depth_camera_filtering
sources are fetched at the exact revisions in `configs/dependencies.json`; their
upstream licenses accompany those downloads. LAVIS and other installed packages
retain their own licenses. `third_party/VENDORED_SHA256.json` records included files.

LangMap annotations, HM3D scenes and model checkpoints are not included. Obtain
them from their owners and follow the applicable terms. The
[LangMap source license](https://github.com/bo-miao/LangMap/blob/3f2cd6e37a7f06818f515b480b0e81293a8269af/LICENSE)
distinguishes its code from annotations (CC BY-NC 4.0) and third-party assets.
