# Third-party notices

## VisA dataset (CC BY 4.0)

This project uses the **VisA (Visual Anomaly) dataset**, `pcb1` and `pcb2` subsets, released by Amazon Science
under the Creative Commons Attribution 4.0 International license
(<https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET>).

> Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. *SPot-the-Difference Self-Supervised
> Pre-training for Anomaly Detection and Segmentation.* ECCV 2022. arXiv:2207.14315.

- The dataset itself is **not** redistributed; `ivad prepare` downloads it from the official URL.
- `manifests/visa_1cls_pcb1.csv` and `manifests/visa_1cls_pcb2.csv` contain the `pcb1` / `pcb2` rows of the official split file
  `split_csv/1cls.csv` from the spot-diff repository (Apache-2.0 code repository; upstream SHA-256
  recorded in `manifests/visa_1cls_*.provenance.json`).
- `manifests/split_pcb1.csv` and `manifests/split_pcb2.csv` add this project's fit/calib/holdout roles and per-image SHA-256.
- Figures under `reports/` contain downscaled VisA images, crops and heat-map overlays. **Changes
  made:** registration (rotation/translation), cropping, resizing and overlaid anomaly maps / defect
  outlines. No endorsement by the dataset authors is implied.

## torchvision and pretrained weights

- torchvision (BSD-3-Clause) is used as a library.
- `Wide_ResNet50_2_Weights.IMAGENET1K_V1` is downloaded at run time from download.pytorch.org and is
  not redistributed. torchvision documents that pretrained models may carry their own terms derived
  from their training data. ImageNet's terms of access limit use to non-commercial research and
  educational purposes. See `docs/handoff.md` → *Pretrained weights*.
