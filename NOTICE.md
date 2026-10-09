# Third-party notices

## VisA dataset (CC BY 4.0)

This project uses the **VisA (Visual Anomaly) dataset**, `pcb1`, `pcb2` and `pcb3` subsets, released by
Amazon Science under the Creative Commons Attribution 4.0 International license
(<https://creativecommons.org/licenses/by/4.0/>; dataset licence file:
<https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET>).

> Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. *SPot-the-Difference Self-Supervised
> Pre-training for Anomaly Detection and Segmentation.* ECCV 2022. arXiv:2207.14315.
> Source: <https://github.com/amazon-science/spot-diff>

- **The dataset itself is not redistributed.** `ivad prepare` downloads the official archive from the
  official URL.
- **Derivative images are redistributed in small numbers:**
  - figures under `reports/` (downscaled VisA images, crops, contact sheets, mean images and heat-map /
    outline overlays);
  - website images under `site/static/img/` (five JPEG files) and the executive PDF
    `reports/executive_summary.pdf`, which embeds three of them;
  - when a built website (`site/dist/`) or visual-QA screenshots of it are included in a review snapshot,
    they show the same derivative images.
- **Changes made:** board registration (rotation/translation), cropping, resizing, averaging (mean images),
  and overlaid anomaly heat maps, scoring-region outlines, board-mask tints and ground-truth defect outlines.
  Scores and labels shown next to images are this project's outputs, not part of the dataset.
- `manifests/visa_1cls_pcb{1,2,3}.csv` contain the corresponding rows of the official split file
  `split_csv/1cls.csv` from the spot-diff repository (Apache-2.0 code repository; upstream SHA-256 recorded
  in `manifests/visa_1cls_*.provenance.json`). `manifests/split_pcb{1,2,3}.csv` add this project's
  fit/calib/holdout roles and per-image SHA-256.
- **No endorsement** of this project by the dataset authors or by Amazon is implied.
- The CC BY 4.0 licence covers the VisA images and their derivatives. It does not apply to this project's
  own source code or documentation (see README → *Data, weights and licensing*).

## torchvision and pretrained weights

- torchvision (BSD-3-Clause) is used as a library.
- `Wide_ResNet50_2_Weights.IMAGENET1K_V1` is downloaded at run time from download.pytorch.org and is
  not redistributed. torchvision documents that pretrained models may carry their own terms derived
  from their training data. ImageNet's terms of access limit use to non-commercial research and
  educational purposes. See `docs/handoff.md` → *Pretrained weights*.
