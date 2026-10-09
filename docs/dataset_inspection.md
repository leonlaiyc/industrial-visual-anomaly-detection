# Dataset inspection report (pre-modelling)

**Scope:** fit-role normal images only. The split was frozen first, so inspection could not touch
calibration, holdout or test images. No anomaly image or label was looked at before the threshold was
frozen. Reproduce with `ivad inspect` → `reports/inspection/`.

## Manifest
Official `split_csv/1cls.csv`, `pcb1` rows: **904 train-normal, 100 test-normal, 100 test-anomaly**,
matching the expected counts. All 1,104 images were present after extraction from the official archive.
The archive size matched the server's Content-Length (1,929,840,640 bytes).

## Visual inspection (20 seeded fit normals)
- **Contact sheet:** `contact_sheet_fit_normals.png`. The board is an HC-SR04 ultrasonic module on a
  dark grey cloth, under overhead lighting.
- **Orientation:** 4/20 boards are rotated 180° (header pins at the bottom). Over all 600 fit normals,
  116 are rotated (19%).
- **Position and tilt:** the 20-image mean is ghosted (`mean_std_fit_normals.png`). Measured on all 600
  fit normals from the board mask (`inspection_stats.json`):
  - centroid x spans ≈ 596–738 px;
  - centroid y spans ≈ 498–554 px;
  - tilt spans ≈ −2.7° to +2.3°.
- **Resolution:** every image is 1404×1070. A 256-px-wide resize (≈5.5× down) smears vias and traces;
  a 1.5–1.8× downsample keeps them (`resize_loss_crops.png`).
- **Brightness:** mean grey level 122–133 across the 20 images (≈ ±4%), so lighting is fairly stable.
- **Variable regions:** the sensor meshes and white rings vary most from image to image
  (`registered_mean_std_fit_normals.png`). This is reflective metal and plastic, and it will widen the
  per-position covariance there.

## Assumptions about camera and alignment
- **Holds:** the camera, lighting and scale appear fixed. Board size in the image varies by < 3% (mask
  area fraction 0.261–0.277).
- **Does not hold:** fixed board pose. Boards are placed by hand or loosely fixtured: shifted, slightly
  tilted, and in either of two orientations.

**Consequence:** position-specific modelling is feasible only after registration. Registration (mask
moments + 0°/180° template check) gives a sharp registered mean image with legible silkscreen, and every
fit image has an orientation decision margin ≥ 0.36. The model therefore keeps position-specific
Gaussians instead of switching to global pooling, which would risk losing small local defects.

## Decisions taken from this inspection (before any test result)
1. Registration step (D2).
2. Crop 1152×768 around the registered board, model input 768×512 (D3).
3. Robustness severities: brightness/contrast ±10% (≈2.5× the natural spread), blur σ = 1.5 px, shift
   within the natural placement range (D11).
4. Contingency (MVTec Transistor) not needed: the data were accessible and the hardware sufficient.
