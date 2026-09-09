# Data

Recordings, datasets and trained weights are deliberately **not** tracked.
They are large binaries no diff can reconstruct, and a 26 MB blob per commit
per video bloats the repository permanently. Everything here is
regenerable — this file says how.

## What is not in the repository

| Path | What it is |
|---|---|
| `scripts/recordings/` | `N.mp4` + paired `N_depth.npz` |
| `scripts/dataset.npz` | hand-track training set |
| `scripts/pose_dataset.npz` | arm-track training set |
| `scripts/models/*.pt` | six trained predictors |
| `scripts/config/depth_calibration.json` | specific to one camera unit |

## Recording

1. **Gesture Monitor** — starts the camera and publishes `/arm/color_image`
   and `/arm/depth_image`.
2. **Data Recorder** — subscribes to both and writes `N.mp4` alongside
   `N_depth.npz`.

Guidance that materially affects the result:

- **~1.2 m from the camera**, lens at chest height. A low camera foreshortens
  the arm and hides the elbow behind the wrist.
- **Hand 70–100 cm.** The depth sensor has a hard **~60 cm minimum**; closer
  than that returns invalid, those pixels are dropped, and the reading falls
  through to whatever is behind you.
- **Keep the hand in frame throughout.** One earlier session detected a hand in
  only 370 of 2572 frames (14%), leaving 250 training samples — which is why
  hand MPJPE is worse than arm MPJPE despite a hand being an easier target.
- **Move across the frame**, not toward and away. The mapping uses
  elbow→wrist geometry in the image plane; MediaPipe's pose depth is its
  weakest output.

Check framing *before* recording: watch the monitor and confirm the skeleton
stays locked to shoulder, elbow and wrist through your full range.

## Depth calibration

Required once per physical camera. **Depth Calibration** tool:

1. Point at a flat surface, measure lens-to-surface with a tape measure
2. Capture at 3+ distances spread across the range (60 / 100 / 150 / 200 cm)
3. **Fit + Save** → writes `scripts/config/depth_calibration.json`

Two points always fit a line perfectly, so two points prove nothing. Three is
the first that can disagree.

The fit for the unit used here was `scale 0.1, offset -1` — one raw `Y11` unit
is one centimetre. **Do not reuse these constants**; measure your own.

## Regenerating datasets and models

Run in order, from `scripts/dataset_ipynb/`:

| Notebook | Produces |
|---|---|
| `00_video_to_dataset` | `dataset.npz` (hand) |
| `01_pose_video_to_dataset` | `pose_dataset.npz` (arm) |
| `02_dataset_exploration` | sanity plots |
| `03` / `04` / `05` | LSTM / GRU / Transformer, both tracks |
| `06_model_comparison` | MPJPE + inference time |

`00` and `01` carry a `USE_DEPTH` switch, defaulting to whether a calibration
exists. Running the pipeline both ways on the same footage is the controlled
comparison the depth question asks for: same frames, same landmarks, one variable changed.

## Depth format

Recordings are stamped with `depth_format` and `calibrated` flags. Files
without them predate the `Y11` discovery, hold `Y12` frames that are not
distance, and are **refused** rather than silently mis-corrected. See
[FINDINGS.md](FINDINGS.md).
