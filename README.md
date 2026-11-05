# DeepVision

Detecting AI-generated videos.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Linux
pip install -r requirements.txt
pip install -e .
```

`ffmpeg` and `ffprobe` must be available on `PATH` (or set `FFMPEG_BIN` / `FFPROBE_BIN`).

## Data pipeline

All paths and source definitions live in `configs/data.yaml`. Raw videos are never modified;
processed outputs go to `<raw_root>/deepvision_processed/`.

| Step | Command | Output |
|------|---------|--------|
| Audit | `python -m deepvision.data.audit -c configs/data.yaml` | `inventory/inventory.csv`, `inventory/summary.json` |
| Validate | `python -m deepvision.data.cleaning check -c configs/data.yaml` | `inventory/video_status.csv` |
| Quarantine broken files | `python -m deepvision.data.cleaning quarantine -c configs/data.yaml --apply` | `logs/quarantine_moves.csv` |
| Manifests | `python -m deepvision.data.manifest -c configs/data.yaml` | `manifests/{train,test}_videos.csv` |
| Extract frames | `python -m deepvision.data.extract -c configs/data.yaml` | `clips/<split>/<label>/<generator>/<video>.tar` |
| Pack shards | `python -m deepvision.data.shards -c configs/data.yaml` | `shards/<split>/<split>-00000.tar`, `shards/<split>_index.csv` |
| Upload | `python -m deepvision.data.drive push -c configs/data.yaml` | `<drive.remote>/shards/`, `<drive.remote>/manifests/` |

Every step is resumable: re-running skips work that is already done.

### Frame sampling

Frames are sampled with ffmpeg `-r <rate>` following the DeMamba preprocessing rule, scaled from
8 to 16 frames: `rate = 2` fps if the video is at least 8 s long, otherwise `ceil(16 / duration)`.
A centered window of 16 consecutive sampled frames is stored per video (fewer if the video is
too short; no padding is written to disk).

With `extraction.output: clips` each video is stored as one tar of 16 JPEGs, already
center-cropped (DeMamba ratios) and resized to 224x224. Set `output: frames` to keep
native-resolution frame folders instead.

### Loading

`deepvision.data.dataset.FrameClipDataset` reads the stored frames, picks `num_frames` frames
evenly spaced over the stored window, and applies the DeMamba transform
(center crop by percentage, `Resize(224, 224)`, ImageNet normalization).

### Google Drive

Shards (~1 GB, 3000 videos each, in balanced priority order) are uploaded with
[rclone](https://rclone.org/drive/). Configure a Drive remote once (`rclone config`, name it as in
`drive.remote`), then run the `drive push` step; uploaded shards are removed locally after rclone
has verified them.

On a server the dataset can be read straight from Drive without storing it:

```python
from deepvision.data.drive import stream_shard

for key, frames, meta in stream_shard("gdrive:DeepVision/dataset_v1", "shards/train/train-00000.tar"):
    ...  # frames: 16 JPEG byte strings (224x224), meta: dict from meta.json
```

`shards/<split>_index.csv` lists which shard holds each video.

## Tests

```bash
pytest
```
