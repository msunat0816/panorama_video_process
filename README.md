# Panorama Video Processing

This repo processes a 360-degree panorama video to the one first looks at one
target view, then smoothly turns away, then turns back smoothly to the same target view.

## Test Data

The test videos in `test_data/` are first 3 panorama videos from:

```text
https://huggingface.co/datasets/quchenyuan/360x_dataset_LR/tree/main
```

Supported video extensions:

```text
.mp4, .mov, .m4v, .mkv, .avi, .webm
```

## Main Logic

The implemented mode is `two_turn`.

For each source video, the script creates this camera path:

```text
target view
  -> smooth turn to another angle
  -> hold the away view
  -> smooth turn back to the target view
  -> final target view
```

More specifically:

1. The target yaw is sampled around a side view.
   - `--side left` uses base yaw `-90`.
   - `--side right` uses base yaw `90`.
   - `--side random` samples either left or right.
   - `--yaw-jitter-deg` adds random jitter around that base yaw.

2. The target FOV is randomly sampled from `--target-fov-x-deg-range`.

3. The away yaw is sampled by adding a large yaw delta to the target yaw.
   - The delta range is controlled by `--away-yaw-delta-deg-range`.
   - The direction is randomly chosen.
   - The default range is `130,150`, so the away view is far from the target.

4. The away FOV is randomly sampled from `--away-fov-x-deg-range`.

5. The turn from target to away uses smoothstep interpolation.
   - Yaw uses smoothstep.
   - FOV also uses smoothstep.
   - The duration is sampled from
     `--turn-away-duration-ratio-range`.

6. The turn from away back to target also uses smoothstep interpolation.
   - The duration is sampled from
     `--turn-back-duration-ratio-range`.

7. The target hold duration and away hold duration are sampled as ratios of the
   source video duration.
   - Initial target hold: `--target-hold-ratio-range`
   - Away hold: `--away-hold-ratio-range`

8. The final target duration is kept at least as long as:
   - `--min-final-target-duration-ratio` of the source video, and
   - `--min-final-target-duration-sec` seconds

All random choices are reproducible with `--seed`. The script also uses the
source video name and variant index when deriving local random samples, so each
video and each variant gets a stable but different camera path.

## Pitch and Roll

`--pitch-deg` and `--roll-deg` can be changed.

In the current implementation, pitch and roll are constant for the whole output
video. Only yaw and FOV change during the two smooth transitions.

## Install

Create and activate a Python environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The script uses:

- `opencv-python` for video reading and writing
- `numpy`
- `equilib` for equirectangular-to-perspective projection

## Quick Start

Run on the included test videos:

```bash
python panorama_memory_probe.py test_data
```

Run on one video:

```bash
python panorama_memory_probe.py test_data/02692940-fd94-4af1-8ebc-577acacc617a.mp4
```

By default, outputs are written to:

```text
outputs/panorama_memory_probes/outputs_<run-timestamp>/
```

For example:

```text
outputs/panorama_memory_probes/outputs_202607211200/
```

Each run writes:

- generated `.mp4` videos
- `<run-timestamp>_metadata.jsonl`
- `<run-timestamp>_errors.jsonl`, only if some videos fail

## Common Commands

Generate one output per input video with the default settings:

```bash
python panorama_memory_probe.py test_data
```

Generate 3 variants per input video:

```bash
python panorama_memory_probe.py test_data \
  --variants-per-video 3
```

Use a fixed timestamp and overwrite existing files from that timestamp:

```bash
python panorama_memory_probe.py test_data \
  --run-timestamp 202607211200 \
  --overwrite
```

Choose target side randomly for each variant:

```bash
python panorama_memory_probe.py test_data \
  --side random
```

Force the target side to the right:

```bash
python panorama_memory_probe.py test_data \
  --side right
```

Make the away turn as large as possible within a custom range:

```bash
python panorama_memory_probe.py test_data \
  --away-yaw-delta-deg-range 150,175
```

Use wider target views and narrower away views:

```bash
python panorama_memory_probe.py test_data \
  --target-fov-x-deg-range 105,115 \
  --away-fov-x-deg-range 80,95
```

Use longer target and away holds:

```bash
python panorama_memory_probe.py test_data \
  --target-hold-ratio-range 0.25,0.35 \
  --away-hold-ratio-range 0.25,0.35
```

Use longer smooth turns:

```bash
python panorama_memory_probe.py test_data \
  --turn-away-duration-ratio-range 0.15,0.20 \
  --turn-back-duration-ratio-range 0.15,0.20
```

Change pitch and roll:

```bash
python panorama_memory_probe.py test_data \
  --pitch-deg 5 \
  --roll-deg 0
```

Write outputs under a different base folder:

```bash
python panorama_memory_probe.py test_data \
  --output-dir outputs/my_probe_run
```

This creates:

```text
outputs/my_probe_run/outputs_<run-timestamp>/
```

Stop immediately when one input video fails:

```bash
python panorama_memory_probe.py test_data \
  --strict
```

## Useful Parameters

| Parameter | Meaning | Default |
| --- | --- | --- |
| `input` | Input video file or directory | required |
| `--output-dir` | Base output folder | `outputs/panorama_memory_probes` |
| `--run-timestamp` | Timestamp used in output names | current time |
| `--seed` | Global random seed | `42` |
| `--mode` | Experiment mode | `two_turn` |
| `--side` | Target side: `left`, `right`, or `random` | `left` |
| `--variants-per-video` | Number of outputs per source video | `1` |
| `--target-hold-ratio-range` | Initial target hold duration range | `0.20,0.25` |
| `--away-yaw-delta-deg-range` | Away yaw delta range in degrees | `130.0,150.0` |
| `--away-hold-ratio-range` | Away hold duration range | `0.20,0.25` |
| `--turn-away-duration-ratio-range` | Target-to-away turn duration range | `0.1,0.15` |
| `--turn-back-duration-ratio-range` | Away-to-target turn duration range | `0.1,0.15` |
| `--min-final-target-duration-ratio` | Minimum final target duration ratio | `0.05` |
| `--min-final-target-duration-sec` | Minimum final target duration in seconds | `1.0` |
| `--yaw-jitter-deg` | Random target yaw jitter around the side yaw | `15.0` |
| `--target-fov-x-deg-range` | Target horizontal FOV range | `100.0,110.0` |
| `--away-fov-x-deg-range` | Away horizontal FOV range | `100.0,110.0` |
| `--pitch-deg` | Constant pitch angle | `0.0` |
| `--roll-deg` | Constant roll angle | `0.0` |
| `--overwrite` | Replace existing outputs with the same names | off |
| `--strict` | Stop on the first failed video | off |

All range arguments use this format:

```text
low,high
```

Do not add spaces inside the range unless your shell handles them safely.

## Metadata

The metadata file is JSON Lines. Each generated video has one JSON record.

Important fields include:

- `source_video`
- `output_video`
- `seed`
- `variant_index`
- `side`
- `target_yaw_deg`
- `away_yaw_deg`
- `away_yaw_delta_deg`
- `target_fov_x_deg`
- `away_fov_x_deg`
- `target_hold_sec`
- `turn_away_duration_sec`
- `away_hold_sec`
- `turn_back_duration_sec`
- `final_target_hold_sec`
- frame boundary fields such as `turn_away_start_frame` and
  `turn_back_end_frame`

Use the metadata when matching generated videos back to their sampled camera
paths.

## Notes

- Output videos keep the same width, height, and FPS as the input video.
- If an input video is not close to `2:1`, the metadata field
  `aspect_ratio_warning` is set to `true`.
- The projection is done with `equilib.equi2pers`.
- Existing output files are not overwritten unless `--overwrite` is passed.
