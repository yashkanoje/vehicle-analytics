# vehicle-analytics

Count vehicles and measure their speed, with an error bar, from fixed-camera video.
Runs on your own machine; once the model weights have been downloaded it needs no network.

![Annotated demo](docs/demo.gif)

*Demo footage: "Motorway A40: on bridge above the traffic" by Kathinka Engels,
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0), annotated by this software —
details in [docs/SAMPLE_VIDEO.md](docs/SAMPLE_VIDEO.md). This scene is not calibrated: its
speeds use an estimated distance and are for illustration only.*

## What it does

- Detects vehicles with YOLO and follows them with DeepSORT.
- Counts a vehicle, by direction and class, when its track crosses a line. A track is
  counted at most once.
- Measures speed between two lines, either from a known distance or from a ground-plane
  calibration, and reports a one-sigma uncertainty with every speed.
- Writes an annotated video, a CSV with one row per vehicle, and a JSON summary.

## How it works

```
frame → YOLO detections → DeepSORT tracks → bottom-centre point per vehicle
      → line crossings (sub-frame) → speed ± error → counts and flow → files
```

**Crossing.** A vehicle crosses a line when the signed distance from its reference point to
the line changes sign between two observations, however far apart they are. The crossing
point is where the straight path between them meets the line. The crossing instant assumes
constant speed between the two observations: on the ground plane when the scene is
calibrated, and in the image (`t = d_prev / (d_prev − d_cur)`) when it is not. In the image
that assumption is biased by perspective when the tracker has missed many frames.

**Reference point.** The bottom-centre of the box, where the vehicle meets the road. That is
the plane the calibration describes.

**Speed.** `distance / |t_b − t_a|` using the interpolated crossing times. Without a
calibration, `distance` is the configured distance between the lines. With one, both crossing
points are projected to the ground plane and the distance is measured there.

**Uncertainty.** Two independent relative terms, combined in quadrature:

| Term | Formula | Default |
|---|---|---|
| Box localisation, calibrated | `√(s_a² + s_b²) / distance` | `jitter_px = 2` |
| Box localisation, not calibrated | `√(p_a² + p_b²) / travel_px` | `jitter_px = 2` |
| Real-world scale | `distance_rel_uncertainty` | `0.05` |

A box-edge error of `jitter_px` across a line moves the crossing along the vehicle's path by
`p = jitter_px / cos θ` pixels, where θ is the angle between the path and the line's normal:
a vehicle that meets the line at a slant is timed less precisely. `p_a` and `p_b` are those
shifts at the two lines, and `s_a` and `s_b` are the same shifts in metres on the ground.
The two differ because a pixel covers more road far from the camera; on the motorway
footage below one pixel is about 0.8 m at the far line and 0.2 m at the near one. Without a
calibration the image scale is taken as uniform between the lines.

Both inputs are stated assumptions, set in the config. The error bar does not grow with the
number of frames the tracker missed around a crossing.

**Flow rate.** Vehicles per hour over the video time elapsed, `frame_index / fps`.

## Install

Python 3.10 or newer.

```bash
pip install -e .
```

YOLO weights download on first run.

## Use

```bash
vehicle-analytics run VIDEO --config configs/highway.yaml
vehicle-analytics run VIDEO --config configs/highway.yaml --out runs/my-run --no-display
```

Outputs go to `runs/<video name>/`:

| File | Content |
|---|---|
| `annotated.mp4` | Boxes, IDs, class, speed ± error, both lines, live counts |
| `events.csv` | `track_id, class, direction, counted_at_s, speed_kmph, speed_err_kmph, speed_mode` |
| `summary.json` | Counts, flow, speed mean / median / max by direction, config used |

## Configure

See [configs/highway.yaml](configs/highway.yaml). All image coordinates are fractions of the
frame, so a config does not depend on resolution.

| Key | Meaning |
|---|---|
| `lines.a`, `lines.b` | The two line segments |
| `count_line` | Which line counts a vehicle |
| `directions` | Labels for the two crossing directions |
| `speed.meters_between_lines` | Ground distance between the lines, when not calibrated |
| `calibration` | Four image points and their ground coordinates in metres |
| `fps_override` | Frame rate, for files that do not report one |

## Calibrate

Pick a rectangle on the road whose real size you know, such as one lane wide and a known
number of lane markings long.

```bash
vehicle-analytics calibrate VIDEO --width-m 3.5 --length-m 12
```

Click its corners in this order: near-left, near-right, far-right, far-left. Paste the
printed block into the config. `--points "x,y x,y x,y x,y"` gives the corners without a window.

Use a rectangle that covers the part of the road you measure on. Four corners of one narrow
lane describe that lane well and the rest of the road badly: on the footage below, a single
3.70 m lane gave the right scale on its own carriageway but measured the opposite
carriageway 8 % short and 15 % narrow. The calibration in `configs/highway.yaml` is
therefore a least-squares fit of one ground plane to 21 lane marks across both
carriageways, with its four point pairs taken from that fitted plane. The comments in that
file give the method and the numbers.

## Results

Measured on a 60-second stretch (1501 frames) of UK motorway footage, 640×360 at 25 fps,
with `configs/highway.yaml`. The footage is third-party and is not included in this
repository.

| Direction | Traffic | Counted | With speed | Median speed | Range |
|---|---|---|---|---|---|
| down (towards camera) | free-flowing | 83 | 76 | 94 km/h | 77–111 km/h |
| up (away from camera) | queuing | 75 | 66 | 37 km/h | 31–46 km/h |

The error bar on each speed is ±7.4 %: ±5 % from the scale and ±5.5 % from box localisation,
most of it at the far line. Vehicles counted without a speed crossed only
one of the two lines under a single track ID, or were still between the lines when the clip
ended.

Counts against a manual count of the same clip:

| Direction | Manual count | System count | Difference |
|---|---|---|---|
| down (towards camera) | 83 | 83 | 0 |
| up (away from camera) | 74 | 75 | +1 |

The manual count was made once, by the author, on the annotated video at half speed, so it
was not blind to the system's boxes. It does not say whether the extra vehicle in the
queuing direction is a double count by the system or a miss by the person. One 60-second
clip in daylight shows that the counts are close here; it is not an accuracy figure.

Calibration, using lane-marking dimensions from UK design standards (lane line 2 m mark +
7 m gap; middle lane 3.70 m, outer lanes 3.65 m — sources in `configs/highway.yaml`):

| Check | Expected | Measured |
|---|---|---|
| Reprojection error of the 21 fitted marks | — | 1.0 px RMS, 1.9 px max |
| Length of the 36 m zone, per lane line (4 lines, used in the fit) | 36.0 m | 35.2–36.7 m |
| Lane line to edge line (4 sides, not used in the fit) | about 3.75 m | 3.62–3.93 m |

Only the edge-line row is independent of the fit. The lengthwise scale, which is what a
speed depends on, is supported by the consistency of the fit and not by a separate
measurement. The road was not surveyed: the dimensions are those of the current standard.
The measured mark positions and the fit are in `tools/fit_highway_calibration.py`, which
prints every figure in this table.

There is no ground truth for speed in this footage, so no speed accuracy is claimed. Each
speed carries the error bar described above.

## Limitations

- Fixed camera only.
- Occlusion and tracker ID switches: a vehicle whose ID changes between the two lines is
  still counted once, but has no speed.
- Speed accuracy depends on the calibration dimensions, which are taken from road standards,
  not surveyed.
- The detector is a general COCO model; vans are sometimes labelled as trucks.
- Not real time: the 1501-frame clip above took about eight minutes on the development
  laptop's GPU, model loading included. The pipeline has not been profiled.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The geometry, crossing, speed, state and statistics modules are pure Python and numpy. The
pipeline takes its detector and tracker as arguments, so its tests run on scripted tracks
with no model and no video.

## Roadmap

Per-vehicle records in SQLite, rule-based violation detection with photo evidence, and a
local dashboard.

## Licence

AGPL-3.0-or-later, to match Ultralytics YOLO.
