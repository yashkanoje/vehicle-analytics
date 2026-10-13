# Demo footage

The animation in the README and `configs/demo.yaml` use this clip.

| | |
|---|---|
| Title | Motorway A40: on bridge above the traffic |
| Author | Kathinka Engels |
| Source | <https://commons.wikimedia.org/wiki/File:Motorway_A40_-_on_bridge_above_the_traffic.webm> |
| Original publication | <http://www.soundsofchanges.eu/sound/motorway-a40-on-bridge-above-the-traffic> |
| Licence | [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0) |
| Downloaded | 13 October 2026 |

Changes made: the clip was shortened, scaled down and converted to GIF, with detection
boxes, track labels, counting lines and a statistics panel drawn on it by this software.

The clip itself is not stored in this repository. To reproduce the demo, download it from
the source above and run:

```bash
vehicle-analytics run "Motorway_A40_-_on_bridge_above_the_traffic.webm" --config configs/demo.yaml
```

The footage used to develop and validate the calibrated configuration
(`configs/highway.yaml`) is third-party material and is not distributed here.
