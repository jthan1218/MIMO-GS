# Rendering Spatial MIMO Channel Scenes: A 3D Gaussian Splatting Approach

This repository implements MIMO Gaussian splatting (MIMO-GS) to learn a spatial channel scene from channel observations and render joint receive–transmit beam pair power maps at user locations. It includes scene training, a PyTorch renderer, an optional CUDA rasterizer, and evaluation on held-out locations.

**Abstract**

The gains of multiple-input multiple-output (MIMO) systems depend on channel state information (CSI), whose acquisition consumes pilot and feedback resources for every user in every coherence interval, while the propagation environment that shapes the channel persists. Exploiting this persistence, however, requires capturing how the surrounding geometry determines the channel at locations where no measurement has been taken, which no compact channel model provides. We formulate spatial MIMO channel rendering as learning a latent beamspace channel scene from channel observations, which renders the long-term joint transmit and receive beam pair power map at unobserved user locations while preserving the coupled angular structure of the MIMO channel. Motivated by 3D Gaussian splatting in visual scene rendering, the proposed MIMO Gaussian splatting (MIMO-GS) represents the scene with primitives that pair a receive and a transmit 3D Gaussian. The two Gaussians are projected from the user location and the base station onto the corresponding beam domains, and the beam associations are accumulated through a beam pair splatting rule with a gain conditioned on the user location. The rendering is validated on a simulated urban scenario and on indoor channels measured over the air, where MIMO-GS reproduces the dominant beam pair structure at unobserved locations. It renders maps 5.5 times faster than ray tracing and requires channel observations alone rather than a calibrated replica of the environment. Applied to multiuser beam management, the rendered maps allow users and transmit beams to be scheduled without sounding while the inter-user interference is taken into account, and the net sum rate reaches 98% of the zero-overhead benchmark and exceeds a position-based beam predictor by 24%.

## Installation

The supplied environment specifies Python 3.10, PyTorch 1.13.1, and CUDA 11.6. The code has been run with Python 3.10.8, PyTorch 1.13.1, and CUDA 11.6. From the repository root:

```bash
conda env create -f environment.yml
conda activate mimogs
```

To build and install the CUDA rasterizer, use a CUDA 11.6 development toolkit with `nvcc`, a C++ compiler, and an NVIDIA GPU. The environment's `pytorch-cuda` dependency supplies the runtime; the development toolkit is a separate prerequisite.

```bash
python -m pip install --no-build-isolation -v ./mimogs_rasterizer
```

The renderer falls back to its PyTorch implementation if the extension is unavailable.

## Dataset

[`dataset/asu_campus_16by256_lt/`](dataset/asu_campus_16by256_lt/) contains long-term beam pair power maps generated from the DeepMIMO `asu_campus_3p5` scenario, split into training and test locations:

| File | Contents |
| --- | --- |
| `train.mat` | Float32 `positions` of shape `(5792, 3)` in meters and `magnitude` of shape `(5792, 16, 256)`. |
| `test.mat` | Float32 `positions` of shape `(1448, 3)` and `magnitude` of shape `(1448, 16, 256)`; `nn_dist_m` of shape `(1448, 1)` stores distance to the nearest training location in meters and is unused by the loader. |
| `bs_info.yml` | Dataset identifier, normalized base-station position, orientation, and notes on dataset selection, normalization, and array geometry. |

The map axes are `(location, receive beam, transmit beam)`: a receive UPA of shape `(4, 4)` has 16 beams, and a transmit UPA of shape `(16, 16)` has 256 beams. Array shapes use `(horizontal, vertical)` ordering.

For your own channel observations, supply a directory with `train.mat`, `test.mat`, and `bs_info.yml`. Both MAT files must be readable by `scipy.io.loadmat` and contain `positions` with shape `(N, 3)` and nonnegative, linear-scale beam pair powers under the key `magnitude`, with shape `(N, Nr, Nt)`. Use the same beam dimensions for both splits. The YAML file must provide `bs1.position` and `bs1.orientation` as three-element lists; `dataset_name: mimo` selects the generic MAT loader.

<!-- The loader divides each split's positions by its own `max(abs(positions)) + 1e-6`. Ensure these normalization factors match across splits, and supply the BS position in that normalized frame. Training normalizes each target power map by its maximum. The renderer assumes panels in the y-z plane, with horizontal +y and vertical +z, and unshifted DFT beam ordering with the horizontal index varying fastest. Set `--rx_shape_h`, `--rx_shape_v`, `--tx_shape_h`, and `--tx_shape_v` for non-square arrays; their defaults are `0` (infer from beam counts). -->

## Training

Run from the repository root; no arguments are required:

```bash
python train.py
```

Common options are:

| Argument | Default | Purpose |
| --- | --- | --- |
| `--source_path` | `./dataset/asu_campus_16by256_lt` | Dataset directory. |
| `--model_path` | Empty; creates `outputs/YYYYMMDD_HHMMSS/` | Output directory. |
| `--num_epochs` | `100` | Training epochs. |
| `--batch_size` | `8` | User locations per training batch. |
| `--target_gaussians` | `10000` | Number of paired Gaussian primitives. |
| `--max_active_rx_beams` / `--max_active_tx_beams` | `8` / `8` | Retained beams per primitive on each side. |
| `--use_cuda_rasterizer` | `1` | Request the CUDA rasterizer; use `0` for the PyTorch path. |

Training saves `model.pth`, `run_args.txt`, and `point_cloud/point_cloud.ply`. It also saves 50 test comparison figures as `pred_compare/00.png` through `49.png` (or fewer if the test set is smaller), using dB power with a shared 50 dB display range per comparison, and prints full-test-set NMSE metrics.

## Pretrained scene

[`outputs/sample/model.pth`](outputs/sample/model.pth) contains a scene trained on the included DeepMIMO ASU campus dataset, with **25,000 paired Gaussian primitives**. It stores scene parameters, the location-conditioned gain network, optimizer states, and the run's model and optimization settings. Example dB comparisons are in [`outputs/sample/pred_compare/`](outputs/sample/pred_compare/).

## Repository layout

- `arguments/` — Command-line parameter groups and defaults.
- `dataset/` — Included DeepMIMO data and BS metadata.
- `gaussian_renderer/` — Geometric projection and beamspace rendering.
- `mimogs_rasterizer/` — PyTorch reference rasterizer, CUDA sources, and extension build configuration.
- `outputs/` — Pretrained scene and example comparison figures.
- `scene/` — Dataset loading, scene management, and Gaussian model.
- `utils/` — Losses, numerical helpers, logging, and filesystem utilities.
- `train.py` — Training and post-training evaluation entry point.
- `environment.yml` — Conda environment dependencies.
- `.gitignore` — Exclusions for generated and local files.
- `README.md` — Project overview and usage.
- `LICENSE.md` — License terms.

## License

See [LICENSE.md](LICENSE.md) for license terms.
