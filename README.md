# Real-Time Hand Tracking and Latency Optimization in Vision-Arm Teleoperation

A vision-based robot-arm teleoperation system that predicts the operator's
next hand/arm state to compensate for capture-to-actuation latency, and
reconstructs 3D depth from the robot's own kinematics instead of relying on
a single camera's native depth estimate. Built on ROS2 and simulated in
Gazebo against a robotics arm.

## Pipeline

```
Camera (Orbbec Astra Pro RGB-D)
  -> MediaPipe Hands (21 landmarks) / MediaPipe Pose (33 landmarks)
  -> Preprocessing (normalise, 30-frame sliding window)
  -> Temporal prediction (LSTM / GRU / Transformer, predicts next-frame pose)
  -> Kinematics-based depth reconstruction (IK/FK over the robot's URDF)
  -> ROS2 joint-command publisher -> robot arm in Gazebo
```

Two parallel tracks share this design: a **hand/gripper track** (MediaPipe
Hands, drives the gripper) and an **arm track** (MediaPipe Pose, drives
shoulder/elbow/forearm).

## Repository Structure

```
src/robot_arm/
  config/            ROS2 controller, MoveIt and kinematics config
  launch/            Gazebo, RViz, MoveIt and motion-control launch files
  meshes/, urdf/     PAROL6 arm description
  scripts/
    perception/      Camera capture + MediaPipe landmark extraction
    models/          LSTM / GRU / Transformer predictor architectures
    robot_control/   Inverse kinematics, joint-command mapping
    data_collection/ Recording + depth calibration tools
    dataset_ipynb/   Notebooks: dataset build -> training -> benchmarking
    evaluation/       Inference-time, latency, RMSE and MPJPE benchmarks
docs/
  DATA.md            What data/weights are excluded and how to regenerate them
  FINDINGS.md        Bugs and dead ends found during the project, and their cost
run_arm.sh           One-command launch: Gazebo + perception + live control
```

## Requirements

- Ubuntu with ROS2 Humble, Gazebo (`ros_gz`), MoveIt
- Python 3.10, with `mediapipe`, `torch`, `ikpy`, `numpy`, `opencv-python`
- An RGB-D camera (Orbbec Astra Pro used here) for live capture/recording

No pinned `requirements.txt` is included yet; the above are the libraries the
scripts import directly.

## Build & Run

```bash
cd src/robot_arm/.. && colcon build
./run_arm.sh                  # build + launch, camera window on
./run_arm.sh --no-build       # skip the colcon build
./run_arm.sh --direct         # bypass the predictor, direct landmark->joint mapping
./run_arm.sh --pose-predictor # opt-in: enable the trained predictor for latency compensation
./run_arm.sh --stop           # kill every node this pipeline started
```

See the header of `run_arm.sh` for the full flag list.

## Data and Trained Models

Recordings, datasets (`.npz`) and trained weights (`.pt`) are **not** tracked
in this repository — they're large, regenerable binaries. See
[docs/DATA.md](docs/DATA.md) for what's excluded and how to rebuild them from
scratch (record -> calibrate depth -> run the `dataset_ipynb/` notebooks in
order).

## Headline Results

| Metric | Result |
|---|---|
| Inference time | GRU fastest on both tracks (0.272 ms), not LSTM as hypothesised |
| End-to-end latency (predictor off) | 470 ms baseline, directly measured (cross-correlation, r = 0.73) |
| End-to-end latency (predictor on) | Could not be resolved — predictor's own reconstructed motion doesn't yet track the operator |
| Depth reconstruction (RMSE) | Kinematics 367.8 mm, MediaPipe native z 372.9 mm — neither beats a 384.2 mm mean-guess baseline |
| Prediction accuracy (MPJPE) | No stable winner — GRU best on hand track, Transformer best on arm track; the split reversed between two independent recording rounds |

These are honestly-reported negative/mixed results with proper controls
(shuffled and predict-the-mean baselines) rather than reframed positives —
see [docs/FINDINGS.md](docs/FINDINGS.md) for what was tried, what broke, and
why.

## License

BSD, per `src/robot_arm/package.xml`.
