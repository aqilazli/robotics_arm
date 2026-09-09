# Findings

Things this project got wrong, and what it cost to find out. Recorded because
each was invisible in the code and only showed up in measurement.

## The depth sensor was streaming the wrong format

The Orbbec Astra Pro advertises its depth stream as **both** `Y11` and `Y12`.
The pipeline read `Y12`. It is not distance:

| | Y12 | Y11 |
|---|---|---|
| Distinct values in a 640×480 frame | ~9 | ~115 |
| Scene structure | none | smooth gradient matching geometry |
| Monotonic with distance | **no** | yes |

Measured against a tape measure, `Y12` gave 300 mm → 55, 400 mm → 46.5,
500 mm → 38, 600 mm → 42. A sensor cannot report the same value for two
distances. `Y11` is the real stream and its units are centimetres.

**Cost:** an affine correction was nearly adopted for `Y12`. Two calibration
points fitted it perfectly — because *any* two points fit a line. The third
and fourth points killed it. Always take 3+.

**Consequence:** every recording made before this carries unusable depth.

## The depth ground truth was random

`sample_ground_truth()` drew **random joint angles** and ran forward
kinematics, never looking at the footage. Sample *i* had no relationship to
frame *i*:

| Predictions fed in | RMSE-kin |
|---|---|
| Real | 1206.009 |
| Shuffled | 1205.255 |
| **All zeros** | **122.861** |

Shuffling changed the score by 0.06%, and predicting nothing scored ten times
better than a trained model. Both RMSE columns were withdrawn. Replaced by
`evaluation/depth_ground_truth.py`, which uses calibrated depth measured in
the same frame as the landmarks.

## joint_state_broadcaster was misreporting positions

Given both a `joints` list and an `interfaces` list, it enters a
custom-mapping mode and warns:

```
Mapping from 'position' to interface 'position' will not be done,
because 'position' is defined in 'interface' parameter
```

`/joint_states` then disagreed with the controller's own feedback for the same
joints at the same instant. That is what RViz draws and what every diagnostic
reads, so several conclusions drawn from it were wrong. Removing both lists
fixed it — left alone the broadcaster publishes whatever the hardware declares.

## The controller ignores gripper-only trajectories

`joint_trajectory_controller` executes a trajectory only when some **arm**
joint actually changes:

```
arm already at 0, gripper 0 → 0.011   never moves  (1 point or 2, 40ms or 2s)
same gripper change + any arm motion  both jaws reach 0.011
```

This defeated four separate attempts to park the gripper open at startup —
`initial_value` in the URDF, `load_controllers._send_home()`, a startup
assertion in `robot_node`, and a dedicated script — all of which reported
success while nothing moved. `open_gripper_startup.py` now nudges L1 by
0.05 rad to carry the gripper command through, then returns it.

## MoveIt executed nothing

MoveIt plans per **planning group** — 6 joints for `arm`, 2 for `gripper` —
while the controller owns 8 and `allow_partial_joints_goal` was `false`. Every
goal was rejected on arrival. Planning worked, because planning never touches
the controller, so it looked like execution alone was broken.

## The arm moved with nobody in front of the camera

Two causes compounding:

1. `robot_node` stamped the pose predictor's output `visibility = 1.0`, so once
   the predictor was active the visibility check could never fail. The arm
   chased predictions made from nothing.
2. MediaPipe's `visibility` is optimistic. With the operator's arm out of shot,
   medians were shoulder **0.992** (real) but elbow **0.356** and wrist
   **0.236** — invented, peaking at 0.746. The old 0.3 threshold passed 13 of
   74 such frames.

Gating on the *real* landmarks at `ARM_CONTROL_MIN_VISIBILITY = 0.85` cut idle
trajectories from 130 to 13, then to none.

## The gripper was one rigid mesh

The exported `L6.STL` had both jaws baked in — visible, immovable. Before this
was noticed, hand-openness was mapped onto `joint_L6`, so **opening your hand
spun the wrist**. It looked like it worked, because something moved.

Analysis showed `L6.STL` was seven separate shells, two of them 336-triangle
jaws mirrored about y. Extracting them into `L7_R_jaw.STL` and `L7_L_jaw.STL`
with their own prismatic joints made the gripper actually open and close, and
freed L6 to be a genuine wrist roll.

---

**The pattern worth carrying forward:** in every case a component reported
confidence it had not earned — a sensor reporting values that were not
distance, a metric scoring random data, a broadcaster publishing wrong
positions, a predictor asserting visibility, a controller reporting success
while doing nothing. Measure the output, not the intent.
