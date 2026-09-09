#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_arm.sh  —  One click: Gazebo + gesture control + robot arm
#
# Pipeline:
#   Camera → OpenCV → MediaPipe → FeatureExtractor → Preprocessor
#   → (LSTM gesture classifier, or direct landmark mapping) → ROS 2 → robot arm
#
# Usage:
#   ./run_arm.sh               # build + run (camera window ON by default)
#   ./run_arm.sh --no-build    # skip colcon build
#   ./run_arm.sh --hide        # hide OpenCV camera window
#   ./run_arm.sh --no-hw       # disable hardware serial (dry-run)
#   ./run_arm.sh --direct      # bypass gesture-classification LSTM, use
#                                 # direct landmark→joint mapping
#   ./run_arm.sh --freeze=joint_L1,joint_L2,...  # debug: hold exactly the
#                                 # listed joints, everything else stays
#                                 # live. Isolates one joint at a time when
#                                 # troubleshooting -- not for normal use.
#   ./run_arm.sh --freeze-l1-l5  # shorthand: freeze L1-L5, isolate L6/L7
#   ./run_arm.sh --freeze-l1-l4  # shorthand: freeze L1-L4, isolate L5-L7
#   ./run_arm.sh --pose-predictor  # opt-in: enable the trained pose predictor
#                                 # for L1-L4 latency compensation. OFF by
#                                 # default -- see the block below for why.
#   ./run_arm.sh --stop        # kill every node this pipeline runs, then exit
#
# Opens one gnome-terminal window with one tab per node, same as always --
# run this yourself and watch the tabs. Reported directly: "i want to run
# run_arm.sh alone to open up all like before" -- a detached/background
# version of this (no visible windows) was tried and didn't fit that.
# ─────────────────────────────────────────────────────────────────────────────

WS=/home/user_02/Desktop/ros2_ws
ROS_SETUP=/opt/ros/humble/setup.bash
SCRIPTS=$WS/src/robot_arm/scripts
DO_BUILD=true
SHOW=true        # camera window ON by default
NO_HW=false
DIRECT=false
FREEZE_JOINTS=""
ENABLE_POSE_PREDICTOR=false
DO_STOP=false

for arg in "$@"; do
  case $arg in
    --no-build)        DO_BUILD=false ;;
    --hide)            SHOW=false     ;;
    --no-hw)           NO_HW=true     ;;
    --direct)          DIRECT=true    ;;
    --freeze-l1-l5)    FREEZE_JOINTS="joint_L1,joint_L2,joint_L3,joint_L4,joint_L5" ;;
    --freeze-l1-l4)    FREEZE_JOINTS="joint_L1,joint_L2,joint_L3,joint_L4" ;;
    --freeze=*)        FREEZE_JOINTS="${arg#--freeze=}" ;;
    --pose-predictor)  ENABLE_POSE_PREDICTOR=true ;;
    --stop)            DO_STOP=true ;;
  esac
done

NODE_PATTERN="ign gazebo server|robot_control/robot_node\.py|scripts/arm_gui\.py|perception/inference_node\.py|robot_control/hardware_interface\.py"

# ── --stop: kill every node this pipeline runs, then exit ─────────────────────
# Matches by process pattern, not a saved PID file -- works the same whether
# a given run was started from a terminal tab (this script's normal mode) or
# any other way, and there is nothing to go stale if the terminal was closed.
if [ "$DO_STOP" = true ]; then
  MATCHES=$(ps -eo pid,cmd --no-headers | grep -E "$NODE_PATTERN" | grep -v grep)
  if [ -n "$MATCHES" ]; then
    echo ">>> Stopping:"
    echo "$MATCHES" | sed 's/^/    /'
    echo "$MATCHES" | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null
    sleep 2
    echo "$MATCHES" | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null
    echo ">>> Stopped."
  else
    echo ">>> Nothing matching this pipeline appears to be running."
  fi
  exit 0
fi

# ── refuse to double-launch -- this is what caused "crazy movement" once ──────
ALREADY=$(ps -eo pid,cmd --no-headers | grep -E "$NODE_PATTERN" | grep -v grep)
if [ -n "$ALREADY" ]; then
  echo ""
  echo "  [REFUSING TO LAUNCH] The rig already looks like it's running:"
  echo "$ALREADY" | sed 's/^/    /'
  echo ""
  echo "  Launching again on top of this creates TWO complete copies fighting"
  echo "  over the same simulated robot -- that is exactly what 'crazy"
  echo "  movement' turned out to be once already. Run:"
  echo "      ./run_arm.sh --stop"
  echo "  first, then relaunch."
  echo ""
  exit 1
fi

# ── auto-detect: no LSTM model → fall back to direct landmark mapping ─────────
MODEL=$SCRIPTS/models/lstm_predictor.pt
if [ ! -f "$MODEL" ] && [ "$DIRECT" = false ]; then
  echo ""
  echo "  [WARNING] LSTM model not found: $MODEL"
  echo "  Falling back to --direct mode (landmark → joint mapping, no LSTM)."
  echo "  To use LSTM: train the model first with lstm_model.py --train"
  echo ""
  DIRECT=true
fi

# ── pose predictor for L1-L4 latency compensation -- OFF by default ───────────
# Was auto-enabled whenever a trained pose predictor file was found. Proven
# live this session (fed a real, held-steady pose with no gesturing) to
# actively FABRICATE motion, not just be occasionally inaccurate -- and its
# training data used depth-fused landmark coordinates, which the live
# pipeline no longer produces for the pose/arm-control path (removed
# separately, see motion_mapping.py/inference_node.py history), so even a
# retrained version would need re-validating before this default should
# change. Pass --pose-predictor to opt in anyway.
POSE_MODEL=""
POSE_MODEL_TYPE="lstm"
if [ -f "$SCRIPTS/models/pose_lstm_predictor.pt" ]; then
  POSE_MODEL=$SCRIPTS/models/pose_lstm_predictor.pt
  POSE_MODEL_TYPE="lstm"
elif [ -f "$SCRIPTS/models/pose_gru_predictor.pt" ]; then
  POSE_MODEL=$SCRIPTS/models/pose_gru_predictor.pt
  POSE_MODEL_TYPE="gru"
elif [ -f "$SCRIPTS/models/pose_transformer_predictor.pt" ]; then
  POSE_MODEL=$SCRIPTS/models/pose_transformer_predictor.pt
  POSE_MODEL_TYPE="transformer"
fi
if [ -n "$POSE_MODEL" ] && [ "$ENABLE_POSE_PREDICTOR" = false ]; then
  echo ">>> Pose predictor file found ($POSE_MODEL_TYPE) but NOT enabled -- known to fabricate"
  echo "    motion (see comment above). Pass --pose-predictor to use it anyway."
  POSE_MODEL=""
fi
if [ -n "$POSE_MODEL" ] && [ "$ENABLE_POSE_PREDICTOR" = true ]; then
  echo ">>> Pose predictor ENABLED by --pose-predictor ($POSE_MODEL_TYPE) — L1-L4 will use predicted pose."
fi

# ── build ─────────────────────────────────────────────────────────────────────
if [ "$DO_BUILD" = true ]; then
  echo ">>> Building …"
  source "$ROS_SETUP"
  cd "$WS"
  colcon build --packages-select robot_arm --symlink-install
  if [ $? -ne 0 ]; then echo ">>> Build FAILED."; exit 1; fi
  echo ">>> Build done."
fi

# ── write temp scripts ────────────────────────────────────────────────────────
TMPD=$(mktemp -d)

# Terminal 1 — Gazebo simulation
cat > "$TMPD/gazebo.sh" << SCRIPT
#!/bin/bash
source $ROS_SETUP
source $WS/install/setup.bash
ros2 launch robot_arm gazebo.launch.py
exec bash
SCRIPT

# Terminal 2 — ML inference node (camera → MediaPipe → LSTM → /gesture_command)
cat > "$TMPD/inference.sh" << SCRIPT
#!/bin/bash
source $ROS_SETUP
source $WS/install/setup.bash
echo ""
echo "  Waiting 20s for Gazebo + controllers..."
sleep 20
echo "  Starting ML inference node (LSTM gesture pipeline)..."
python3 $SCRIPTS/perception/inference_node.py \
  --ros-args -p show_window:=$SHOW
exec bash
SCRIPT

# Terminal 3 — Robot node (/gesture_command → /joint_trajectory)
# Build the --ros-args list here rather than interpolating variables into the
# command, because ROS rejects an override with an empty value outright:
#
#   -p freeze_joints:=      ->  "Couldn't parse parameter override rule"
#
# and that kills robot_node at rclpy.init(), before it can log a single line.
# Gazebo, RViz, the camera window and the Arm GUI all still come up, so the
# rig looks completely healthy while nothing whatsoever drives the arm.
# This already happened once with pose_predictor_path and was fixed inline;
# adding freeze_joints reintroduced the same bug, so both now go through the
# one guarded path below. Only append a flag when it actually has a value.
ROBOT_ARGS="-p use_direct_map:=$DIRECT"
if [ -n "$POSE_MODEL" ]; then
  ROBOT_ARGS="$ROBOT_ARGS -p pose_predictor_path:=$POSE_MODEL"
  ROBOT_ARGS="$ROBOT_ARGS -p pose_predictor_type:=$POSE_MODEL_TYPE"
fi
if [ -n "$FREEZE_JOINTS" ]; then
  ROBOT_ARGS="$ROBOT_ARGS -p freeze_joints:=$FREEZE_JOINTS"
fi

cat > "$TMPD/robot.sh" << SCRIPT
#!/bin/bash
source $ROS_SETUP
source $WS/install/setup.bash
echo ""
echo "  Waiting 22s for inference node..."
sleep 22
echo "  Starting robot node..."
if [ -n "$FREEZE_JOINTS" ]; then
  echo "  [DEBUG] freeze active: $FREEZE_JOINTS will hold, everything else stays live."
fi
if [ -z "$POSE_MODEL" ]; then
  echo "  [!] No pose predictor active -- running WITHOUT latency compensation"
  echo "      (this is the default; pass --pose-predictor to opt in)."
fi
echo "  args: $ROBOT_ARGS"
python3 $SCRIPTS/robot_control/robot_node.py --ros-args $ROBOT_ARGS
RC=\$?
# If robot_node exits, say so loudly instead of dropping to a silent prompt --
# a dead node here looks exactly like a working rig that ignores you.
echo ""
echo "  ############################################################"
echo "  #  robot_node EXITED (code \$RC) -- THE ARM WILL NOT MOVE.  #"
echo "  #  The traceback above is the reason. Nothing else in the  #"
echo "  #  system will report this fault.                          #"
echo "  ############################################################"
exec bash
SCRIPT

# Terminal 4 — Hardware interface (/joint_trajectory → serial → Arduino)
cat > "$TMPD/hardware.sh" << SCRIPT
#!/bin/bash
source $ROS_SETUP
source $WS/install/setup.bash
echo ""
echo "  Waiting 23s for robot node..."
sleep 23
echo "  Starting hardware interface..."
python3 $SCRIPTS/robot_control/hardware_interface.py \
  --ros-args -p enabled:=$([ "$NO_HW" = true ] && echo false || echo true)
exec bash
SCRIPT

# Terminal 5 — Arm GUI (manual joint control / AI model toggle, no camera)
# GUI_ARGS mirrors ROBOT_ARGS' freeze_joints handling above -- same reason:
# an empty -p override crashes rclpy at startup ("Couldn't parse parameter
# override rule"), so only append the flag when FREEZE_JOINTS is non-empty.
# Without this, --freeze=... only ever reached robot_node.py: the GUI's own
# ArmNode came up with its checkboxes defaulting to all-unfrozen and its 2s
# _reassert_freeze heartbeat then overwrote robot_node's command-line freeze
# with that empty state within 2 seconds, silently undoing the flag every
# single time. Reported directly, after asking for "--freeze=joint_L4,
# joint_L6" to isolate L5 for tuning and getting L4/L6 still live: verified
# via /freeze_status reading '' (nothing held) despite the flag being passed.
GUI_ARGS=""
if [ -n "$FREEZE_JOINTS" ]; then
  GUI_ARGS="-p freeze_joints:=$FREEZE_JOINTS"
fi
cat > "$TMPD/arm_gui.sh" << SCRIPT
#!/bin/bash
source $ROS_SETUP
source $WS/install/setup.bash
echo ""
echo "  Waiting 8s for Gazebo + controllers..."
sleep 8
echo "  Starting Arm GUI..."
python3 $SCRIPTS/arm_gui.py --ros-args $GUI_ARGS
exec bash
SCRIPT

chmod +x "$TMPD/gazebo.sh" "$TMPD/inference.sh" "$TMPD/robot.sh" "$TMPD/hardware.sh" "$TMPD/arm_gui.sh"

# ── open one terminal window, one tab per node ────────────────────────────────
# NOTE: "-e" is deprecated, but it is the only form that works for multiple
# tabs. gnome-terminal's "--" terminates option parsing and treats the ENTIRE
# rest of the command line as one command, so the previous "--tab ... -- bash
# a.sh --tab ... -- bash b.sh" opened a SINGLE tab running a.sh with every
# other tab's arguments passed to it as argv. Only Gazebo ever started; the
# camera GUI, robot node, hardware interface and arm GUI were silently never
# launched. Verified on GNOME Terminal 3.44 that -e runs each tab correctly.
gnome-terminal \
  --tab --title="Gazebo"              -e "bash $TMPD/gazebo.sh" \
  --tab --title="ML Inference (LSTM)" -e "bash $TMPD/inference.sh" \
  --tab --title="Robot Node"          -e "bash $TMPD/robot.sh" \
  --tab --title="Hardware Interface"  -e "bash $TMPD/hardware.sh" \
  --tab --title="Arm GUI"             -e "bash $TMPD/arm_gui.sh" &

# ── maximize Gazebo ───────────────────────────────────────────────────────────
echo ">>> Waiting for Gazebo window…"
for i in $(seq 1 30); do
  sleep 2
  if wmctrl -l 2>/dev/null | grep -qi "gazebo"; then
    wmctrl -r "gazebo" -b add,maximized_vert,maximized_horz
    echo ">>> Gazebo maximized."
    break
  fi
done

echo ""
echo "  ┌──────────────────────────────────────────────────────────────┐"
echo "  │  Window 1  Gazebo + RViz2    → simulation (maximized)        │"
echo "  │  Window 2  ML Inference      → camera GUI, starts in 20s     │"
echo "  │  Window 3  Robot Node        → starts in 22s                 │"
echo "  │  Window 4  Hardware Interface→ starts in 23s                 │"
echo "  │  Window 5  Arm GUI           → starts in 8s                  │"
echo "  │                                                              │"
echo "  │  Pipeline:                                                   │"
echo "  │  Camera → MediaPipe → LSTM → ROS 2 → Robot Arm              │"
echo "  │                                                              │"
echo "  │  Perform a gesture with your RIGHT arm.                      │"
echo "  │  LSTM classifies → robot arm executes movement.              │"
echo "  │                                                              │"
echo "  │  Flags:  --hide    hide the debug camera window              │"
echo "  │          --no-hw   disable hardware serial (dry-run)         │"
echo "  │          --direct  bypass LSTM, use direct landmark mapping  │"
echo "  │          --no-build  skip colcon build                       │"
echo "  │          --freeze-l1-l5  debug: hold L1-L5, L6/L7 stay live  │"
echo "  │          --freeze-l1-l4  debug: hold L1-L4, L5-L7 stay live  │"
echo "  │          --freeze=joint_L1,...  debug: hold exactly these    │"
echo "  │          --pose-predictor  opt-in: enable pose predictor     │"
echo "  │          --stop    kill every node this pipeline runs        │"
echo "  └──────────────────────────────────────────────────────────────┘"
