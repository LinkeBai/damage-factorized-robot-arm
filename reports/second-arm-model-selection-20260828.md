# Second-arm model selection: Franka Panda

**Decision:** use the official Franka Emika Panda from Google DeepMind MuJoCo
Menagerie as the second, structurally different arm.

## Frozen source

- Repository: `https://github.com/google-deepmind/mujoco_menagerie`
- Upstream commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Selected directory: `franka_emika_panda`
- License: Apache-2.0; the upstream `LICENSE` is retained verbatim.
- Local task wrapper: `sim/assets/panda_push_grasp.xml`.

The sparse upstream checkout is stored at `third_party/mujoco_menagerie` so
that source assets, history identifier and license remain auditable.

## Why Panda rather than another 5-DoF proxy

Panda has seven revolute arm joints rather than GenkiArm's five, different
kinematic axes/link geometry and identified inertial parameters.  Its official
MJCF also includes two physical finger slide joints, fingertip collision pads,
finger equality synchronization, a force-splitting tendon and a gripper
actuator.  It can therefore support both Push and contact-defined Grasp without
an ideal weld or hand-written `grasped` flag.

The wrapper adds only a shared cube, table, target and two fixed eye-to-hand
cameras.  Automated tests verify eight actuators, seven arm joints, both finger
joints, a free cube, both cameras and finite home-pose dynamics.

## Critical method implication

The current IPWM implementation and 14-D Push interface are fixed to five arm
joints.  Loading Panda successfully does **not** establish cross-structure
generalization.  Before formal experiments, the method must expose a
variable-node graph/state interface or use a frozen per-robot encoder with a
shared intervention mechanism.  Simply padding Panda to five joints, dropping
two joints, or training an unrelated Panda MLP would not test the paper's
claimed structural generalization.
