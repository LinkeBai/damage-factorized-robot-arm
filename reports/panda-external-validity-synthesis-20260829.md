# Panda simulation external-validity synthesis

## Decision

**Partial structural external validity; object/contact external validity
No-Go.** Panda evidence strengthens the implementation and failure-boundary
story, but it does not establish cross-robot task-level SI-IPWM generalization.

## Evidence chain

1. **Model provenance and task physics -- pass.** The Panda arm and gripper are
   from MuJoCo Menagerie at the frozen upstream commit. The wrapper retains all
   seven arm joints, two physical fingers, tendon/equality coupling, collision
   pads and a free cube.
2. **Variable-DoF analytic interface -- pass.** One parameter-sharing core
   accepts the complete 5-DoF GenkiArm and 7-DoF Panda chains without robot ID,
   joint deletion or per-robot heads. Padding invariance and exact lock
   projection pass at 1e-7 tolerance.
3. **Held-out-lock robot transition -- small Gate pass.** The shared structural
   model beats the parameter-matched flat baseline on both robots in seeds 7
   and 27, but seed 17 regresses on GenkiArm. This is joint training on both
   structures, not unseen-robot zero-shot transfer.
4. **Object/contact propagation -- No-Go.** The structured model regresses
   pooled object-response RMSE by 28.00%, 63.58% and 13.18% for seeds 7/17/27.
   The frozen both-robots criterion passes in 0/3 seeds.
5. **Grasp task feasibility -- pass, method absent.** A no-weld scripted Panda
   baseline physically grasps and lifts the cube in 5/5 small position
   perturbations. No learned SI-IPWM Grasp comparison exists.
6. **Dual eye-to-hand placement -- observability pass only.** The object remains
   visible in both cameras under all frozen extrinsic perturbations, but the
   state-based SI-IPWM does not consume RGB.

## Permitted paper statement

The mechanism and data interface can represent two structurally different
robots and gives a narrow held-out-lock robot-transition benefit. This benefit
does not reliably extend through contact to object dynamics. Panda therefore
serves as a transparent external-validity boundary and task-feasibility model,
not as positive cross-robot Push/Grasp evidence.

## Prohibited statement

Do not claim cross-robot object propagation, unseen-robot transfer, Grasp
improvement, multi-view visual robustness or task-level external validity. The
0/3 contact/object No-Go is the decisive result for those claims.
