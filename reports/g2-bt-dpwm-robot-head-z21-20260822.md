# BT-DPWM robot-head adaptation: Z21

Z21 keeps the h136 scaffold encoder, messages and temporal state frozen. Only
18,906 existing robot-head parameters are adapted for 40 epochs at learning
rate 1e-4, followed by the standard object32 training stage.

Robot training loss falls from 0.02162 to 0.01005 and object loss reaches
0.000872. Nevertheless, seed-7 primary performance is free -60.00%, object
+42.08%, overall -52.14% relative to shared h136/240.

Z21 is NO-GO and seeds 17/27 are not run. Low-dimensional, low-rate supervised
refitting still destroys held-out robot generalization; the full h136 robot
scaffold must remain frozen under the current leave-D3-out data.
