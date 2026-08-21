# BT-DPWM fixed-budget robot ensemble: Z20

Z20 replaces one h136 robot block with two independently initialized h96 graph
experts. Their transitions are averaged before analytic projection and feed one
independent object32 block. Total parameters are 337,448, 654 below the strong
h136/240 baseline.

Under the frozen budget, robot training loss falls from 0.1790 to 0.01070 after
240 epochs, below the strong baseline's approximately 0.0176. Object loss then
reaches 0.000726 after 120 epochs. Despite the lower training loss, seed-7
primary performance is free -30.20%, object +42.01%, overall -24.31%.

Z20 is NO-GO and seeds 17/27 are not run. Equal total capacity and prediction
averaging do not recover the held-out generalization of the wide pretrained
h136 scaffold.
