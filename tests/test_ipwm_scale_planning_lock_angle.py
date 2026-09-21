"""A replan must not reinterpret observed solver drift as a new diagnosis."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ipwm_scale_planning import select

@pytest.mark.skipif(not torch.cuda.is_available(),reason='Planner uses CUDA')
def test_diagnosed_angle_survives_observed_drift():
    diagnosed=np.array([.4,.7],dtype=np.float32)
    states=np.zeros((2,14),dtype=np.float32)
    states[:,2]=diagnosed+.02
    actions=np.zeros((2,128,5,5),dtype=np.float32)
    expected=torch.tensor(np.repeat(diagnosed,128),device='cuda')
    class Recorder:
        calls=0
        def step(self,state,action,mask,angles,hidden):
            assert torch.equal(angles[:,2],expected)
            self.calls+=1
            return state,hidden
    model=Recorder()
    select(model,states,actions,np.zeros((2,2)),2,diagnosed)
    assert model.calls==50
