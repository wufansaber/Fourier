from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tdi_charge_transfer import charge_transfer_image,fixed_illumination_weights


@pytest.mark.parametrize("stages",[1,25,256])
def test_charge_tracks_same_sample_without_motion_smear(stages):
    image=np.random.default_rng(73).random((37,11))
    q,gain=charge_transfer_image(image,stages)
    assert gain==stages
    np.testing.assert_allclose(q/gain,image,atol=2e-14,rtol=2e-14)


@pytest.mark.parametrize("stages",[25,256])
def test_fixed_illumination_changes_gain_not_registered_geometry(stages):
    image=np.eye(31,17)
    weights=fixed_illumination_weights(stages,.00025,.00625)
    q,gain=charge_transfer_image(image,stages,weights)
    np.testing.assert_allclose(q/gain,image,atol=2e-14)


def test_pipeline_fill_edges_and_no_wraparound():
    image=np.zeros((9,3));image[0,0]=1;image[-1,-1]=2
    q,gain=charge_transfer_image(image,256)
    assert np.count_nonzero(q)==2
    np.testing.assert_allclose(q/gain,image)


def test_image_is_not_convolved_with_stage_height():
    image=np.zeros((513,1));image[256,0]=1
    q,gain=charge_transfer_image(image,256)
    assert np.flatnonzero(q[:,0]).tolist()==[256]
    assert (q/gain)[256,0]==1
