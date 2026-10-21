import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import ch3_linescan as m
from patent_ringing import transform,transform_quad


def test_scales_and_channel():
    c=m.config();s=m.scales(c)
    assert c["optics"]["channel"]=="Ch3"
    assert c["optics"]["illumination_angle_deg"]==c["optics"]["collection_angle_deg"]==0
    assert s["pupil_mm"]==36
    assert s["filter_pupil_mm"]==36
    assert s["mechanical_clear_diameter_mm"]==50
    assert s["relay_magnification"]==1
    assert s["f_ff_mm"]==30
    assert s["order_pitch_mm"]==pytest.approx(.5325)
    assert s["tooth_first_order_shift_um"]==pytest.approx(14.2)


def test_mechanical_aperture_is_not_pupil_magnification():
    c=m.config();original=m.scales(c)
    c["optics"]["filter_clear_diameter_mm"]=60
    assert m.scales(c)["alpha_mm2"]==original["alpha_mm2"]
    c["optics"]["filter_clear_diameter_mm"]=35
    with pytest.raises(ValueError,match="clips optical pupil"):
        m.scales(c)


@pytest.mark.parametrize("profile",["linear","sine"])
def test_patent_analytic_transform(profile):
    for a,b in [(.05,.1),(.5,1)]:
        for q in [0,.5,1/(4*b),7.3,37.1]:
            assert float(transform(q,a,b,profile))==pytest.approx(transform_quad(q,a,b,profile),abs=1e-10)


def test_patent_eq36_counterexample():
    assert abs(float(transform(.5,.5,1,"linear")))<1e-14
    assert 1*np.sinc(.5)**2==pytest.approx(.40528473457)


@pytest.mark.parametrize("profile",["linear","sine"])
def test_exported_binary_geometry_matches_zero_order(profile):
    c=m.config();p=c["filter"]["tooth_pitch_mm"];tip=c["filter"]["tooth_tip_minimum_mm"]/p
    x=np.linspace(0,p,200000,endpoint=False)
    phase=np.clip(np.abs(np.remainder(x+p/2,p)-p/2)/(p/2),tip,1-tip)
    edge=.05+.1*(phase if profile=="linear" else 2/np.pi*np.arcsin(phase))
    ff=np.linspace(0,.20,301)
    binary_average=np.array([np.mean(edge<y) for y in ff])
    predicted=m.mean_transmission(ff,c,dict(profile=profile,core_width_mm=.1,b_mm=.1,threshold=0,orders=[0]),leak=0)
    assert np.max(abs(binary_average-predicted))<2e-5
    t=.4
    assert t*t!=t  # zero-order power is NOT total binary transmitted power


def test_full_scanned_kernel_matches_direct_illuminate_propagate_sample():
    c=m.config();n=256;window=.064;dx=window/n
    design=dict(profile="sine",core_width_mm=.1,b_mm=.1,threshold=0,orders=[-2,-1,0,1,2])
    g,x=m.full_transfer(c,n,window,design,0)
    xx=x[None,:];yy=x[:,None]
    obj=np.exp(-2*np.log(2)*((xx-.002)**2+(yy-.001)**2)/.0007**2).astype(np.float32)
    fast=m.ifftc(m.fftc(obj)*g)
    nu=np.fft.fftshift(np.fft.fftfreq(n,dx));sc=m.scales(c)
    pupil=(nu[:,None]**2+nu[None,:]**2<=sc["cutoff_per_mm"]**2)
    t=m.mean_transmission(nu*sc["alpha_mm2"],c,design)
    for j in [n//2,n//2+7,n//2+8,n//2+9]:
        illumination=np.exp(-2*np.log(2)*(x-x[j])**2/c["camera"]["illumination_intensity_fwhm_mm"]**2)
        direct=m.ifftc(m.fftc(obj*illumination[None,:])*pupil*t[:,None])
        assert np.max(abs(direct[:,j]-fast[:,j]))<3e-7


def test_background_filtering_does_not_predelete_object_orders():
    c=m.config();a=m.search_arrays(c,n=4096,window=.256)
    none=m.evaluate_candidate(c,a,dict(profile="none",core_width_mm=0,b_mm=0,threshold=0))
    assert none["worst_background_suppression_db"]==pytest.approx(0)
    assert none["nominal_min_retention"]==pytest.approx(1)
    assert none["maximum_ee90_ratio"]==pytest.approx(1)


def test_rod_limit_and_unselected_orders_are_transparent():
    c=m.config();d=dict(profile="sine",core_width_mm=.1,b_mm=.075,threshold=0,orders=[-2,-1,0,1,2])
    pitch=m.scales(c)["order_pitch_mm"]
    values=m.mean_transmission(np.array([0,pitch,2*pitch,3*pitch]),c,d)
    assert np.allclose(values[:3],.01)
    assert values[3]==1
    with pytest.raises(ValueError,match="exceeds"):
        m.active_orders(c,{**d,"orders":list(range(-5,6))})
    with pytest.raises(ValueError,match="Explicit"):
        m.active_orders(c,dict(profile="sine"))


def test_common_intensity_reference_is_independent_of_rod_design():
    c=m.config();a=m.camera_intensity_reference(c)
    assert .98<a<1
    assert m.camera_intensity_reference(c,20)==pytest.approx(a,abs=1e-12)
    c["filter"]["blocked_intensity_transmission"]=.5
    assert m.camera_intensity_reference(c)==a


def test_common_display_contains_all_three_true_defects():
    from ch3_display import LIMITS,display_extent
    assert len(m.DEFECTS)==3
    for x,y,_ in m.DEFECTS:
        assert LIMITS[0]<1000*x<LIMITS[1]
        assert LIMITS[0]<1000*y<LIMITS[1]
    assert display_extent(np.array([-1.,0.,1.]))==[-1.5,1.5,-1.5,1.5]


def test_uniform_object_camera_reference_and_quadratic_intensity():
    c=m.config();nodes,weights=np.polynomial.legendre.leggauss(12)
    n=256;value=0.
    for u,w in zip(nodes*m.scales(c)["line_height_object_mm"]/2,weights/2):
        transfer,_=m.full_transfer(c,n,.064,dict(profile="none"),u)
        # The zero-frequency transfer of an infinite uniform amplitude object.
        value+=w*float(abs(transfer[n//2,n//2])**2)
    assert value/m.camera_intensity_reference(c)==pytest.approx(1,abs=2e-6)
    assert (.5**2*value)/m.camera_intensity_reference(c)==pytest.approx(.25,abs=1e-6)


def test_search_profile_continuous_amplitude_calibration():
    from scipy import fft,special
    c=m.config();a=m.search_arrays(c,n=1024,window=.064)
    d=c["object"]["defect_fwhm_um"][0]/1000
    wx=1/np.sqrt(1/d**2+1/c["camera"]["illumination_intensity_fwhm_mm"]**2)
    fxmax=np.sqrt(np.maximum(0,m.scales(c)["cutoff_per_mm"]**2-a["nu"]**2))
    obj=np.exp(-2*np.log(2)*a["axis"]**2/d**2)
    measured=fft.fftshift(fft.ifft(fft.ifftshift(fft.fftshift(fft.fft(fft.ifftshift(obj),norm="ortho"))*special.erf(np.pi*wx*fxmax/np.sqrt(2*np.log(2)))),norm="ortho"))
    e=fft.fftshift(fft.ifft(fft.ifftshift(a["targets"][0]),norm="ortho"))
    scale=d*np.sqrt(np.pi/(2*np.log(2)))/(np.sqrt(a["n"])*a["dx"])
    assert np.max(abs(e*scale-measured))<1e-12
