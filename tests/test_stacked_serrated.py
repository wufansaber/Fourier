import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import stacked_serrated as s
import ch3_linescan as m


def design():
    return dict(centres_mm=[-.8,0,.8],layers=[0,1,0],widths_mm=[.2,.3],depths_mm=[.25,.2],
                tooth_pitch_mm=.75,relative_phase=.25,z_mm=[-2.,2.],layout='test',rod_count=3,profile='triangle')


def test_all_serrated_constraints():
    d=design();assert s.check_design(d)
    for change in [dict(tooth_pitch_mm=.49),dict(depths_mm=[0,.1]),dict(layers=[0,3,0]),
                   dict(centres_mm=[-.1,0,.1]),dict(centres_mm=list(range(11)),layers=[0]*11)]:
        with pytest.raises(ValueError):s.check_design({**d,**change})


@pytest.mark.parametrize('phase',[0.,.13,.25,.5])
def test_binary_projection_not_independent_mean(phase):
    q=(np.arange(200000)+.5)/200000-.5
    a=np.linspace(.01,.99,31);b=a[::-1]**2
    numerical=np.mean((abs(q[None,:])<a[:,None]/2)&
                      (abs((q[None,:]-phase+.5)%1-.5)<b[:,None]/2),axis=1)
    np.testing.assert_allclose(s.interval_overlap(a,b,phase),numerical,atol=1e-5)


def test_returning_orders_are_kept():
    d=design();c=m.config();ff=np.linspace(-1.5,1.5,501)
    d['centres_mm']=[-.5,0,.5]
    order,cs=s.coefficients(ff,c,d,256)
    product=np.sum(cs[0][:,::-1]*cs[1],axis=1)
    np.testing.assert_allclose(product,s.projected_transmission(ff,c,d),atol=.0008)
    assert np.max(abs(cs[0][:,256]*cs[1][:,256]-product))>.01


def test_fresnel_harmonic_against_independent_phase_cycled_fft2():
    # Uniform phase cycling selects net order zero WITHOUT separately deleting
    # intermediate orders. Integer replica shift eliminates interpolation error.
    n=128;dx=.0005;x=(np.arange(n)-n//2)*dx;c=m.config();d=design();kmax=2
    d['tooth_pitch_mm']=m.scales(c)['alpha_mm2']/(8*dx)
    ff=np.fft.fftshift(np.fft.fftfreq(n,dx))*m.scales(c)['alpha_mm2']
    xx=x[None,:];yy=x[:,None]
    e=np.exp(-((xx-.003)**2+(yy+.002)**2)/.008**2)*(1+.3j*xx/.01)
    orders,cs=s.coefficients(ff,c,d,kmax)
    phase1=s.phase_fresnel(x,d['z_mm'][0],c)
    phase2=s.phase_fresnel(x,np.diff(d['z_mm'])[0],c)
    phase3=s.phase_fresnel(x,-d['z_mm'][1],c)
    out=np.zeros_like(e,dtype=complex)
    for phi in np.arange(4*kmax+1)/(4*kmax+1):
        masks=[sum(cs[layer][:,j,None]*np.exp(2j*np.pi*k*(ff[None,:]/d['tooth_pitch_mm']-phi))
                   for j,k in enumerate(orders)) for layer in [0,1]]
        v=e*phase1[:,None]*phase1[None,:]
        v=m.ifftc(m.fftc(v)*masks[0])*phase2[:,None]*phase2[None,:]
        v=m.ifftc(m.fftc(v)*masks[1])*phase3[:,None]*phase3[None,:]
        out+=v/(4*kmax+1)
    ids=np.arange(n//2-10,n//2+11)
    fast=s.zero_order_stack(e[:,ids],x,ff,x[ids],c,d,kmax)
    np.testing.assert_allclose(fast,out[:,ids],atol=2e-14)


def test_zero_separation_agrees_with_projected_mask():
    c=m.config();d={**design(),'z_mm':[0.,0.]};a=m.search_arrays(c,8192,1.024)
    e=s.source_fields(c,a)
    fast=s.zero_order_stack(e,a['axis'],a['ff'],0,c,d,128)
    exact=s.ift(s.ft(e)*s.projected_transmission(a['ff'],c,d)[:,None])
    assert np.linalg.norm(fast-exact)/np.linalg.norm(exact)<.001


def test_no_masks_cancel_virtual_propagation():
    c=m.config();d={**design(),'centres_mm':[],'layers':[]};a=m.search_arrays(c,2048,.256)
    e=s.source_fields(c,a)
    f=s.zero_order_stack(e,a['axis'],a['ff'],.001,c,d,8)
    np.testing.assert_allclose(f,e,atol=2e-14)


def test_continuous_tapered_object_transform():
    from stacked_images import tapered_interval
    from scipy.integrate import quad
    edge=.25;taper=.03;a=-.2435;b=-.2365
    nu=np.array([0.,15.3,127.7])
    def envelope(y):return .5+.5*np.cos(np.pi*np.clip((abs(y)-(edge-taper))/taper,0,1))
    expected=np.array([quad(lambda y:envelope(y)*np.cos(2*np.pi*f*y),a,b,epsabs=1e-13)[0]
                      -1j*quad(lambda y:envelope(y)*np.sin(2*np.pi*f*y),a,b,epsabs=1e-13)[0] for f in nu])
    np.testing.assert_allclose(tapered_interval(nu,a,b,edge,taper),expected,atol=1e-13)
