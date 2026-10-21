"""Independent convergence, physical 2-D propagation, and off-axis validation."""
import copy
import json
import numpy as np
from scipy import fft
import stacked_serrated as s
import ch3_linescan as m
from ideal_na06 import save_json


def phase_2d(axis,z,cfg,exact=False):
    lam=cfg['optics']['wavelength_mm'];alpha=m.scales(cfg)['alpha_mm2']
    r2=(axis[:,None]**2+axis[None,:]**2)
    if exact:
        v=lam**2*r2/alpha**2
        # Rational form avoids cancellation in sqrt(1-v)-1.
        exponent=-2j*np.pi*z/lam*v/(1+np.sqrt(1-v+0j))
    else:exponent=-1j*np.pi*z*lam*r2/alpha**2
    return np.exp(exponent).astype(np.complex64)


def binary_mask(ff,cfg,d,layer):
    duty=s.layer_duty(ff,d,layer).astype(np.float32)
    phi=0 if layer==0 else d['relative_phase']
    q=abs((ff/d['tooth_pitch_mm']-phi+.5)%1-.5)
    leak=np.sqrt(cfg['filter']['blocked_intensity_transmission'])
    return np.where(q[None,:]<duty[:,None]/2,1.,leak).astype(np.float32)


def direct_stack(field,axis,ff,cfg,d,exact=False):
    """All optical orders retained; no fictitious aperture at intermediate planes."""
    z1,z2=d['z_mm']
    out=field*phase_2d(axis,z1,cfg,exact)
    out=m.ifftc(m.fftc(out)*binary_mask(ff,cfg,d,0))
    out*=phase_2d(axis,z2-z1,cfg,exact)
    out=m.ifftc(m.fftc(out)*binary_mask(ff,cfg,d,1))
    out*=phase_2d(axis,-z2,cfg,exact)
    return out


def run():
    cfg=m.config();tc=s.config();row=json.loads((s.OUT/'layer_refinement.json').read_text())['selected'];d=row['design']
    arr=m.search_arrays(cfg,tc['refine_n'],tc['refine_window_mm']);base=s.source_fields(cfg,arr)
    convergence=[];previous=None
    for k in [6,12,24,48]:
        fields=s.zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,d,k)
        metrics=s.score_fields(cfg,arr,base,fields)
        metrics.update(kmax=k,relative_field_change=None if previous is None else float(np.linalg.norm(fields-previous)/np.linalg.norm(fields)))
        convergence.append(metrics);previous=fields
    variations=[]
    for y in tc['field_offsets_y_mm']:
        for dw,shift in [(0.,0.),(-.010,.025),(-.010,-.025),(.010,0.)]:
            trial=copy.deepcopy(d);trial['widths_mm']=[w+dw for w in d['widths_mm']]
            trial['centres_mm']=[v+shift for v in d['centres_mm']]
            f=s.zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,trial,24,y_offset=y)
            metric=s.score_fields(cfg,arr,base,f);metric.update(y_offset_mm=y,width_error_mm=dw,common_position_error_mm=shift)
            variations.append(metric)
    save_json(s.OUT/'validation.json',dict(convergence=convergence,variations=variations,
        caveat='Common position error only; not independent errors for each rod. Off-axis uses local targets, background proxy shifted with target.'))
    print('Harmonic / tolerance validation complete',flush=True)
    # Full binary-mask point response: independent order leakage and propagation.
    n=4096;window=1.024;axis=(np.arange(n)-n//2)*window/n
    nu=fft.fftshift(fft.fftfreq(n,window/n));ff=nu*m.scales(cfg)['alpha_mm2']
    pupil=(nu[:,None]**2+nu[None,:]**2<=m.scales(cfg)['cutoff_per_mm']**2).astype(np.float32)
    h=m.ifftc(pupil);del pupil
    f=direct_stack(h,axis,ff,cfg,d);g=direct_stack(h,axis,ff,cfg,d,True)
    roi=abs(axis)<=.045;ids=np.flatnonzero(roi);sl=slice(ids[0],ids[-1]+1)
    # Point response normalization is kept common with its own NO-ROD point
    # input. It is NOT labelled as extended-object camera I/I_ref.
    scale=float(abs(h[n//2,n//2])**2)
    np.savez_compressed(s.OUT/'all_order_point.npz',axis_mm=axis[sl],field=f[sl,sl],baseline=h[sl,sl],point_peak_reference=scale)
    comparison=dict(n=n,window_mm=window,local_exact_vs_fresnel_field_relative=float(np.linalg.norm(f-g)/np.linalg.norm(g)),
        local_exact_vs_fresnel_intensity_relative=float(np.linalg.norm(abs(f)**2-abs(g)**2)/np.linalg.norm(abs(g)**2)),
        retained_all_orders=True,point_normalization='one fixed unfiltered point peak; not extended object intensity reference')
    # A narrow illuminated Gaussian defect checks total-zero applicability at
    # the actual line camera. Nonzero orders are not artificially deleted here.
    del h,g,f
    x=axis[None,:];y=axis[:,None];size=.0005
    source=np.exp(-2*np.log(2)*((x*x+y*y)/size**2+x*x/.001**2)).astype(np.float32)
    spectrum=m.fftc(source)
    spectrum[nu[:,None]**2+nu[None,:]**2>m.scales(cfg)['cutoff_per_mm']**2]=0
    initial=m.ifftc(spectrum);del source,spectrum
    direct=direct_stack(initial,axis,ff,cfg,d)
    fast=s.zero_order_stack(initial[:,n//2],axis,ff,0.,cfg,d,48)[:,0]
    cy=abs(axis)<.045
    comparison['all_order_vs_total_zero_line_field_relative']=float(np.linalg.norm(direct[cy,n//2]-fast[cy])/np.linalg.norm(direct[cy,n//2]))
    comparison['all_order_vs_total_zero_line_intensity_relative']=float(np.linalg.norm(abs(direct[cy,n//2])**2-abs(fast[cy])**2)/np.linalg.norm(abs(direct[cy,n//2])**2))
    comparison['binary_frequency_sampling_mm']=float(ff[1]-ff[0])
    comparison['comparison_caveat']='Includes binary-edge rasterization error as well as nonzero-order leakage; not a pure leakage measurement.'
    save_json(s.OUT/'direct_validation.json',comparison)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':run()
