"""Fixed-line scanned images, then two-layer net-zero complex propagation.

Object scan position s is NOT the physical detector coordinate u. No TDI
boxcar, per-image normalization, or independent intensity sums for defects.
"""
import argparse
import gc
import json
import numpy as np
from scipy import fft,ndimage,special
import stacked_serrated as s
import ch3_linescan as m
from ideal_na06 import save_json


def interval_integral(nu,a,b):
    return (b-a)*np.sinc(nu*(b-a))*np.exp(-2j*np.pi*nu*(a+b)/2)


def tapered_interval(nu,a,b,edge,taper):
    """Analytic Fourier integral of the raised-cosine field envelope."""
    out=np.zeros_like(nu,dtype=complex);inner=edge-taper
    for lo,hi,kind in [(-edge,-inner,-1),(-inner,inner,0),(inner,edge,1)]:
        left=max(a,lo);right=min(b,hi)
        if right<=left:continue
        plain=interval_integral(nu,left,right)
        if kind==0:out+=plain
        else:
            freq=1/(2*taper);phase=-kind*np.pi*inner/taper
            out+=.5*plain+.25*(np.exp(1j*phase)*interval_integral(nu-freq,left,right)+
                              np.exp(-1j*phase)*interval_integral(nu+freq,left,right))
    return out


def reference_spectrum(cfg,axis,case):
    """Continuous object transforms avoid grid-dependent binary edge widths."""
    n=len(axis);dx=axis[1]-axis[0];nu=fft.fftshift(fft.fftfreq(n,dx))
    edge=cfg['object']['field_mm']/2;taper=cfg['object']['line_edge_taper_mm'];p=cfg['object']['pitch_mm']
    radius=cfg['object']['feature_width_mm']/2;centers=np.arange(-int(edge/p),int(edge/p)+1)*p
    if case=='stripes':
        fy=sum(tapered_interval(nu,cy-radius,cy+radius,edge,taper) for cy in centers)/(np.sqrt(n)*dx)
        fx=tapered_interval(nu,-edge,edge,edge,taper)/(np.sqrt(n)*dx)
        return (fy[:,None]*fx[None,:]).astype(np.complex64)
    envelope=.5+.5*np.cos(np.pi*np.clip((abs(centers)-(edge-taper))/taper,0,1))
    array=sum(a*np.exp(-2j*np.pi*nu*cy) for cy,a in zip(centers,envelope))
    rho=np.sqrt(nu[:,None]**2+nu[None,:]**2).astype(np.float32)
    t=2*np.pi*radius*rho
    disk=np.pi*radius**2*2*special.j1(t)/np.where(t==0,1,t);disk[t==0]=np.pi*radius**2
    return (disk*array[:,None]*array[None,:]/(n*dx**2)).astype(np.complex64)


def scan(case,n=None,kmax=12):
    cfg=m.config();tc=s.config();n=n or tc['full_n'];window=tc['full_window_mm'];dx=window/n
    d=json.loads((s.OUT/'layer_refinement.json').read_text())['selected']['design']
    step=int(round(m.scales(cfg)['pixel_object_mm']/dx))
    assert step in (1,2) and dx<=cfg['optics']['wavelength_mm']/(2*cfg['optics']['numerical_aperture'])
    axis=(np.arange(n)-n//2)*dx;nu=fft.fftshift(fft.fftfreq(n,dx));ff=nu*m.scales(cfg)['alpha_mm2']
    ids=np.flatnonzero(abs(axis)<=tc['display_half_um']/1000)[::step]
    sl=slice(ids[0],ids[-1]+1,step)
    nodes,weights=np.polynomial.legendre.leggauss(tc['camera_quadrature'])
    offsets=nodes*m.scales(cfg)['line_height_object_mm']/2
    output={key:np.zeros((len(ids),len(ids)),dtype=np.float32)
            for key in ['unfiltered','filtered','background','defects_only','d1','d2','d3','unfiltered_background','unfiltered_d1','unfiltered_d2','unfiltered_d3']}
    energy={key:0. for key in output}
    obj,defs=m.full_objects(cfg,axis.astype(np.float32),case)
    obj_display=(obj[sl,sl]+sum(v[sl,sl] for v in defs))**2
    del obj,defs
    fref=reference_spectrum(cfg,axis,case)
    Iref=m.camera_intensity_reference(cfg)
    def add(key,field,weight):
        # Squaring doubles bandwidth: upsample coarse complex fields BEFORE
        # intensity, then integrate the exact rectangular y pixel by sinc.
        scale=2 if step==1 else 1
        if scale==2:
            spec=s.ft(field);pad=np.pad(spec,((n//2,n//2),(0,0)))
            field=s.ift(pad)*np.sqrt(2)
        intensity=abs(field)**2;fine_dx=dx/scale;nf=len(intensity)
        energy[key]+=float(weight*np.sum(intensity,dtype=np.float64)*fine_dx*m.scales(cfg)['pixel_object_mm']/Iref)
        frequencies=fft.fftshift(fft.fftfreq(nf,fine_dx))
        v=s.ift(s.ft(intensity)*np.sinc(frequencies*m.scales(cfg)['pixel_object_mm'])[:,None]).real
        # All index coordinates refer to the same centered physical grid.
        image_ids=nf//2+(ids-n//2)*scale
        output[key]+=weight*np.maximum(v[image_ids,:],0)/Iref
    for j,(u,weight) in enumerate(zip(offsets,weights/2)):
        g,_=m.full_transfer(cfg,n,window,dict(profile='none',orders=[]),float(u))
        whole=m.ifftc(fref*g);initial=whole[:,sl].copy();del whole
        filtered=s.zero_order_stack(initial,axis,ff,float(u),cfg,d,kmax)
        add('background',filtered,weight);add('unfiltered_background',initial,weight)
        total=filtered.copy();total_initial=initial.copy();del initial
        defects_field=np.zeros_like(filtered);del filtered
        for k,(cx,cy,size) in enumerate(m.DEFECTS):
            # A separable Gaussian's 2-D transform is the outer product of
            # its unitary 1-D transforms; avoids three huge stored FFT arrays.
            ax=np.exp(-2*np.log(2)*(axis-cx)**2/size**2).astype(np.float32)
            ay=np.exp(-2*np.log(2)*(axis-cy)**2/size**2).astype(np.float32)
            spec=s.ft(ay)[:,None]*s.ft(ax)[None,:]
            whole=m.ifftc(spec*g);initial=whole[:,sl].copy();del whole,spec
            filtered=s.zero_order_stack(initial,axis,ff,float(u),cfg,d,kmax)
            add(f'd{k+1}',filtered,weight);add(f'unfiltered_d{k+1}',initial,weight)
            total_initial+=initial;total+=filtered;defects_field+=filtered
            del initial,filtered
        add('filtered',total,weight);add('unfiltered',total_initial,weight)
        add('defects_only',defects_field,weight)
        del total,total_initial,defects_field,g;gc.collect()
        print(f'Layered {case} n={n}: line-height sample {j+1}/{len(offsets)}',flush=True)
    output['difference']=output['filtered']-output['background']
    np.savez_compressed(s.OUT/f'{case}_scan_n{n}.npz',axis_um=axis[sl]*1000,object_intensity=obj_display,**output)
    metric=dict(case=case,n=n,dx_um=dx*1000,kmax=kmax,display_roi_half_um=tc['display_half_um'],
        intensity_unit='I/I_ref',intensity_reference=Iref,
        source_sampling='Analytic continuous stripes/disk transforms. Disk edge taper is per-disk center amplitude.',
        detector_integration='Bandlimited complex-field upsample before squaring if needed; exact sinc response of rectangular y pixel.',
        background_suppression_roi_db=float(-10*np.log10(output['background'].sum()/output['unfiltered_background'].sum())),
        isolated_defect_retention_roi=[float(output[f'd{k}'].sum()/output[f'unfiltered_d{k}'].sum()) for k in [1,2,3]],
        background_suppression_full_y_db=float(-10*np.log10(energy['background']/energy['unfiltered_background'])),
        isolated_defect_retention_full_y=[energy[f'd{k}']/energy[f'unfiltered_d{k}'] for k in [1,2,3]],
        energy_integration='Full 1.024 mm y window, scan x +/-45 um. Display ROI energy is separate.',
        caveat='Total-zero-order two-layer field; finite-line-height integration; display ROI energy is not full-plane energy.')
    save_json(s.OUT/f'{case}_scan_n{n}_metrics.json',metric)
    print(json.dumps(metric,indent=2),flush=True)
    return metric


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int);parser.add_argument('--kmax',type=int,default=12)
    parser.add_argument('--case',choices=['dots','stripes','both'],default='both');a=parser.parse_args()
    for case in (['dots','stripes'] if a.case=='both' else [a.case]):scan(case,a.n,a.kmax)
