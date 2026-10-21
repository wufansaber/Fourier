"""Two all-triangular-serrated decks: projection prescreen and Fresnel fields.

No per-deck zero-order deletion: k and -k paths can return to total order zero.
All Fourier transforms are centered, unitary; physical source scaling is kept
separate from diagnostic energy ratios. Layer distance is physical mm.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import fft
import yaml
import ch3_linescan as m
from ideal_na06 import save_csv,save_json

ROOT=m.ROOT
OUT=ROOT/'results/stacked_all_serrated'


@lru_cache(maxsize=1)
def config():
    return yaml.safe_load((ROOT/'config/stacked_serrated.yml').read_text(encoding='utf-8'))


def ft(a,axis=0):return fft.fftshift(fft.fft(fft.ifftshift(a,axes=axis),axis=axis,norm='ortho',workers=4),axes=axis)
def ift(a,axis=0):return fft.fftshift(fft.ifft(fft.ifftshift(a,axes=axis),axis=axis,norm='ortho',workers=4),axes=axis)


def check_design(d,tc=None):
    tc=tc or config()
    if not 1<=len(d['centres_mm'])<=tc['maximum_rods']:raise ValueError('Physical rod count must be <=10')
    if len(d['layers'])!=len(d['centres_mm']):raise ValueError('One layer per rod required')
    if any(k not in (0,1) for k in d['layers']):raise ValueError('Only two layers are defined')
    if d['tooth_pitch_mm']<tc['minimum_tooth_pitch_mm']:raise ValueError('Tooth pitch must be >=0.5 mm')
    for layer in [0,1]:
        w=d['widths_mm'][layer];b=d['depths_mm'][layer]
        if w<tc['minimum_core_mm']-1e-9 or b<=0:raise ValueError('Every rod must have a manufacturable serrated edge')
        centers=sorted(y for y,k in zip(d['centres_mm'],d['layers']) if k==layer)
        if len(centers)>1 and min(np.diff(centers))<w+2*b+tc['minimum_same_layer_gap_mm']+.01-1e-9:
            raise ValueError('Same-layer rods violate envelope clearance')
    return True


def layer_duty(ff,d,layer,shift=0.,width_error=0.,tip=None):
    centres=np.array([y+shift for y,k in zip(d['centres_mm'],d['layers']) if k==layer])
    if len(centres)==0:return np.ones_like(ff)
    distance=abs(np.asarray(ff)[:,None]-centres[None,:])
    if 'rod_widths_mm' in d:
        ids=np.flatnonzero(np.array(d['layers'])==layer)
        widths=np.array(d['rod_widths_mm'])[ids]+width_error
        depths=np.array(d['rod_depths_mm'])[ids]
        t=np.min(np.clip((distance-widths[None,:]/2)/depths[None,:],0,1),axis=1)
    else:
        distance=np.min(distance,axis=1)
        t=np.clip((distance-(d['widths_mm'][layer]+width_error)/2)/d['depths_mm'][layer],0,1)
    cutoff=(d.get('tip_mm',config()['minimum_tip_mm']) if tip is None else tip)/d['tooth_pitch_mm']
    return np.where(t<cutoff,0,np.where(t>1-cutoff,1,t))


def interval_overlap(a,b,phase):
    """Mean product of two binary transparent intervals on a unit circle."""
    phase=phase%1.;result=np.zeros(np.broadcast(a,b).shape)
    for k in [-1,0,1]:
        result+=np.maximum(0,np.minimum(a/2,phase+k+b/2)-np.maximum(-a/2,phase+k-b/2))
    return result


def projected_transmission(ff,cfg,d,shift=0.,width_error=0.):
    a=layer_duty(ff,d,0,shift,width_error);b=layer_duty(ff,d,1,shift,width_error)
    leak=np.sqrt(cfg['filter']['blocked_intensity_transmission'])
    overlap=interval_overlap(a,b,d['relative_phase'])
    return leak**2+leak*(1-leak)*(a+b)+(1-leak)**2*overlap


def coefficients(ff,cfg,d,kmax):
    orders=np.arange(-kmax,kmax+1)
    leak=np.sqrt(cfg['filter']['blocked_intensity_transmission'])
    result=[]
    for layer in [0,1]:
        duty=layer_duty(ff,d,layer)
        c=(1-leak)*duty[:,None]*np.sinc(duty[:,None]*orders[None,:])
        c[:,kmax]+=leak
        phase=0 if layer==0 else d['relative_phase']
        result.append(c*np.exp(-2j*np.pi*phase*orders)[None,:])
    return orders,result


def phase_fresnel(axis,z,cfg):
    return np.exp(-1j*np.pi*z*np.asarray(axis)**2/(cfg['optics']['wavelength_mm']*cfg['optics']['objective_focal_length_mm']**2))


def zero_order_stack(field,axis,ff,x,cfg,d,kmax=12,y_offset=0.):
    """Propagate all retained -k,+k paths to total zero order.

    field shape (y, independent scan positions / object cases); x is the actual
    physical camera-line x coordinate of each column, not the scanned object
    coordinate. For a fixed line camera x=u for every scan position s.
    y_offset shifts the object's absolute field while axis stays target-relative.
    """
    if field.ndim==1:field=field[:,None]
    dtype=np.complex64 if field.dtype==np.complex64 else np.complex128
    x=np.broadcast_to(np.asarray(x),field.shape[1:])
    z1,z2=d['z_mm'];dz=z2-z1
    delta=m.scales(cfg)['alpha_mm2']/d['tooth_pitch_mm']
    orders,cs=coefficients(ff,cfg,d,kmax)
    y=axis+y_offset
    cs=[c.astype(dtype) for c in cs]
    incoming=ft(field*phase_fresnel(y,z1,cfg).astype(dtype)[:,None])
    result=np.zeros_like(field,dtype=dtype)
    between=phase_fresnel(y,dz,cfg).astype(dtype)
    for j,k in enumerate(orders):
        intermediate=ift(incoming*cs[0][:,len(orders)-1-j,None])
        propagated=ift(ft(intermediate*between[:,None])*cs[1][:,j,None])
        result+=propagated*phase_fresnel(x+k*delta,dz,cfg).astype(dtype)[None,:]
    result*=phase_fresnel(y,-z2,cfg)[:,None]
    result*= (phase_fresnel(x,z1,cfg)*phase_fresnel(x,-z2,cfg))[None,:]
    return result


def source_fields(cfg,arr):
    # The background proxy keeps complex amplitudes, not |spectrum| alone.
    sc=m.scales(cfg);fxmax=np.sqrt(np.maximum(0,sc['cutoff_per_mm']**2-arr['nu']**2))
    from scipy.special import erf
    transfer=erf(np.pi*cfg['camera']['illumination_intensity_fwhm_mm']*fxmax/np.sqrt(2*np.log(2)))
    axis=arr['axis'];edge=cfg['object']['field_mm']/2;taper=cfg['object']['line_edge_taper_mm']
    envelope=.5+.5*np.cos(np.pi*np.clip((abs(axis)-(edge-taper))/taper,0,1))
    wrap=np.remainder(axis+cfg['object']['pitch_mm']/2,cfg['object']['pitch_mm'])-cfg['object']['pitch_mm']/2
    fields=[ift(ft((abs(wrap)<w/2)*envelope)*transfer) for w in [cfg['object']['feature_width_mm'],.0049]]
    for size,spec in zip(cfg['object']['defect_fwhm_um'],arr['targets']):
        scale=size/1000*np.sqrt(np.pi/(2*np.log(2)))/(np.sqrt(arr['n'])*arr['dx'])
        fields.append(ift(spec)*scale)
    return np.column_stack(fields)


def field_measures(axis,field,cfg,diameter):
    intensity=m.pixel_average_1d(abs(field)**2,axis[1]-axis[0],m.scales(cfg)['pixel_object_mm'])
    center=len(axis)//2;peak=intensity[center]
    halves=[]
    for sign in [-1,1]:
        v=intensity[center::sign];r=abs(axis[center::sign])
        j=np.flatnonzero(v<=peak/2)
        j=int(j[0]) if len(j) else len(v)-1
        halves.append(float(np.interp(peak/2,[v[j],v[j-1]],[r[j],r[j-1]])))
    total=float(intensity.sum());idx=np.argsort(abs(axis));cum=np.cumsum(intensity[idx])
    w90=2*abs(axis[idx[min(int(np.searchsorted(cum,.9*total)),len(idx)-1)]])*1000
    guard=max(.00075,1.5*diameter/1000)
    return dict(fwhm_um=1000*sum(halves),ee90_um=float(w90),energy=total,
                halo=float(intensity[abs(axis)>guard].sum()/max(total,1e-30)),peak=float(peak))


def score_fields(cfg,arr,baseline,fields):
    # Both signs are required: axial propagation need not preserve even real PSF.
    background=max(float(np.sum(abs(fields[:,i])**2)/np.sum(abs(baseline[:,i])**2)) for i in range(2))
    defects=[];ret=[];width=[];ee=[];halo=[]
    for i,size in enumerate(cfg['object']['defect_fwhm_um'],2):
        a=field_measures(arr['axis'],fields[:,i],cfg,size);b=field_measures(arr['axis'],baseline[:,i],cfg,size)
        ret.append(a['energy']/b['energy']);width.append(a['fwhm_um']/b['fwhm_um']);ee.append(a['ee90_um']/b['ee90_um']);halo.append(a['halo'])
        defects.append(dict(defect_um=size,retention=ret[-1],**a,baseline_fwhm_um=b['fwhm_um'],baseline_ee90_um=b['ee90_um']))
    sep=np.linspace(.001,.025,481);e=fields[:,2];center=abs(e[len(e)//2])
    cr=[]
    for sign in [-1,1]:
        ratio=np.interp(sign*sep,arr['axis'],abs(e))/max(center,1e-30)
        cr.append(2*ratio+ratio**2)
    cross=float(np.max(cr));r=min(ret);s=-10*np.log10(max(background,1e-30));h=max(halo);w=max(width);we=max(ee)
    score=s+10*np.log10(max(r,1e-30))-10*np.log10(1+h/.05)-10*np.log10(1+cross/.10)-10*np.log10(1+max(we-1,0))
    return dict(suppression_db=float(s),minimum_retention=float(r),maximum_fwhm_ratio=float(w),maximum_ee90_ratio=float(we),
                maximum_halo=float(h),maximum_crosstalk=cross,score=float(score),feasible=bool(r>=.6 and w<=1.1),defects=defects)


def generate_designs(cfg,tc):
    pitch=m.scales(cfg)['order_pitch_mm']
    layouts=[]
    for count,step in itertools.product(tc['rod_counts'],tc['spacing_multiples']):
        centers=(np.arange(count)-count//2)*pitch*step
        if max(abs(centers))<18:layouts.append((centers.tolist(),f'equal_{count}_{step:g}'))
    layouts.append(([k*pitch for k in [-5,-3,-2,-1,0,1,2,3,5]],'old_positions'))
    for (centres,name),w,b,ratio,phase in itertools.product(layouts,tc['core_widths_mm'],tc['single_side_depths_mm'],tc['second_layer_width_scales'],tc['relative_tooth_phases']):
        d=dict(centres_mm=centres,layers=[i%2 for i in range(len(centres))],widths_mm=[w,max(.1,w*ratio)],
               depths_mm=[b,b*ratio],relative_phase=phase,tooth_pitch_mm=.75,z_mm=[-2.,2.],
               layout=name,rod_count=len(centres),profile='triangle')
        try:check_design(d,tc)
        except ValueError:continue
        yield d


def prescreen():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=m.config();tc=config();arr=m.search_arrays(cfg,tc['prescreen_n'],tc['prescreen_window_mm'])
    started=time.time();rows=[]
    for number,d in enumerate(generate_designs(cfg,tc),1):
        r=m.evaluate_candidate(cfg,arr,d,transmission_function=projected_transmission)
        r['candidate_id']=number;rows.append(r)
        if number%500==0:print(f'Projection prescreen: {number} / elapsed {time.time()-started:.1f}s',flush=True)
    feasible=[r for r in rows if r['feasible']]
    eligible=[r for r in feasible if r['worst_background_suppression_db']>=tc['background_target_db']]
    chosen=[]
    # Keep different layouts/widths, not only phase-near-duplicates.
    for pool,key in [(eligible,'score_db'),(feasible,'score_db'),(feasible,'worst_background_suppression_db')]:
        quota=tc['prescreen_shortlist']//2 if key=='score_db' else 2
        added=0
        for r in sorted(pool,key=lambda v:v[key],reverse=True):
            signature=(r['layout'],tuple(r['widths_mm']),tuple(r['depths_mm']))
            if any((v['layout'],tuple(v['widths_mm']),tuple(v['depths_mm']))==signature for v in chosen):continue
            chosen.append(r);added+=1
            if added>=quota:break
    save_csv(OUT/'projection_candidates.csv',rows)
    save_json(OUT/'projection_search.json',dict(config=cfg,stack_config=tc,candidate_count=len(rows),feasible_count=len(feasible),
              target_count=len(eligible),shortlist=chosen,elapsed_seconds=time.time()-started,
              caveat='Coplanar projected binary union only; NOT final layered performance. Tooth period .75 mm only fixes tip duty in this prescreen.'))
    print('Projection count',len(rows),'eligible',len(eligible),'shortlist',len(chosen),flush=True)
    return chosen


def design_only(r):
    return {k:r[k] for k in ['centres_mm','layers','widths_mm','depths_mm','relative_phase','tooth_pitch_mm','z_mm','layout','rod_count','profile']}


def reference_design():
    # Reference-dimension control, not a reverse-engineered Puma prescription.
    return dict(centres_mm=(np.arange(-4,5)*4.).tolist(),layers=[i%2 for i in range(9)],
        widths_mm=[1.1,1.1],depths_mm=[1.45,1.45],relative_phase=0.,tooth_pitch_mm=.75,
        z_mm=[-2.,2.],layout='reference_dimensions',rod_count=9,profile='triangle')


def refine():
    cfg=m.config();tc=config();pre=json.loads((OUT/'projection_search.json').read_text(encoding='utf-8'))
    arr=m.search_arrays(cfg,tc['refine_n'],tc['refine_window_mm']);base=source_fields(cfg,arr)
    rows=[];started=time.time()
    for source,p,gap in itertools.product(pre['shortlist'],tc['tooth_pitches_mm'],tc['axial_gaps_mm']):
        d={**design_only(source),'tooth_pitch_mm':p,'z_mm':[-gap/2,gap/2]}
        field=zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,d,tc['harmonic_kmax'])
        r=score_fields(cfg,arr,base,field);r.update(design=d,source_candidate=source['candidate_id'],refine_id=len(rows)+1)
        rows.append(r)
        if len(rows)%8==0:print(f'Layer refinement: {len(rows)} / {time.time()-started:.1f}s',flush=True)
    feasible=[r for r in rows if r['feasible']]
    eligible=[r for r in feasible if r['suppression_db']>=tc['background_target_db']]
    selected=max(eligible or feasible,key=lambda r:r['score']) if feasible else max(rows,key=lambda r:r['score'])
    strongest=max(feasible or rows,key=lambda r:r['suppression_db'])
    save_json(OUT/'layer_refinement.json',dict(rows=rows,selected=selected,background_priority=strongest,
        candidate_count=len(rows),elapsed_seconds=time.time()-started,kmax=tc['harmonic_kmax'],
        caveat='Finite-harmonic total-zero Fresnel field, nominal dimensions at centre. Direct all-order and off-axis verification follows.'))
    save_csv(OUT/'layer_refinement.csv',[{k:v for k,v in r.items() if k!='defects'} for r in rows])
    for name,row in [('selected',selected),('background_priority',strongest)]:
        f=zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,row['design'],tc['validation_kmax'])
        np.savez_compressed(OUT/f'{name}_profiles.npz',axis_mm=arr['axis'],baseline=base,fields=f)
        save_json(OUT/f'{name}_refined_metrics.json',score_fields(cfg,arr,base,f))
    d=reference_design();f=zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,d,tc['validation_kmax'])
    save_json(OUT/'reference_dimensions_metrics.json',dict(design=d,**score_fields(cfg,arr,base,f)))
    np.savez_compressed(OUT/'reference_dimensions_profiles.npz',axis_mm=arr['axis'],baseline=base,fields=f)
    print('Selected layered candidate:',json.dumps(selected,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prescreen','refine'])
    args=parser.parse_args()
    prescreen() if args.stage=='prescreen' else refine()
