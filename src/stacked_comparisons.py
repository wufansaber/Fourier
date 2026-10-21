"""Manufacturing-size controls and explicitly qualified TDI point acceptance."""
import ast
import csv
import json
import numpy as np
from scipy import ndimage
import stacked_serrated as s
import ch3_linescan as m
from ideal_na06 import save_json,save_csv
from tdi_compare import gate


def run():
    cfg=m.config();tc=s.config();arr=m.search_arrays(cfg,tc['refine_n'],tc['refine_window_mm']);base=s.source_fields(cfg,arr)
    with (s.OUT/'projection_candidates.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for key in ['centres_mm','layers','widths_mm','depths_mm','relative_phase','tooth_pitch_mm','z_mm','rod_count']:
            r[key]=ast.literal_eval(r[key])
    pool=[r for r in rows if r['feasible']=='True' and min(r['widths_mm'])>=.3 and min(r['depths_mm'])>=.2]
    # Large-dimension candidate selection uses the same R/FWHM guardrails,
    # maximizing background suppression, not claiming overall superiority.
    r=max(pool,key=lambda v:float(v['worst_background_suppression_db']));candidates=[]
    for p in tc['tooth_pitches_mm']:
        for gap in tc['axial_gaps_mm']:
            d={**s.design_only(r),'tooth_pitch_mm':p,'z_mm':[-gap/2,gap/2]}
            f=s.zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,d,24)
            candidates.append(dict(design=d,**s.score_fields(cfg,arr,base,f)))
    chosen=max([r for r in candidates if r['feasible']],key=lambda r:r['suppression_db'])
    f=s.zero_order_stack(base,arr['axis'],arr['ff'],0.,cfg,chosen['design'],24)
    np.savez_compressed(s.OUT/'manufacturing_profiles.npz',axis_mm=arr['axis'],baseline=base,fields=f)
    old=json.loads((m.OUT/'optimization.json').read_text(encoding='utf-8'))['choices']['selected']
    oldfield=s.ift(s.ft(base)*m.mean_transmission(arr['ff'],cfg,old)[:,None])
    np.savez_compressed(s.OUT/'old_single_layer_profiles.npz',axis_mm=arr['axis'],baseline=base,fields=oldfield)
    save_json(s.OUT/'controls.json',dict(manufacturing_priority=chosen,manufacturing_candidate_count=len(candidates),
        large_dimension_prescreen_feasible_count=len(pool),old_single_layer_same_nominal_metric=s.score_fields(cfg,arr,base,oldfield),
        old_design=old,manufacturing_rule='core >=0.30 mm; each one-sided depth >=0.20 mm. Illustrative manufacturability tier, not a supplier guarantee.'))
    z=np.load(s.OUT/'all_order_point.npz');axis=z['axis_mm'];h=z['field'];hn=z['baseline'];pointref=float(z['point_peak_reference'])
    d=json.loads((s.OUT/'layer_refinement.json').read_text())['selected']['design'];delta=m.scales(cfg)['alpha_mm2']/d['tooth_pitch_mm']
    maps={};metrics=[];orders=[];dx=axis[1]-axis[0]
    for stages in [25,256]:
        height=stages*.005/20;width=.00625;g=gate(axis,height,width)
        for name,field in [('none',hn),('all',h)]:
            a=abs(field)**2*g[None,:]/pointref
            a=ndimage.convolve1d(ndimage.convolve1d(a,[.125,.75,.125],axis=0),[.125,.75,.125],axis=1)
            maps[f'{stages}_{name}']=a
            metrics.append(dict(stages=stages,filter=name,central_gate=float(gate(0,height,width)),
                integrated_point_response=float(a.sum()*dx**2),scan_halo_outside_2um_fraction=float(a[:,abs(axis)>.002].sum()/a.sum())))
        for k in range(-4,5):
            band=abs(axis-k*delta)<.002;a=maps[f'{stages}_all']
            orders.append(dict(stages=stages,order=k,shift_um=k*delta*1000,gate=float(gate(k*delta,height,width)),
                band_energy_fraction=float(a[:,band].sum()/a.sum())))
    np.savez_compressed(s.OUT/'tdi_acceptance_approx.npz',axis_um=axis*1000,**maps)
    save_json(s.OUT/'tdi_acceptance_approx.json',dict(metrics=metrics,orders=orders,
        tooth_pitch_mm=d['tooth_pitch_mm'],delta_um=delta*1000,illumination_fwhm_um=6.25,
        model='Frozen on-axis all-order point response multiplied by the synchronised TDI collection gate. Thin-screen scalar approximation, NOT full moving-field TDI validation.',
        intensity_reference='one common unfiltered point peak and infinite illumination exposure; NOT main scene I/I_ref',
        limitation='Two-layer filters are not strictly shift invariant. Full position-dependent moving-charge propagation is outside this comparison. No N-dependent blur kernel is applied.'))
    save_csv(s.OUT/'tdi_order_acceptance.csv',orders)
    print('Manufacturing control:',json.dumps(chosen),flush=True)


if __name__=='__main__':run()
