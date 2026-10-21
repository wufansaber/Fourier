"""Two recipes on the SAME recommended independently adjustable hardware.

Metrology additionally constrains EE90 growth; detection explicitly does not
claim unchanged defect size. This script reuses no robustness claims from the
different b=.075-mm strict-comparator hardware.
"""
import json
import numpy as np
import joint_adjustable as j
import stacked_serrated as s
from joint_robustness import trials
from ideal_na06 import save_json


def run():
    src=json.loads((j.OUT/'search.json').read_text());sel=json.loads((j.OUT/'selection.json').read_text());ref=json.loads((j.OUT/'refined.json').read_text())
    background=next(v['best'] for v in sel['winners'] if v['process']=='precision' and v['gate']=='background_priority')
    w=background['w_mm'];b=background['b_mm'];rows=[]
    for period in j.config()['periods_um']:
        pool=[r for r in src['rows'] if r['period_um']==period and r['process']=='precision' and r['widths_mm'][0]==w and r['depths_mm'][0]==b
              and r['mechanics']['dynamic_model_ok'] and j.gate_ok(r,j.config()['gates']['strict']) and r['maximum_ee90_ratio']<=2.0]
        if not pool:
            rows.append(dict(period_um=period,mode='metrology',available=False));continue
        r=max(pool,key=lambda x:x['worst_background_suppression_db']);d=j.optical_design(r);cfg,a,base=j.arrays(period,16384)
        f=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,d,24);nom=s.score_fields(cfg,a,base,f)
        np.savez_compressed(j.OUT/f'metrology_{period}_profiles.npz',axis_mm=a['axis'],baseline=base,fields=f)
        cfg,a,base=j.arrays(period,8192);cases=[]
        for trial,offset,kind in trials(d):
            field=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,trial,12)
            if offset:field[:,2:]=s.zero_order_stack(base[:,2:],a['axis'],a['ff'],0.,cfg,trial,12,y_offset=offset)
            cases.append(s.score_fields(cfg,a,base,field))
        worst={key:(min if key in ['suppression_db','minimum_retention'] else max)(v[key] for v in cases) for key in
               ['suppression_db','minimum_retention','maximum_halo','maximum_crosstalk','maximum_fwhm_ratio','maximum_ee90_ratio']}
        passes=j.gate_ok(worst,j.config()['gates']['strict'],True) and worst['maximum_ee90_ratio']<=2
        rows.append(dict(period_um=period,mode='metrology',available=True,search_id=r['search_id'],design=d,nominal=nom,worst=worst,
            sampled_guard_pass=passes,mechanics=j.mechanical(d),trials=len(cases)))
        print(f'Same-hardware metrology p={period}, S={nom["suppression_db"]:.3f}, tolerance pass={passes}',flush=True)
    for r in background['recipes']:
        v=next(v for v in ref['rows'] if v['search_id']==r['search_id'])
        rows.append(dict(period_um=r['period_um'],mode='detection',available=True,search_id=r['search_id'],design=v['design'],nominal=v,
            mechanics=v['mechanics']))
    save_json(j.OUT/'operating_modes.json',dict(core_mm=w,one_sided_depth_mm=b,process='precision',rows=rows,
        caveat='One fixed geometry; positions alone change. Metrology gate also EE90 <=2x baseline. Detection does not certify defect-size fidelity. Perturbation sampling is not an exhaustive tolerance guarantee.'))


if __name__=='__main__':run()
