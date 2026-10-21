"""Asymmetric/even active-rod extension on the same recommended hardware."""
import itertools
import json
import copy
import numpy as np
import joint_adjustable as j
import stacked_serrated as s
from joint_robustness import trials
from ideal_na06 import save_json,save_csv


def run():
    c=j.config();rows=[];pool=[-5,-3,-2,-1,1,2,3,5]
    for period in c['periods_um']:
        cfg,a,base=j.arrays(period,8192)
        for bits in itertools.product([False,True],repeat=len(pool)):
            orders=sorted([0]+[k for k,t in zip(pool,bits) if t]);d=j.design(.1,.05,'precision',orders,period)
            if not j.mechanical(d)['feasible'] or not j.mechanical(d)['dynamic_model_ok']:continue
            cfg['filter']['core_full_width_tolerance_mm']=.02
            r=j.m.evaluate_candidate(cfg,a,d,transmission_function=s.projected_transmission)
            r.update(extension_id=len(rows)+1);rows.append(r)
        print(f'Asymmetric recipes p={period}: {len(rows)} evaluated',flush=True)
    save_json(j.OUT/'asymmetric_search.json',dict(rows=rows,count=len(rows),installed_rods=9,hardware=[.1,.05]))
    save_csv(j.OUT/'asymmetric_search.csv',rows)
    # Guard-band front screen, then actual finite-layer independent-error tests.
    # Candidates from the previous symmetric family remain eligible.
    previous=json.loads((j.OUT/'search.json').read_text())['rows']
    eligible_old=[r for r in previous if r['process']=='precision' and r['widths_mm'][0]==.1 and r['depths_mm'][0]==.05 and r['mechanics']['dynamic_model_ok']]
    presets=[]
    modes=dict(metrology=dict(halo=.10,crosstalk=.10,retention=.60,ee90=2.),
               balanced=dict(halo=.20,crosstalk=.15,retention=.60,ee90=None))
    for period,mode in itertools.product(c['periods_um'],modes):
        gate=modes[mode]
        candidates=[r for r in rows+eligible_old if r['period_um']==period and r['worst_retention']>=.65
            and r['maximum_halo_fraction']<=gate['halo']*.84 and r['worst_pair_intensity_perturbation']<=gate['crosstalk']*.8
            and (gate['ee90'] is None or r['maximum_ee90_ratio']<=1.8)]
        candidates.sort(key=lambda r:r['worst_background_suppression_db'],reverse=True)
        # Keep one of mirrored/equivalent order sets only for trial budget.
        unique=[];seen=set()
        for r in candidates:
            o=tuple(r['object_orders']);signature=min(o,tuple(sorted(-k for k in o)))
            if signature in seen:continue
            seen.add(signature);unique.append(r)
        tested=[];chosen=None
        for r in unique[:5]:
            cfg,a,base=j.arrays(period,8192);d=j.optical_design(r);cases=[]
            for trial,offset,kind in trials(d):
                f=s.zero_order_stack(base,a['axis'],a['ff'],0,cfg,trial,12)
                if offset:f[:,2:]=s.zero_order_stack(base[:,2:],a['axis'],a['ff'],0,cfg,trial,12,y_offset=offset)
                cases.append(s.score_fields(cfg,a,base,f))
            worst={key:(min if key in ['suppression_db','minimum_retention'] else max)(v[key] for v in cases) for key in
                ['suppression_db','minimum_retention','maximum_halo','maximum_crosstalk','maximum_fwhm_ratio','maximum_ee90_ratio']}
            passed=j.gate_ok(worst,gate,True) and (gate['ee90'] is None or worst['maximum_ee90_ratio']<=gate['ee90'])
            tested.append(dict(orders=d['object_orders'],worst=worst,pass_all=passed))
            if passed:
                cfg,a,base=j.arrays(period,16384);f=s.zero_order_stack(base,a['axis'],a['ff'],0,cfg,d,24)
                chosen=dict(design=d,nominal=s.score_fields(cfg,a,base,f),worst=worst,mechanics=j.mechanical(d),
                    trials=len(cases),pass_all=True)
                np.savez_compressed(j.OUT/f'final_{mode}_{period}_profiles.npz',axis_mm=a['axis'],baseline=base,fields=f)
                break
        presets.append(dict(period_um=period,mode=mode,gate=gate,selected=chosen,tests=tested,
            candidates_after_margin=len(unique)))
        print(f'Guarded {mode} p={period}: {None if chosen is None else (chosen["design"]["object_orders"],round(chosen["worst"]["suppression_db"],3))}',flush=True)
    save_json(j.OUT/'final_presets.json',dict(hardware=dict(w_mm=.1,b_mm=.05,thickness_mm=.05,tooth_pitch_mm=.75,gap_mm=4,process='precision'),
        presets=presets,extension_count=len(rows),
        rule='Manufacturing and physical envelopes first; nominal guard band; up to 5 distinct symmetry classes per period/mode tested with 35 perturbations. Highest screen-S passing tested candidate, not global robust optimum.',
        important='Balanced preset does not constrain EE90 growth. Metrology requires EE90 <=2x. All installed rods including parked rods count towards 9.'))


if __name__=='__main__':run()
