"""Independent rod tolerances and continuous-motion collision checks."""
import copy
import itertools
import json
import numpy as np
import joint_adjustable as j
import stacked_serrated as s
from ideal_na06 import save_json,save_csv


def trials(d,seed=355):
    rng=np.random.default_rng(seed);edge=j.manufacture(d['widths_mm'][0],d['depths_mm'][0],d['process'])['edge_tolerance_mm']
    pos=j.config()['mechanics']['position_tolerance_mm']
    yield d,0.,'nominal'
    for core,tip,shift in itertools.product([-edge,edge],[-edge,edge],[-pos,pos]):
        trial=copy.deepcopy(d);trial['centres_mm']=[y+shift for y in d['centres_mm']]
        trial['rod_widths_mm']=[d['widths_mm'][0]+2*core]*9
        trial['rod_depths_mm']=[d['depths_mm'][0]+tip-core]*9
        yield trial,0.,'common_corner'
    for offset in [-.2,.2]:yield d,offset,'off_axis'
    for _ in range(24):
        trial=copy.deepcopy(d);trial['centres_mm']=(np.array(d['centres_mm'])+rng.uniform(-pos,pos,9)).tolist()
        core=rng.uniform(-edge,edge,9);tip=rng.uniform(-edge,edge,9)
        trial['rod_widths_mm']=(d['widths_mm'][0]+2*core).tolist()
        trial['rod_depths_mm']=(d['depths_mm'][0]+tip-core).tolist()
        yield trial,float(rng.choice([-.2,0,.2])),'independent_random'


def run():
    tc=j.config();ref=json.loads((j.OUT/'refined.json').read_text());rows=[]
    for num,r in enumerate(ref['rows'],1):
        d=r['design'];cfg,a,base=j.arrays(d['period_um'],8192);cases=[]
        for t,offset,kind in trials(d):
            field=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,t,12)
            if offset:
                field[:,2:]=s.zero_order_stack(base[:,2:],a['axis'],a['ff'],0.,cfg,t,12,y_offset=offset)
            metric=s.score_fields(cfg,a,base,field);metric.update(kind=kind,target_offset_y_mm=offset)
            cases.append(metric)
        worst=dict(suppression_db=min(v['suppression_db'] for v in cases),minimum_retention=min(v['minimum_retention'] for v in cases),
            maximum_halo=max(v['maximum_halo'] for v in cases),maximum_crosstalk=max(v['maximum_crosstalk'] for v in cases),
            maximum_fwhm_ratio=max(v['maximum_fwhm_ratio'] for v in cases),maximum_ee90_ratio=max(v['maximum_ee90_ratio'] for v in cases))
        rows.append(dict(search_id=r['search_id'],design=d,worst=worst,trials=len(cases),cases=cases,
            gate_pass={name:j.gate_ok(worst,g,refined=True) for name,g in tc['gates'].items()}))
        print(f'Independent rod tolerance check {num}/{len(ref["rows"])}: p={d["period_um"]}, Smin={worst["suppression_db"]:.3f}',flush=True)
    save_json(j.OUT/'robustness.json',dict(rows=rows,random_seed=355,
        perturbation='Nominal + 8 common core/tip/position corners + 2 off-axis + 24 independent random samples; not exhaustive worst-case proof.',
        shape_error='core and tip edge errors each +/-epsilon; w=w0+2*core, b=b0+tip-core; outer envelope error stays +/-2epsilon.',
        offset_note='Defects shifted to +/-0.20 mm; finite background stays at its actual centered field.'))
    mechanical_verification(ref)


def mechanical_verification(ref):
    mc=j.config()['mechanics'];body_gap=mc['body_z_outset_mm']*2+j.config()['axial_gap_mm']-mc['body_thickness_z_mm']-2*mc['body_position_tolerance_mm']
    variants=[]
    for w in ref['selection']:
        if w['best'] is None:continue
        recipes=w['best']['recipes'];path_checks=[]
        for a,b in itertools.combinations(recipes,2):
            da=j.optical_design(a);db=j.optical_design(b)
            endpoints=min(j.mechanical(d)['minimum_same_layer_clearance_mm'] for d in [da,db]);sampled=1e9
            for fraction in np.linspace(0,1,101):
                moved={**da,'centres_mm':((1-fraction)*np.array(da['centres_mm'])+fraction*np.array(db['centres_mm'])).tolist()}
                gap=j.mechanical(moved)['minimum_same_layer_clearance_mm'];sampled=min(sampled,gap)
            assert sampled>=endpoints-1e-10
            path_checks.append(dict(from_period_um=a['period_um'],to_period_um=b['period_um'],
                minimum_sampled_gap_mm=sampled,analytic_lower_bound_mm=endpoints))
        variants.append(dict(process=w['process'],gate=w['gate'],rod_geometry_mm=[w['best']['w_mm'],w['best']['b_mm']],
            worst_recipe={key:min(j.mechanical(j.optical_design(r))[key] for r in recipes) for key in
                ['minimum_same_layer_clearance_mm','stroke_margin_mm','cross_layer_rod_clearance_mm','stage_lane_clearance_mm','stage_to_50mm_opening_clearance_mm']},
            path_checks=path_checks))
    save_json(j.OUT/'mechanical_verification.json',dict(variants=variants,
        cross_layer_stage_body_clearance_mm=body_gap,
        station_x_mm=mc['station_x_mm'],stage_width_x_mm=mc['stage_width_x_mm'],
        claimed_envelope_mm=[156,76,20],installed_rods=9,independent_y_axes=9,paired_end_sliders=18,
        assumed_material_proof_stress_mpa=mc['required_material_proof_stress_mpa'],
        caveat='Defined boxes/stems/rod envelopes and synchronized interpolation only; NOT a verified supplier CAD assembly, FEA, collision-free arbitrary asynchronous motion, or measured dynamic cross coupling.'))


if __name__=='__main__':run()
