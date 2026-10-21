"""Optical/manufacturing/mechanical constrained search for 9 movable rods.

Installed count includes parked rods. Identical rod geometry is held fixed when
changing object period; only independently driven positions are changed.
"""
import argparse
import copy
import itertools
import json
import time
from functools import lru_cache
import numpy as np
import yaml
import ch3_linescan as m
import stacked_serrated as s
from stacked_images import tapered_interval
from ideal_na06 import save_json,save_csv

OUT=m.ROOT/'results/joint_adjustable'


@lru_cache(maxsize=1)
def config():return yaml.safe_load((m.ROOT/'config/joint_adjustable.yml').read_text(encoding='utf-8'))


def manufacture(w,b,process,p=None):
    c=config();p=p or c['tooth_pitch_mm'];e=c['processes'][process]['edge_tolerance_mm'];t=c['processes'][process]['foil_thickness_mm']
    radius=t;angle=np.arctan(2*b/p)
    required_flat=2*radius*np.tan(angle/2)+.005
    tip=max(.020,2*e,required_flat)
    margins=[(w-2*e)/.050,b/(4*e),tip/(2*e),p/.5]
    return dict(process=process,edge_tolerance_mm=e,foil_thickness_mm=t,tip_mm=float(tip),
        minimum_corner_radius_mm=radius,minimum_width_after_error_mm=w-2*e,
        minimum_margin=float(min(margins)),feasible=bool(min(margins)>=1-1e-9),
        corner_deviation_bound_mm=float(radius*(1/np.cos(angle/2)-1)))


def positions(orders,period_um):
    """Map requested active orders to fixed 5A+4B hardware IDs.

    Relative order of hardware on each plane never changes. Unused end rods
    are parked beyond the 18-mm pupil radius and still count as installed.
    """
    orders=sorted(orders);active=[[],[]]
    for i,k in enumerate(orders):active[(i+len(orders)//2)%2].append(k*m.scales(m.config())['alpha_mm2']/(period_um/1000))
    all_y=np.zeros(9);flags=np.zeros(9,dtype=bool)
    for layer,count in [(0,5),(1,4)]:
        ys=active[layer];unused=count-len(ys);left=unused//2;right=unused-left
        vals=[-20-j for j in reversed(range(left))]+ys+[20+j for j in range(right)]
        ids=np.arange(layer,9,2);all_y[ids]=vals
        flags[ids[left:left+len(ys)]]=True
    return all_y.tolist(),flags.tolist()


def design(w,b,process,orders,period_um,p=None):
    c=config();p=p or c['tooth_pitch_mm'];mf=manufacture(w,b,process,p);centers,active=positions(orders,period_um)
    return dict(centres_mm=centers,layers=[i%2 for i in range(9)],widths_mm=[w,w],depths_mm=[b,b],
        relative_phase=0.,tooth_pitch_mm=p,z_mm=[-c['axial_gap_mm']/2,c['axial_gap_mm']/2],
        layout='movable_'+','.join(map(str,orders)),rod_count=9,active_rods=len(orders),active_flags=active,
        object_orders=sorted(orders),period_um=period_um,profile='triangle',tip_mm=mf['tip_mm'],process=process)


def mechanical(d):
    c=config();mc=c['mechanics'];mf=manufacture(d['widths_mm'][0],d['depths_mm'][0],d['process'],d['tooth_pitch_mm'])
    e=mc['position_tolerance_mm'];edge=mf['edge_tolerance_mm'];envelope=max(d['widths_mm'][i]+2*d['depths_mm'][i] for i in [0,1])
    width=max(envelope+2*edge,mc['finger_width_y_mm'])
    gaps=[]
    for layer in [0,1]:
        ys=np.array([y for y,k in zip(d['centres_mm'],d['layers']) if k==layer])
        gaps.extend((np.diff(ys)-width-2*e).tolist())
    stroke=mc['stroke_half_y_mm']-max(abs(np.array(d['centres_mm'])))-e
    axial=np.diff(d['z_mm'])[0]-mf['foil_thickness_mm']-2*mc['axial_position_tolerance_mm']-2*mc['displacement_budget_um']/1000
    # Stage bodies' full swept-Y boxes never intersect because X stations are
    # disjoint. Opposite planes' stage bodies are on opposite outward sides.
    station=np.array(mc['station_x_mm']);body_x_gap=min(np.diff(station))-mc['stage_width_x_mm']-2*mc['body_position_tolerance_mm']
    clear_aperture=min(station)-mc['stage_width_x_mm']/2-25-mc['body_position_tolerance_mm']
    lengths=np.array([2*station[i//2] for i in range(9)])/1000
    core=mf['minimum_width_after_error_mm'];beta=(d['widths_mm'][0]+d['depths_mm'][0])/core
    stress=mc['working_stress_mpa']*1e6;rho=mc['density_kg_m3']
    deflections=rho*beta*mc['operating_acceleration_g']*9.81*lengths**2/(8*stress)*1e6
    frequencies=np.sqrt(stress/(rho*beta))/(2*lengths)
    active=np.array(d['active_flags'],dtype=bool)
    dynamics=dict(working_stress_mpa=mc['working_stress_mpa'],mass_to_minimum_core_ratio=float(beta),
        force_per_rod_n=float(core*mf['foil_thickness_mm']*mc['working_stress_mpa']),
        maximum_active_2g_deflection_um=float(max(deflections[active])),
        maximum_installed_2g_deflection_um=float(max(deflections)),minimum_string_frequency_hz=float(min(frequencies)),
        displacement_assumption_g=mc['operating_acceleration_g'],dynamic_model_ok=bool(max(deflections[active])<=mc['displacement_budget_um']),
        required_material_proof_stress_mpa=mc['required_material_proof_stress_mpa'])
    return dict(**dynamics,minimum_same_layer_clearance_mm=float(min(gaps)),stroke_margin_mm=float(stroke),
        cross_layer_rod_clearance_mm=float(axial),stage_lane_clearance_mm=float(body_x_gap),
        stage_to_50mm_opening_clearance_mm=float(clear_aperture),
        conventional_same_x_slider_clearance_mm=float(min(np.diff(np.array(d['centres_mm'])[::2]))-mc['stage_width_x_mm']-2*e),
        feasible=bool(min(gaps)>=mc['minimum_clearance_mm'] and stroke>=0 and axial>=.1 and body_x_gap>=.1 and clear_aperture>=.1))


def arrays(period_um,n):
    cfg=copy.deepcopy(m.config());cfg['object']['pitch_mm']=period_um/1000
    cfg['object']['feature_width_mm']=cfg['object']['pitch_mm']*.35
    a=m.search_arrays(cfg,n,config()['window_mm']);base=s.source_fields(cfg,a)
    nu=a['nu'];edge=cfg['object']['field_mm']/2;taper=cfg['object']['line_edge_taper_mm'];p=period_um/1000
    centers=np.arange(-int(edge/p),int(edge/p)+1)*p
    from scipy.special import erf
    fx=np.sqrt(np.maximum(0,m.scales(cfg)['cutoff_per_mm']**2-nu**2))
    tf=erf(np.pi*cfg['camera']['illumination_intensity_fwhm_mm']*fx/np.sqrt(2*np.log(2)))
    a['backgrounds']=[]
    for j,duty in enumerate(config()['duty_cycles']):
        radius=duty*p/2
        spec=sum(tapered_interval(nu,cy-radius,cy+radius,edge,taper) for cy in centers)/(np.sqrt(n)*a['dx'])*tf
        a['backgrounds'].append(abs(spec)**2);base[:,j]=s.ift(spec)
    return cfg,a,base


def layouts():
    pool=config()['order_pair_pool']
    for n in range(5):
        for pairs in itertools.combinations(pool,n):yield sorted([0]+list(pairs)+[-v for v in pairs])


def gate_ok(r,gate,refined=False):
    if refined:
        return (r['minimum_retention']>=gate['retention'] and r['maximum_halo']<=gate['halo'] and
                r['maximum_crosstalk']<=gate['crosstalk'] and r['maximum_fwhm_ratio']<=1.10)
    return (r['worst_retention']>=gate['retention'] and r['maximum_halo_fraction']<=gate['halo'] and
            r['worst_pair_intensity_perturbation']<=gate['crosstalk'] and r['maximum_fwhm_ratio']<=1.10)


def scan():
    OUT.mkdir(parents=True,exist_ok=True);tc=config();rows=[];start=time.time();counter=0
    for period in tc['periods_um']:
        cfg,a,base=arrays(period,tc['search_n'])
        for process,w,b,orders in itertools.product(tc['processes'],tc['core_widths_mm'],tc['depths_mm'],layouts()):
            mf=manufacture(w,b,process)
            if not mf['feasible']:continue
            d=design(w,b,process,orders,period);mech=mechanical(d)
            if not mech['feasible']:continue
            cfg['filter']['core_full_width_tolerance_mm']=2*mf['edge_tolerance_mm']
            r=m.evaluate_candidate(cfg,a,d,transmission_function=s.projected_transmission)
            # Save all actual evaluated candidates, not just passing optics.
            r.update(manufacturing=mf,mechanics=mech,search_id=len(rows)+1)
            rows.append(r)
            if len(rows)%500==0:print(f'Joint search {len(rows)} ({time.time()-start:.1f}s)',flush=True)
        print(f'Finished period {period} um: {len(rows)} total',flush=True)
    save_json(OUT/'search.json',dict(config=tc,rows=rows,candidate_count=len(rows),elapsed_s=time.time()-start,
        caveat='Projected binary screening; refined finite-gap fields and independent errors follow.'))
    save_csv(OUT/'search.csv',[{k:v for k,v in r.items() if k not in ['manufacturing','mechanics']} for r in rows])
    summarize(rows)


def summarize(rows):
    c=config();hardware=[];local=[]
    # A hardware tuple must serve EVERY period. Positions/layout may change,
    # but core, depth, thickness and process must remain unchanged.
    for process,w,b,gname in itertools.product(c['processes'],c['core_widths_mm'],c['depths_mm'],c['gates']):
        recipes=[]
        for period in c['periods_um']:
            pool=[r for r in rows if r['process']==process and r['period_um']==period and r['widths_mm'][0]==w and r['depths_mm'][0]==b and r['mechanics'].get('dynamic_model_ok',True) and gate_ok(r,c['gates'][gname])]
            if pool:
                top=max(pool,key=lambda r:r['worst_background_suppression_db']);recipes.append(top)
        if recipes:
            hardware.append(dict(process=process,w_mm=w,b_mm=b,gate=gname,period_count=len(recipes),
                covers_all_periods=len(recipes)==len(c['periods_um']),
                minimum_suppression_db=min(r['worst_background_suppression_db'] for r in recipes),
                mean_suppression_db=float(np.mean([r['worst_background_suppression_db'] for r in recipes])),recipes=recipes))
    for process,gname,period in itertools.product(c['processes'],c['gates'],c['periods_um']):
        pool=[r for r in rows if r['process']==process and r['period_um']==period and r['mechanics'].get('dynamic_model_ok',True) and gate_ok(r,c['gates'][gname])]
        local.append(dict(process=process,gate=gname,period_um=period,best=max(pool,key=lambda r:r['worst_background_suppression_db']) if pool else None))
    winners=[]
    for process,gname in itertools.product(c['processes'],c['gates']):
        pool=[h for h in hardware if h['process']==process and h['gate']==gname and h['covers_all_periods']]
        winners.append(dict(process=process,gate=gname,best=max(pool,key=lambda h:h['minimum_suppression_db']) if pool else None))
    save_json(OUT/'selection.json',dict(hardware=hardware,local_optima=local,winners=winners,
        rule='Manufacture and collision/stroke hard screens, then R/H/C/FWHM gate; max worst-period background suppression with identical rods.'))
    for w in winners:
        v=w['best'];print(w['process'],w['gate'],None if v is None else (v['w_mm'],v['b_mm'],v['minimum_suppression_db'],[r['active_rods'] for r in v['recipes']]),flush=True)


def qualify():
    path=OUT/'search.json';data=json.loads(path.read_text())
    for r in data['rows']:r['mechanics']=mechanical(optical_design(r))
    data['config']=config();save_json(path,data);summarize(data['rows'])


def optical_design(row):
    return {key:row[key] for key in ['centres_mm','layers','widths_mm','depths_mm','relative_phase','tooth_pitch_mm',
        'z_mm','layout','rod_count','active_rods','active_flags','object_orders','period_um','profile','tip_mm','process']}


def refine():
    tc=config();sel=json.loads((OUT/'selection.json').read_text());winners=[v for v in sel['winners'] if v['best'] is not None]
    targets={r['search_id']:r for w in winners for r in w['best']['recipes']}
    rows=[]
    for number,r in enumerate(targets.values(),1):
        cfg,a,base=arrays(r['period_um'],tc['refine_n']);d=optical_design(r)
        f=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,d,tc['check_kmax'])
        metric=s.score_fields(cfg,a,base,f)
        metric.update(search_id=r['search_id'],design=d,mechanics=mechanical(d),manufacturing=r['manufacturing'])
        rows.append(metric)
        np.savez_compressed(OUT/f'profiles_{r["search_id"]}.npz',axis_mm=a['axis'],baseline=base,fields=f)
        print(f'Joint layered refinement {number}/{len(targets)}: p={r["period_um"]}, active={r["active_rods"]}, S={metric["suppression_db"]:.3f}',flush=True)
    save_json(OUT/'refined.json',dict(rows=rows,selection=sel['winners'],caveat='Selected screen winners re-evaluated at the physical 4-mm layer gap; robust uncertainty follows.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scan','qualify','refine']);a=p.parse_args()
    dict(scan=scan,qualify=qualify,refine=refine)[a.stage]()
