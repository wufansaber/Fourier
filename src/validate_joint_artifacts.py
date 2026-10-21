"""Audit final report artifacts against saved computations, not physical acceptance."""
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import joint_adjustable as j
import stacked_serrated as s
from ideal_na06 import save_json


def read(name):
    return json.loads((j.OUT/name).read_text(encoding='utf-8'))


def run():
    data=read('final_presets.json'); search=read('search.json'); ext=read('asymmetric_search.json')
    mech=read('final_mechanical_checks.json'); report=j.m.ROOT/'独立可调锯齿杆_三维联合最优方案.md'
    assert len(search['rows'])==search['candidate_count']==17923
    assert len(ext['rows'])==ext['count']==1280
    assert all(r['manufacturing']['feasible'] and r['mechanics']['feasible'] for r in search['rows'])
    assert len(data['presets'])==10
    with (j.OUT/'all_position_recipes.csv').open(encoding='utf-8-sig',newline='') as f:
        recipes=list(csv.DictReader(f))
    assert len(recipes)==90
    nominal_checks=[]; baselines={}
    for item in data['presets']:
        v=item['selected']; assert v and v['pass_all'] and v['trials']==35
        d=v['design']; gate=item['gate']; worst=v['worst']
        assert j.gate_ok(worst,gate,refined=True)
        if gate['ee90'] is not None:assert worst['maximum_ee90_ratio']<=gate['ee90']
        assert d['rod_count']==len(d['centres_mm'])==9
        assert sum(d['active_flags'])==d['active_rods']<=9
        assert d['widths_mm']==[.1,.1] and d['depths_mm']==[.05,.05]
        assert d['z_mm']==[-2.,2.] and d['tooth_pitch_mm']==.75 and d['tip_mm']>=.02
        mf=j.manufacture(.1,.05,d['process']); geometry=j.mechanical(d)
        assert mf['feasible'] and geometry['feasible'] and geometry['dynamic_model_ok']
        assert mf['foil_thickness_mm']==.05
        for i,(y,on) in enumerate(zip(d['centres_mm'],d['active_flags'])):
            if not on:assert abs(y)-.1-.025-.01>18
            row=next(r for r in recipes if r['mode']==item['mode'] and float(r['period_um'])==item['period_um'] and r['rod_id']==f'R{i+1}')
            assert abs(float(row['y_mm'])-y)<1e-12
            assert row['active'].lower()==str(on).lower()
        path=j.OUT/f'final_{item["mode"]}_{item["period_um"]}_profiles.npz'
        with np.load(path) as z:
            axis=z['axis_mm']; base=z['baseline']; fields=z['fields']
            assert len(axis)==16384 and fields.shape==base.shape
            assert np.isfinite(fields).all() and np.isfinite(base).all()
            period=item['period_um']
            if period in baselines:np.testing.assert_array_equal(base,baselines[period])
            else:baselines[period]=base.copy()
            cfg,a,recomputed_base=j.arrays(period,16384)
            np.testing.assert_allclose(base,recomputed_base,atol=1e-12,rtol=1e-11)
            metric=s.score_fields(cfg,a,base,fields)
            keys=['suppression_db','minimum_retention','maximum_halo','maximum_crosstalk','maximum_fwhm_ratio','maximum_ee90_ratio']
            delta=max(abs(metric[k]-v['nominal'][k]) for k in keys)
            assert delta<1e-9
            nominal_checks.append(dict(period_um=period,mode=item['mode'],maximum_saved_metric_difference=delta))
    assert mech['recipes']==10 and mech['motion_paths']==45
    assert mech['slider_swept_boxes']==18 and mech['box_pair_checks']==153 and mech['collisions']==0
    assert mech['minimum_rod_gap_mm']>=.1 and mech['minimum_slider_separating_gap_mm']>=.1
    text=report.read_text(encoding='utf-8')
    assert text.count('$$')%2==0
    assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',text)
    links=[]
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if target.startswith(('https://','http://','#')):continue
        path=report.parent/target
        if path.name!='artifact_validation.json':assert path.exists(),target
        links.append(target)
    xml=ET.parse(j.OUT/'pytest.xml').getroot()
    suites=list(xml.iter('testsuite'))
    tests=sum(int(x.get('tests',0)) for x in suites)
    failures=sum(int(x.get('failures',0))+int(x.get('errors',0)) for x in suites)
    assert tests>=67 and failures==0
    hashes={str(path.relative_to(j.m.ROOT)).replace('\\','/'):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [report,j.m.ROOT/'config/joint_adjustable.yml',*(j.m.ROOT/'src').glob('joint_*.py'),
                     Path(__file__),j.OUT/'final_presets.json',j.OUT/'final_mechanical_checks.json']}
    result=dict(status='PASS',timestamp=datetime.now().astimezone().isoformat(),tests=tests,failures=failures,
        screen_candidates=17923,extension_candidates=1280,candidate_families_overlap=True,
        tested_presets=10,perturbations_per_preset=35,physical_rods=9,position_csv_rows=90,
        independent_y_degrees_of_freedom=9,paired_end_sliders=18,motion_paths=45,
        mechanical_checks=mech,nominal_metric_recalculations=nominal_checks,
        common_baseline_across_modes=True,camera_intensity_reference=j.m.camera_intensity_reference(j.m.config()),
        local_report_links=links,sha256=hashes,
        limitations=['Audit of numerical artifacts only, not experimental validation.',
          '35 perturbations are not an exhaustive continuous worst-case proof.',
          'Mechanical dynamics use assumed material/stress/input; no supplier CAD, FEA or measured zero crosstalk.',
          '9-rod, fixed tooth pitch and layer gap candidate family; no global optimum proof.',
          'Final-net-zero main-line collection model, not all-order 256-stage TDI optimization.'])
    save_json(j.OUT/'artifact_validation.json',result)
    print(f'PASS: {tests} tests; 10 presets, 90 positions, 45 paths, 153 envelope pairs; shared intensity baseline.',flush=True)


if __name__=='__main__':run()
