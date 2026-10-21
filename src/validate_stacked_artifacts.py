"""Checks measured output consistency, constraints, units and Markdown links."""
import csv
import json
import re
import hashlib
import sys
import xml.etree.ElementTree as ET
import numpy as np
import stacked_serrated as s
import ch3_linescan as m
from ideal_na06 import save_json


def run():
    cfg=m.config();ref=m.camera_intensity_reference(cfg)
    pre=json.loads((s.OUT/'projection_search.json').read_text())
    opt=json.loads((s.OUT/'layer_refinement.json').read_text())
    with (s.OUT/'projection_candidates.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==pre['candidate_count']==9636
    assert len(opt['rows'])==144
    for r in opt['rows']:s.check_design(r['design'])
    selected=opt['selected']['design'];assert selected['rod_count']==len(selected['centres_mm'])==9
    assert m.scales(cfg)['pupil_mm']==m.scales(cfg)['filter_pupil_mm']==36
    with (s.OUT/'rod_dimensions.csv').open(encoding='utf-8-sig') as f:rods=list(csv.DictReader(f))
    assert len(rods)==9 and all(float(r['tooth_pitch_mm'])>=.5 and float(r['one_sided_depth_mm'])>0 for r in rods)
    assert (s.OUT/'rod_silhouettes_not_for_release.dxf').read_text().count('LWPOLYLINE')==9
    checks=[]
    for case in ['dots','stripes']:
        metrics=json.loads((s.OUT/f'{case}_scan_n8192_metrics.json').read_text())
        assert metrics['intensity_reference']==ref and metrics['intensity_unit']=='I/I_ref'
        z=np.load(s.OUT/f'{case}_scan_n8192.npz');axis=z['axis_um']
        assert axis[0]==-45 and axis[-1]==45
        for key in ['unfiltered','filtered','background','defects_only','d1','d2','d3']:
            v=z[key];assert np.isfinite(v).all() and v.min()>=0
            assert v.max()<=1.2,(case,key,v.max())
            checks.append(dict(case=case,array=key,maximum=float(v.max())))
        np.testing.assert_allclose(z['difference'],z['filtered']-z['background'],atol=1e-7)
        for j,(x,y,size) in enumerate(m.DEFECTS,1):
            ix=np.argmin(abs(axis-x*1000));iy=np.argmin(abs(axis-y*1000))
            assert z[f'd{j}'][iy,ix]>.1
    grid=json.loads((s.OUT/'image_grid_check.json').read_text())
    assert grid['metrics']['filtered']['relative_intensity_l2']<.01
    assert grid['metrics']['d1']['relative_intensity_l2']<.01
    testroot=ET.parse(s.OUT/'pytest.xml').getroot()
    suites=list(testroot.iter('testsuite'))
    counts={key:sum(int(v.attrib.get(key,0)) for v in suites) for key in ['tests','failures','errors','skipped']}
    assert counts['tests']>=58 and counts['failures']==counts['errors']==0
    doc=s.ROOT/'全锯齿双层傅里叶滤波_优化结果.md';body=doc.read_text(encoding='utf-8')
    assert len(re.findall(r'^\$\$$',body,re.M))%2==0
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in body)
    links=re.findall(r'\]\(([^)]+)\)',body);local=[s.ROOT/p for p in links if not p.startswith(('http:','https:'))]
    missing=[str(p) for p in local if not p.exists() and p.name!='artifact_validation.json']
    assert not missing,missing
    save_json(s.OUT/'artifact_validation.json',dict(status='passed',tests=counts,projection_candidates=len(rows),
        layered_candidates=len(opt['rows']),physical_rods=9,optical_pupil_mm=36,tooth_pitch_mm=selected['tooth_pitch_mm'],
        intensity_reference=ref,unit='I/I_ref',linear_limits=[0,1.2],all_three_original_defects_visible=True,
        local_links=len(local),array_checks=checks,grid_checks=grid['metrics'],python_executable=sys.executable,
        python_version=sys.version,numpy_version=np.__version__,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (s.ROOT/'src').glob('stacked_*.py')},
        caveat='Artifact and scalar-numerical checks only. Manufacturing release and full moving-field 256-row TDI validation are not certified.'))
    print(f'PASS: {counts["tests"]} tests, {len(rows)} prescreen / {len(opt["rows"])} layered candidates, 9 rods, {len(local)} document links.')


if __name__=='__main__':run()
