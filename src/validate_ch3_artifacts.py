"""Cross-check delivered run data, finite rods, common image units and MD links."""
import csv
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import ch3_linescan as m
from ch3_display import I_MAX,LIMITS
from ideal_na06 import save_json


def validate():
    cfg=m.config();out=m.OUT;reference=m.camera_intensity_reference(cfg)
    opt=json.loads((out/'optimization.json').read_text(encoding='utf-8'))
    orders=m.active_orders(cfg,opt['choices']['selected'])
    with (out/'candidate_scan.csv').open(encoding='utf-8-sig') as f:candidates=list(csv.DictReader(f))
    assert len(candidates)==opt['candidate_count']
    assert max(int(r['rod_count']) for r in candidates)<=cfg['filter']['maximum_rods']
    with (out/'fourier_rod_positions.csv').open(encoding='utf-8-sig') as f:rods=list(csv.DictReader(f))
    assert len(rods)==len(orders)==9
    assert [int(r['object_order_n']) for r in rods]==orders
    assert np.allclose([float(r['centre_y_mm']) for r in rods],np.array(orders)*m.scales(cfg)['order_pitch_mm'])
    svg=ET.parse(out/'selected_rods_checkplot.svg').getroot()
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}polygon'))==len(orders)
    assert (out/'selected_rods_checkplot.dxf').read_text(encoding='ascii').count('LWPOLYLINE')==len(orders)
    for p in out.glob('*_full_metrics.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        assert r['intensity_unit']=='I/I_ref (dimensionless)'
        assert r['intensity_reference']==reference
        assert r['derived']['filter_pupil_mm']==36
        assert len(m.active_orders(cfg,r['design']))<=10
    panels=[]
    for case in ['dots','stripes']:
        for choice in ['none_fine','selected_fine']:
            with np.load(out/f'{case}_{choice}_scan.npz') as z:
                axis=z['axis_um'];ids=np.flatnonzero((axis>=LIMITS[0])&(axis<=LIMITS[1]))
                for name in ['actual','reference','defects_only','isolated']:
                    a=z[name]
                    assert np.all(np.isfinite(a)) and a.min()>=0
                    maximum=float(a.max())
                    assert maximum<=I_MAX,f'{case}/{choice}/{name} exceeds common colour limit'
                    panels.append(dict(case=case,choice=choice,array=name,maximum=maximum))
                assert np.allclose(z['image_difference'],z['actual']-z['reference'],atol=2e-7)
                for x,y,d in m.DEFECTS:
                    assert axis[ids[0]]<1000*x<axis[ids[-1]]
                    assert axis[ids[0]]<1000*y<axis[ids[-1]]
    doc=m.ROOT/'傅里叶滤波_原理到Python仿真及最优结果_综合报告.md'
    text=doc.read_text(encoding='utf-8')
    assert len(re.findall(r'^\$\$$',text,re.M))%2==0
    links=re.findall(r'\]\(([^)]+)\)',text)
    local=[v for v in links if re.match(r'^[A-Za-z]:/',v)]
    missing=[v for v in local if not Path(v).exists()]
    assert not missing,missing
    assert '尚未生成' not in text
    save_json(out/'artifact_validation.json',dict(status='passed',physical_rods=len(orders),candidate_count=len(candidates),
        maximum_candidate_rods=max(int(r['rod_count']) for r in candidates),optical_pupil_mm=36,
        camera_intensity_reference=reference,intensity_unit='I/I_ref (dimensionless)',linear_colour_limits=[0,I_MAX],
        all_three_defects_in_common_frame=True,local_links_checked=len(local),image_array_checks=panels,
        limitations='Does not prove high-NA vector accuracy or hardware manufacturability; numerical grid differences remain reported.'))
    print(f'Artifact QA passed: {len(candidates)} candidates, {len(orders)} rods, {len(local)} local links, common image unit/reference.')


if __name__=='__main__':validate()
