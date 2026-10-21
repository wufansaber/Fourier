"""Collect evidence status and lint the delivered Markdown without fabricating results."""
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'real_objective'

def main():
    metrics=json.loads((OUT/'native_pop'/'image_metrics.json').read_text())
    f=metrics.get('fine_sampling',{})
    native=dict(engine='Local OpticStudio 2022 R2.02 native POP',
                native_2048_image_cases=4,all_2048_cases_have_grids=all((OUT/'native_pop'/(name+'_s28_n2048.npz')).exists() for name in ('real_regular_open','real_regular_rods','real_defect_open','real_defect_rods')),
                coarse_roi_suppression_db=metrics['regular_suppression_db'],
                fine_sampling=f,relative_suppression_comparison_limit_db=.5,
                relative_suppression_within_half_db=bool(f and abs(f.get('change_from_2048_db',999))<.5),
                absolute_power_change_percent=(100*(f['real_regular_open']['power_W']/metrics['real_regular_open']['power_W']-1) if f else None),
                absolute_power_convergence='Not established: about 10 percent change; source edge sampling requires further checks',
                native_fourier_peaks_mm=metrics.get('native_fourier_peaks_mm'),
                native_defect_field_energy_retention='Not calculated: exported native grids contain intensity, not complex field',
                joint_model_suppression_db=40.53927309561285,
                difference_attribution='Not uniquely established. Joint model is on-axis/shift-invariant with nominal hard pupil; native POP propagates full finite object surface-by-surface with actual surface apertures. Do not substitute the joint figure for native performance.')
    (OUT/'native_pop_status.json').write_text(json.dumps(native,indent=2),encoding='utf-8')
    p=OUT/'native_pop'/'real_defect_open_s28_n2048.json'
    if not p.exists():
        data=dict(metrics['real_defect_open'],model='real_defect_open',sampling=2048,end_surface=28,
                  source='probe_real_pop.py, explicit IAS_PhysicalOpticsPropagation and TopHat; output copied from diagnostics/scan_s28',messages=[],elapsed_s=None)
        p.write_text(json.dumps(data,indent=2),encoding='utf-8')
    source_cfg=OUT/'diagnostics'/'real_pop_settings.CFG'
    if source_cfg.exists():shutil.copy2(source_cfg,OUT/'native_pop'/'real_defect_open_s28_n2048.CFG')
    report=ROOT/'真实物镜傅里叶滤波_建模与验证.md'
    md=report.read_text(encoding='utf-8')
    no_code=re.sub(r'```.*?```','',md,flags=re.S)
    displays=re.findall(r'^\$\$$',no_code,flags=re.M)
    assert len(displays)%2==0
    assert not any(s in no_code for s in (r'\[',r'\]',r'\(',r'\)'))
    missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',md):
        if re.match(r'^[A-Za-z]:/',target) and not Path(target).exists():missing.append(target)
    assert not missing,missing
    for script in list((ROOT/'zemax').glob('*real*.py'))+list((ROOT/'src').glob('*real*.py')):
        ast.parse(script.read_text(encoding='utf-8'))
    hashes={}
    for folder in (OUT/'models',OUT/'uda',OUT/'native_pop'):
        for file in folder.iterdir():
            if file.suffix.lower() in ('.zos','.zmx','.uda','.json','.cfg'):
                hashes[str(file.relative_to(OUT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    qa=dict(markdown_display_formulas=len(displays)//2,missing_local_links=missing,
            python_source_parse='passed',math_delimiters='passed; renderer-specific visual compatibility not asserted',
            all_native_cases_present=native['all_2048_cases_have_grids'])
    (OUT/'delivery_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(native,indent=2));print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
