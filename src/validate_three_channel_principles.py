"""Small numerical and artifact audit for the three-channel principles note."""
import csv
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((ROOT/'results/three_channel_principles/channel_centres_and_spacing.csv').open(encoding='utf-8-sig')))
assert len(rows)==6
normal=[r for r in rows if r['case']=='normal']
oblique=[r for r in rows if r['case']=='oblique75_out_of_screen']
assert len(normal)==len(oblique)==3
expected=math.sin(math.radians(75))/.000355
assert all(abs(float(r['qy_cycles_per_mm'])+expected)<1e-9 for r in oblique)
assert all(abs(float(r['qy_cycles_per_mm']))<1e-12 for r in normal)
assert {int(r['nearest_m_for_20um']) for r in normal}=={-40,0,40}
assert {int(r['nearest_n_for_20um']) for r in oblique}=={-54}
assert float(next(r for r in normal if r['channel']=='Ch1')['qx_cycles_per_mm'])>0
assert float(next(r for r in normal if r['channel']=='Ch2')['qx_cycles_per_mm'])<0
translations=list(csv.DictReader((ROOT/'results/three_channel_principles/tilt_translation_by_channel.csv').open(encoding='utf-8-sig')))
assert len(translations)==3
assert all(abs(float(r['delta_qx_cycles_per_mm']))<1e-12 for r in translations)
assert all(abs(float(r['delta_qy_cycles_per_mm'])+expected)<1e-9 for r in translations)
assert all(int(r['nearest_cancel_n'])==-54 for r in translations)
spacing=list(csv.DictReader((ROOT/'results/three_channel_principles/oblique75_local_bfp_peak_spacing.csv').open(encoding='utf-8-sig')))
assert len(spacing)==3
assert all(abs(float(r['delta_Y_mm'])-.5325)<1e-10 for r in spacing)
assert abs(float(next(r for r in spacing if r['channel']=='Ch3')['delta_X_plus_mm'])-.5325)<1e-10
assert abs(float(next(r for r in spacing if r['channel']=='Ch3')['delta_X_minus_mm'])-.5325)<1e-10
assert any(abs(float(r['delta_X_plus_mm'])-float(r['delta_X_minus_mm']))>1e-3 for r in spacing if r['channel']!='Ch3')
text=(ROOT/'三通道傅里叶滤波_原理思路.md').read_text(encoding='utf-8')
assert text.count('$$')%2==0
assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',text)
assert 'pattern_dimensionality_channel_entry.png' not in text
for path in ['wavevector_direction_convention.png','three_channel_frequency_coverage.png',
             'tilt_spectrum_translation_by_channel.png','normal_and_oblique_pattern_entry.png',
             'oblique75_local_bfp_peak_spacing.png','oblique75_local_bfp_peak_spacing.csv',
             'local_bfp_spacing_vs_rigid_rod.png','channel_centres_and_spacing.csv',
             'tilt_translation_by_channel.csv']:
    assert (ROOT/'results/three_channel_principles'/path).exists()
print('PASS: independent-channel 75-degree translation, normal incidence, formulas, CSV and figures checked')
