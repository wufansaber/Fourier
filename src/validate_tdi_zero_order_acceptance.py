"""Audit the TDI zero-order-only geometry and generated artifacts."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tdi_zero_order_acceptance'
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
assert math.isclose(s['order_shift_image_um'],284.0,rel_tol=0,abs_tol=1e-10)
assert math.isclose(s['order_shift_object_um'],14.2,rel_tol=0,abs_tol=1e-10)
assert math.isclose(s['no_margin_Hmax_99pct_um'],294.53650233440175,rel_tol=0,abs_tol=1e-9)
assert s['current_serration_order_direction'].startswith('image x')
assert s['current_rod_width_diffraction_direction'].startswith('image y')
assert s['illumination_line_direction'].startswith('image y')
assert s['illumination_line_width_direction'].startswith('image x')
rows=list(csv.DictReader((OUT/'tdi_linewidth_limits.csv').open(encoding='utf-8-sig')))
strict=next(r for r in rows if float(r['central_energy_fraction'])==.9999 and float(r['one_side_margin_um'])==10)
assert 134.9<float(strict['max_tdi_active_height_um'])<135.0
assert strict['passes_25_rows']=='True' and strict['passes_256_rows']=='False'
for name in ['tdi_zero_order_linewidth_condition.png','tdi_zero_order_linewidth_condition.svg',
             'rod_serration_tdi_direction_audit.png','rod_serration_tdi_direction_audit.svg',
             'tdi_linewidth_limits.csv','tooth_pitch_linewidth_trade.csv','summary.json']:
    assert (OUT/name).exists()
text=(ROOT/'三通道傅里叶滤波_原理思路.md').read_text(encoding='utf-8')
assert text.count('$$')%2==0
assert 'H<134.95' in text and '最多启用26行' in text
print('PASS: 284 um order shift, 25/256-stage limits, Gaussian containment widths and artifacts checked')
