"""Regression checks for the saved real-objective experiment and its evidence."""
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'real_objective'

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def test_source_unchanged_and_no_ideal_lens():
    q=read('engineering_qa.json')
    assert q['original_objective_preserved']
    assert q['all_standard_surfaces']
    assert q['original_source_sha256']==q['project_copy_sha256']

def test_all_sampled_rays_valid():
    for r in read('ray_validation_summary.json'):
        assert r['rays']==3209
        assert r['valid']==r['rays']

def test_measured_sine_and_order_errors():
    q=read('engineering_qa.json')
    assert q['finite_sine_relative_max_percent']<.1
    assert q['axial_fourier_residual_max_um']<3.
    assert max(q['order_residual_max_um_by_field'].values())<50.

def test_real_masks_and_mechanical_dimensions():
    assert read('uda_validation.json')['all_expected']
    d=read('filter_definition.json')
    assert len(d['rod_centers_mm'])==9
    assert d['rod_width_mm']==1.
    c=d['rod_centers_mm']
    assert min(b-a for a,b in zip(c,c[1:]))>2.33
    assert all(g['min_axial_thickness_mm']>1.5 for g in read('engineering_qa.json')['added_lens_geometry'])

def test_fourier_scales_and_sampling_sensitivity():
    assert math.isclose(.000584601*20./.005,2.338404,rel_tol=1e-10)
    coarse=read('coherent_metrics_2048_1.024_w1_a0.json')
    fine=read('coherent_metrics_4096_1.024_w1_a0.json')
    guard=read('coherent_metrics_3072_1.536_w1_a0.json')
    assert fine['dx_um']<.000584601/(2*.4998)*1000
    assert abs(coarse['suppression_db']-fine['suppression_db'])<.1
    assert abs(coarse['suppression_db']-guard['suppression_db'])<.5
    assert 55.<fine['defect_field_energy_retention_percent']<60.

def test_native_pop_is_separate_and_reports_sampling_limit():
    status=read('native_pop_status.json')
    assert status['all_2048_cases_have_grids']
    assert 16.<status['coarse_roi_suppression_db']<18.
    assert status['relative_suppression_within_half_db']
    assert 9.<status['absolute_power_change_percent']<11.
    assert 'Not established' in status['absolute_power_convergence']

def test_native_fourier_peaks_match_real_ray_positions_within_pixel():
    status=read('native_pop_status.json')
    expected=[-7.0160757726,-2.3385227067,0.,2.3385227067,7.0160757726]
    actual=status['native_fourier_peaks_mm']
    assert len(actual)==len(expected)
    assert max(abs(a-b) for a,b in zip(actual,expected))<.01142
