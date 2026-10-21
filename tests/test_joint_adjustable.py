import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joint_adjustable as j
import stacked_serrated as s


def test_nine_installed_rods_including_parking():
    for n in [1,3,5,7,9]:
        orders=list(range(-(n//2),n//2+1));d=j.design(.1,.05,'precision',orders,20)
        assert len(d['centres_mm'])==d['rod_count']==9
        assert sum(d['active_flags'])==n
        for y,active in zip(d['centres_mm'],d['active_flags']):
            assert abs(y)<18 if active else abs(y)>=20
        for layer in [0,1]:assert np.all(np.diff(np.array(d['centres_mm'])[layer::2])>0)
        assert j.mechanical(d)['feasible']


def test_process_rules_not_just_tooth_period():
    assert j.manufacture(.1,.05,'precision')['feasible']
    assert not j.manufacture(.1,.05,'standard')['feasible']
    assert j.manufacture(.1,.1,'standard')['feasible']
    assert j.manufacture(.1,.1,'standard')['tip_mm']>=.05


def test_large_teeth_can_violate_mechanical_clearance_at_large_period():
    narrow=j.design(.1,.05,'precision',list(range(-4,5)),40)
    broad=j.design(.2,.2,'precision',list(range(-4,5)),40)
    assert j.mechanical(narrow)['feasible']
    assert not j.mechanical(broad)['feasible']


def test_homotopy_motion_preserves_same_layer_clearance():
    a=j.design(.1,.05,'precision',[-5,-3,-2,-1,0,1,2,3,5],10)
    b=j.design(.1,.05,'precision',[-1,0,1],40)
    endpoints=min(j.mechanical(d)['minimum_same_layer_clearance_mm'] for d in [a,b])
    for u in np.linspace(0,1,101):
        d={**a,'centres_mm':((1-u)*np.array(a['centres_mm'])+u*np.array(b['centres_mm'])).tolist()}
        assert j.mechanical(d)['minimum_same_layer_clearance_mm']>=endpoints-1e-12


def test_physical_layered_propagation_counts_parked_rods():
    cfg,a,base=j.arrays(20,8192);d=j.design(.1,.05,'precision',[-1,0,1],20)
    full=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,d,12)
    active={**d,'centres_mm':[y for y,f in zip(d['centres_mm'],d['active_flags']) if f],
            'layers':[k for k,f in zip(d['layers'],d['active_flags']) if f]}
    reduced=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,active,12)
    assert np.linalg.norm(full-reduced)/np.linalg.norm(full)<1e-5


def test_string_model_accounts_for_tooth_mass():
    d=j.design(.1,.05,'precision',list(range(-4,5)),20);r=j.mechanical(d)
    assert r['mass_to_minimum_core_ratio']==pytest.approx(.15/.08)
    L=.142;rho=8000;stress=100e6
    expected=rho*(.15/.08)*2*9.81*L*L/(8*stress)*1e6
    assert r['maximum_installed_2g_deflection_um']==pytest.approx(expected)
    assert r['force_per_rod_n']==pytest.approx(.4)


def test_asymmetric_even_rod_recipes_preserve_hardware_order():
    import itertools
    pool=[-5,-3,-2,-1,1,2,3,5]
    for bits in itertools.product([False,True],repeat=8):
        orders=sorted([0]+[k for k,on in zip(pool,bits) if on])
        d=j.design(.1,.05,'precision',orders,20)
        assert len(d['centres_mm'])==9
        assert sum(d['active_flags'])==len(orders)
        actual=sorted(y for y,on in zip(d['centres_mm'],d['active_flags']) if on)
        assert actual==pytest.approx(np.array(orders)*.5325)
        for layer in [0,1]:assert np.all(np.diff(np.array(d['centres_mm'])[layer::2])>0)
        assert j.mechanical(d)['feasible']


def test_independent_edge_errors_bound_core_and_outer_envelope():
    from joint_robustness import trials
    d=j.design(.1,.05,'precision',[-5,0,3,5],20)
    cases=list(trials(d))
    assert len(cases)==35
    for t,offset,kind in cases:
        w=np.array(t.get('rod_widths_mm',[.1]*9))
        b=np.array(t.get('rod_depths_mm',[.05]*9))
        assert np.max(abs(w-.1))<=.020000000001
        assert np.max(abs(w+2*b-.2))<=.020000000001
        assert np.min(b)>0
        assert np.max(abs(np.array(t['centres_mm'])-d['centres_mm']))<=.025000000001


def test_per_rod_width_extension_matches_uniform_default():
    cfg,a,base=j.arrays(20,8192)
    d=j.design(.1,.05,'precision',[-5,0,3,5],20)
    per={**d,'rod_widths_mm':[.1]*9,'rod_depths_mm':[.05]*9}
    expected=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,d,12)
    actual=s.zero_order_stack(base,a['axis'],a['ff'],0.,cfg,per,12)
    np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-12)
