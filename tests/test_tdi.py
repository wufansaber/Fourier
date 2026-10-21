from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tdi_compare import gate,tooth_coefficient


@pytest.mark.parametrize("height",[.00625,.064])
def test_tdi_gate_is_intensity_integral(height):
    w=.00625;normalization=w*np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))
    for x in [0,.001,.019722222,.039444444]:
        q=quad(lambda u:np.exp(-4*np.log(2)*(u-x)**2/w**2),-height/2,height/2,epsabs=1e-14)[0]/normalization
        assert gate(x,height,w)==pytest.approx(q,abs=1e-12)


def test_25_to_256_accepts_first_tooth_order():
    d=.019722222222;w=.00625
    assert gate(d,.00625,w)<1e-8
    assert gate(d,.064,w)>.999
    assert gate(0,.064,w)/gate(0,.00625,w)==pytest.approx(1.3141,abs=1e-4)
    assert gate(0,.064,w)/gate(0,.00625,w)<256/25


def test_synchronised_pair_cross_term_not_optical_field_average():
    w=.00625;height=.064;d=.01972222;x=d
    a=2*np.log(2)/w**2;norm=w*np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))
    # Neighbour at 0, target at d; arbitrary complex PSF sample values.
    h1=.08+.02j;h2=1+.1j
    direct=quad(lambda u:abs(h1*np.exp(-a*(u-x)**2)+h2*np.exp(-a*(u-x+d)**2))**2,
                -height/2,height/2,epsabs=1e-13)[0]/norm
    expected=(abs(h1)**2*gate(x,height,w)+abs(h2)**2*gate(x-d,height,w)
              +2*np.real(h1*np.conj(h2))*np.exp(-np.log(2)*d*d/w**2)*gate(x-d/2,height,w))
    assert direct==pytest.approx(expected,abs=1e-10)


def test_tooth_coefficients_obey_binary_power_not_zero_order_only():
    t=.37;k=np.arange(1,20001)
    total=tooth_coefficient(t,0,0)**2+2*np.sum(tooth_coefficient(t,k,0)**2)
    assert total==pytest.approx(t,abs=1e-5)
    assert tooth_coefficient(t,0,0)**2==pytest.approx(t*t)
