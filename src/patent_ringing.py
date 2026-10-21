"""US7397557B2 piecewise profiles, analytic transforms and numerical checks.

Lengths mm. Fourier convention integral B(u) exp(-2 pi i q u) du.
Figures are recalculations at this project's wavelength/focal scale, not scans
or pixel reproductions of the patent figures. No patent legal conclusions.
"""
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ch3_linescan import OUT, config, scales
from ideal_na06 import save_csv, save_json


def opacity(u,a,b,profile):
    if b==0:
        return (np.abs(u)<=a).astype(float) if isinstance(u,np.ndarray) else float(abs(u)<=a)
    s=np.clip((np.abs(u)-a)/b,0,1)
    return 1-(s if profile=="linear" else np.sin(np.pi*s/2))


def transform_quad(q,a,b,profile):
    # Split at the core edge so integration never crosses a derivative jump.
    core=2*a*np.sinc(2*a*q)
    if b==0:
        return core
    return core+2*quad(lambda u:float(opacity(u,a,b,profile))*np.cos(2*np.pi*q*u),a,a+b,
                       epsabs=2e-12,limit=300)[0]


def transform(q,a,b,profile):
    q=np.asarray(q,dtype=float)
    if b==0 or profile=="hard":
        return 2*a*np.sinc(2*a*q)
    if profile=="linear":
        return (2*a+b)*np.sinc((2*a+b)*q)*np.sinc(b*q)
    denominator=1-(4*b*q)**2
    numerator=2*(a+b)*np.sinc(2*(a+b)*q)-4*b/np.pi*np.cos(2*np.pi*a*q)
    out=np.divide(numerator,denominator,out=np.zeros_like(q),where=np.abs(denominator)>1e-8)
    singular=np.abs(denominator)<=1e-8
    if np.any(singular):
        out=np.array(out,copy=True)
        out[singular]=[transform_quad(v,a,b,profile) for v in q[singular].flat]
    return out


def figures():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    alpha=scales(config())["alpha_mm2"]
    checks=[]
    for profile in ["linear","sine"]:
        for a,b in [(.5,1),(.05,.1),(.1,.05)]:
            for q in [0,.1,.5,1/(4*b),3.7,10,37]:
                an=float(transform(q,a,b,profile)); numeric=transform_quad(q,a,b,profile)
                checks.append(dict(profile=profile,a_mm=a,b_mm=b,q_per_mm=q,analytic=an,quadrature=numeric,error=abs(an-numeric)))
    assert max(r["error"] for r in checks)<1e-9
    save_csv(OUT/"patent_transform_checks.csv",checks)
    x=np.linspace(0,150,12001);q=x/1000/alpha
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout="constrained")
    for profile,label in [("hard","Hard"),("linear","Linear"),("sine","Sine")]:
        b=0 if profile=="hard" else 1
        axes[0].plot(x,np.log10(np.maximum(abs(transform(q,.5,b,profile)),1e-7)),label=label,lw=1)
    for ax,profile in zip(axes[1:],["linear","sine"]):
        for b in [0,.5,1,1.5,2]:
            ax.plot(x,np.log10(np.maximum(abs(transform(q,.5,b,profile)),1e-7)),label=f"b={b:g} mm",lw=.9)
        ax.set_title(f"{profile.capitalize()}: a=0.5 mm, variable b")
    axes[0].set_title("a=0.5 mm, b=1 mm (hard: b=0)")
    for ax in axes:
        ax.set(xlabel="Object-equivalent distance (um)",ylabel="log10 |B transform / 1 mm|",ylim=(-5,.6),xlim=(0,150))
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle(f"Patent Fig.8-10-style recalculation | 355 nm, effective Fourier f={scales(config())['f_ff_mm']:g} mm")
    fig.savefig(OUT/"patent_fig8_10_recalculated.png",dpi=180);plt.close(fig)
    # Fixed far-distance band, same unnormalised incident field. Increasing b
    # changes blocked area too, so retain that area as a separate column.
    xfar=np.linspace(25,250,30001);qfar=xfar/1000/alpha
    rows=[]
    for a in [.05,.10,.50]:
        hard=np.trapezoid(abs(transform(qfar,a,0,"hard"))**2,xfar)
        for b in ([0,.025,.05,.075,.10,.15,.20] if a<.5 else [0,.5,1,1.5,2]):
            for profile in (["hard"] if b==0 else ["linear","sine"]):
                energy=np.trapezoid(abs(transform(qfar,a,b,profile))**2,xfar)
                rows.append(dict(a_mm=a,core_width_mm=2*a,b_mm=b,profile=profile,outer_width_mm=2*(a+b),
                                 opacity_area_mm=float(transform(0,a,b,profile)),
                                 tail_band_um="25-250",tail_energy_ratio_to_same_core_hard=float(energy/hard),
                                 tail_reduction_db=float(-10*np.log10(energy/hard))))
    save_csv(OUT/"patent_size_scan.csv",rows)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for a in [.05,.1]:
        for profile,style in [("linear","--"),("sine","-")]:
            r=[r for r in rows if r["a_mm"]==a and r["profile"]==profile]
            axs[0].plot([v["b_mm"] for v in r],[v["tail_reduction_db"] for v in r],style,marker="o",label=f"core={2*a:g} mm, {profile}")
    xx=np.linspace(0,100,10001)
    for profile,b in [("hard",0),("linear",.1),("sine",.1)]:
        axs[1].semilogy(xx,np.maximum(abs(transform(xx/1000/alpha,.05,b,profile))**2,1e-12),label=f"{profile}, b={b:g}",lw=1)
    axs[0].set(xlabel="One-side transition b (mm)",ylabel="25-250 um tail reduction vs same-core hard (dB)")
    axs[1].set(xlabel="Object-equivalent distance (um)",ylabel="Unnormalised diffraction |B transform|^2 (mm^2)",ylim=(1e-10,.1))
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Single rod only: less distant ringing is NOT proof of low periodic-array crosstalk")
    fig.savefig(OUT/"patent_actual_size_effect.png",dpi=180);plt.close(fig)
    save_json(OUT/"patent_validation.json",dict(maximum_quadrature_error=max(r["error"] for r in checks),checks=len(checks),
        eq36_direct_read_counterexample=.4052847345693511,eq31_correct_counterexample=float(transform(.5,.5,1,"linear")),
        zero_order_energy_linear=1/3,zero_order_energy_sine=.5,alpha_mm2=alpha))


if __name__=="__main__":
    figures()
