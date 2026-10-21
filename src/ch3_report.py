"""Reproducible numerical checks, scientific figures and Chinese MD report.

Run after ch3_linescan.py optimization/full propagation and patent_ringing.py.
No Zemax calls. All headline values are read from calculation outputs.
"""
from pathlib import Path
import csv
import json
import gc
import ast
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ch3_linescan as m
from ideal_na06 import save_json,save_csv

ROOT=m.ROOT;OUT=m.OUT
COLORS={"none":"#666666","selected":"#087e8b","best_hard":"#c65b22","background_only":"#8a4f9e","ring_priority":"#4f6d3c"}
NAMES={"none":"No filter","selected":"Joint / >=15 dB","best_hard":"Hard / >=15 dB","background_only":"Background priority","ring_priority":"Ringing priority"}


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def savefig(fig,name):
    fig.savefig(OUT/name,dpi=170)
    plt.close(fig)


def validations(cfg,opt):
    arr=m.search_arrays(cfg,n=65536,window=2.048)
    design=m.design_parameters(opt["choices"]["selected"])
    fine=m.evaluate_candidate(cfg,arr,design)
    rows=[]
    for od in [np.inf,4,3,2]:
        c=json.loads(json.dumps(cfg));c["filter"]["blocked_intensity_transmission"]=0 if np.isinf(od) else 10.**-od
        r=m.evaluate_candidate(c,arr,design)
        rows.append(dict(od="opaque" if np.isinf(od) else od,**r))
    for b in [max(0,design["b_mm"]-.01),design["b_mm"]+.01]:
        rows.append(dict(od="4, depth tolerance",**m.evaluate_candidate(cfg,arr,{**design,"b_mm":b})))
    save_csv(OUT/"coating_depth_sensitivity.csv",rows)
    check=dict(refined_object_sampling=fine,refined_frequency_sampling=read("fine_search_validation.json"))
    base=read("dots_selected_full_metrics.json")
    check["full_grid_checks"]=[]
    for tag in ["selected_fine","selected_wide","selected_q5"]:
        r=read(f"dots_{tag}_full_metrics.json")
        check["full_grid_checks"].append(dict(tag=tag,n=r["n"],window_mm=r["window_mm"],dx_um=r["dx_um"],
            background_energy_change_percent=100*(r["reference_energy"]/base["reference_energy"]-1),
            smallest_defect_energy_change_percent=100*(r["defect_energies"][0]/base["defect_energies"][0]-1)))
    save_json(OUT/"validation_summary.json",check)
    return check


def optimization_figures(opt):
    with (OUT/"candidate_scan.csv").open(encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
    feasible=[r for r in rows if r["feasible"]=="True"]
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout="constrained")
    sc=axs[0].scatter([float(r["worst_background_suppression_db"]) for r in feasible],
        [100*float(r["worst_retention"]) for r in feasible],
        c=[100*float(r["maximum_halo_fraction"]) for r in feasible],s=13,cmap="viridis_r",vmin=0,vmax=40)
    fig.colorbar(sc,ax=axs[0],label="Maximum halo energy (%)")
    # Non-dominated front in S/R/C/halo (all four, not score alone).
    values=np.array([[float(r["worst_background_suppression_db"]),float(r["worst_retention"]),
                      -float(r["worst_pair_intensity_perturbation"]),-float(r["maximum_halo_fraction"])] for r in feasible])
    front=[]
    for i in np.argsort(-values[:,0]):
        v=values[i]
        if front and np.any(np.all(values[front]>=v,axis=1)):continue
        front=[j for j in front if not (np.all(v>=values[j]) and np.any(v>values[j]))]
        front.append(int(i))
    save_csv(OUT/"pareto_candidates.csv",[feasible[i] for i in front])
    axs[1].scatter(values[:,0],-values[:,2]*100,s=11,c="#bfc7cf",label="Candidates")
    axs[1].scatter(values[front,0],-values[front,2]*100,s=16,c="#566b84",label="4-metric non-dominated")
    for key in ["none","selected","best_hard","background_only","ring_priority"]:
        r=opt["choices"][key]
        for ax,y in [(axs[0],100*r["worst_retention"]),(axs[1],100*r["worst_pair_intensity_perturbation"])]:
            ax.scatter(r["worst_background_suppression_db"],y,color=COLORS[key],s=80,marker="D",edgecolor="white",zorder=5,
                       label=NAMES[key] if ax is axs[0] else None)
    axs[0].set(xlabel="Worst-case background suppression (dB)",ylabel="Worst-case defect retention (%)")
    axs[1].set(xlabel="Worst-case background suppression (dB)",ylabel="Worst pair-centre intensity perturbation (%)")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7,loc="best")
    fig.suptitle(f"{opt['candidate_count']} candidates | <=10 rods | ringing and crosstalk included")
    savefig(fig,"joint_optimization.png")
    # Fixed core and active orders isolate the transition-size effect.
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout="constrained")
    for profile,style in [("linear","--"),("sine","-")]:
        r=sorted([r for r in rows if float(r["core_width_mm"])==.1 and ast.literal_eval(r["orders"])==opt["choices"]["selected"]["orders"] and r["profile"]==profile],key=lambda r:float(r["b_mm"]))
        for ax,key,factor in zip(axs,["worst_background_suppression_db","maximum_halo_fraction","worst_pair_intensity_perturbation"],[1,100,100]):
            ax.plot([float(v["b_mm"]) for v in r],[factor*float(v[key]) for v in r],style,marker="o",label=profile)
    for ax,label in zip(axs,["Worst background suppression (dB)","Maximum halo energy (%)","Worst pair perturbation (%)"]):
        ax.set(xlabel="One-side transition b (mm)",ylabel=label);ax.grid(alpha=.2);ax.legend()
    fig.suptitle("Finite rod array: core=0.10 mm, selected order positions fixed | width sensitivity")
    savefig(fig,"array_size_effect.png")


def profile_figures(cfg,opt):
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    loaded={}
    # Search profiles store a unit-peak Gaussian SPECTRUM, not a unit-peak
    # object field. Restore its continuous Fourier amplitude before plotting.
    n=cfg["numerics"]["search_n"];dx=cfg["numerics"]["search_window_mm"]/n
    defect_mm=cfg["object"]["defect_fwhm_um"][0]/1000
    field_scale=defect_mm*np.sqrt(np.pi/(2*np.log(2)))/(np.sqrt(n)*dx)
    intensity_scale=field_scale**2/m.camera_intensity_reference(cfg)
    for key in ["none","selected","best_hard","background_only"]:
        z=np.load(OUT/f"{key}_line_profiles.npz");loaded[key]=z
        y=z["y_um"];e=z["fields"][0];i=m.pixel_average_1d(abs(e)**2,(y[1]-y[0])/1000,m.scales(cfg)["pixel_object_mm"])
        i=i*intensity_scale
        axs[0].plot(y,i,color=COLORS[key],label=NAMES[key]);axs[1].semilogy(y,np.maximum(i,1e-8),color=COLORS[key],label=NAMES[key])
        axs[2].semilogy(z["sep_um"],np.maximum(z["crosstalk"]*100,1e-5),color=COLORS[key],label=NAMES[key])
    axs[0].set(xlim=(-1.5,1.5),ylim=(0,1.2),xlabel="Along-line distance (um)",ylabel="I / I_ref (dimensionless)",title="0.5 um defect: centre-line profile")
    axs[1].set(xlim=(-35,35),ylim=(1e-7,1.2),xlabel="Along-line distance (um)",ylabel="I / I_ref (dimensionless)",title="Same reference: long tails")
    axs[2].set(xlim=(1,25),ylim=(1e-3,200),xlabel="Pair separation along line (um)",ylabel="Worst-phase centre perturbation (%)",title="Pair bound 2r+r^2 (pre-pixel)")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=7)
    savefig(fig,"defect_core_tail_crosstalk.png")
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for key in ["none","selected","best_hard","background_only"]:
        r=opt["choices"][key]["defect_metrics"]
        ds=[v["defect_fwhm_um"] for v in r]
        axs[0].plot(ds,[v["fwhm_um"] for v in r],"o-",color=COLORS[key],label=NAMES[key])
        axs[1].plot(ds,[v["ee90_width_um"] for v in r],"o-",color=COLORS[key],label=NAMES[key])
    axs[0].set(xlabel="Object intensity FWHM (um)",ylabel="Image core FWHM, object-equivalent (um)")
    axs[1].set(xlabel="Object intensity FWHM (um)",ylabel="90% energy full width along line (um)")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("FWHM alone misses defect enlargement by distant energy")
    savefig(fig,"defect_size_sweep.png")
    # Coherent double-defect fields, pixel-integrated AFTER field superposition.
    z=loaded["selected"];y=z["y_um"];e=z["fields"][0]
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout="constrained");pairs=[]
    pix=m.scales(cfg)["pixel_object_mm"];dx=(y[1]-y[0])/1000
    baseline=m.pixel_average_1d(abs(e)**2,dx,pix)[len(y)//2]
    for ax,d in zip(axs,[3.,20.]):
        e1=np.interp(y+d/2,y,e.real)+1j*np.interp(y+d/2,y,e.imag)
        e2=np.interp(y-d/2,y,e.real)+1j*np.interp(y-d/2,y,e.imag)
        incoherent=m.pixel_average_1d(abs(e1)**2+abs(e2)**2,dx,pix)/baseline
        ax.plot(y,incoherent*baseline*intensity_scale,"--",color="#777777",label="Intensity sum (not coherent)")
        for phase,color in [(0,"#087e8b"),(np.pi,"#c65b22")]:
            intensity=m.pixel_average_1d(abs(e1+np.exp(1j*phase)*e2)**2,dx,pix)/baseline
            ax.plot(y,intensity*baseline*intensity_scale,color=color,label=f"Field sum, phase={phase/np.pi:g} pi")
            pairs.append(dict(separation_um=d,relative_phase_pi=phase/np.pi,pixel_integrated_right_peak_relative_to_isolated=float(np.interp(d/2,y,intensity))))
        ax.set(xlim=(-d/2-2,d/2+2),ylim=(0,1.2),xlabel="Along-line distance (um)",ylabel="I / I_ref (dimensionless)",title=f"Two 0.5 um defects, separation {d:g} um")
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    savefig(fig,"coherent_pair_examples.png");save_csv(OUT/"coherent_pair_metrics.csv",pairs)


def imaging_figures(cfg,opt):
    from ch3_display import scene_figures,rod_map
    scene_figures(cfg,opt)
    rod_map(cfg,opt)
    # A single scan instant makes the Fourier-plane stage physically explicit.
    n=2048;window=.512;dx=window/n;x=(np.arange(n)-n//2)*dx
    ref,defs=m.full_objects(cfg,x.astype(np.float32),"dots")
    obj=ref+sum(defs)
    nu=fft.fftshift(fft.fftfreq(n,dx));s=m.scales(cfg);ff=nu*s["alpha_mm2"]
    pupil=nu[:,None]**2+nu[None,:]**2<=s["cutoff_per_mm"]**2
    t=m.mean_transmission(ff,cfg,opt["choices"]["selected"])
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout="constrained")
    spectra=[]
    for scanpos in [0,.009]:
        illuminated=obj*np.exp(-2*np.log(2)*(x[None,:]-scanpos)**2/cfg["camera"]["illumination_intensity_fwhm_mm"]**2)
        spectra.append(m.fftc(illuminated)*pupil)
    peak=max(float(np.max(abs(f)**2)) for f in spectra)
    for row,scanpos in enumerate([0,.009]):
        f=spectra[row]
        for ax,data,title in zip(axs[row],[abs(f)**2/peak,np.broadcast_to(t[:,None]**2,(n,n))*pupil,abs(f*t[:,None])**2/peak],
                                 [f"Incident spectrum: s={scanpos*1000:g} um","Effective k=0 intensity transmission","After filter: common spectral reference"]):
            im=ax.imshow(np.maximum(data,1e-8),origin="lower",extent=[ff[0],ff[-1],ff[0],ff[-1]],cmap="magma",norm=LogNorm(1e-6,1))
            ax.set(xlim=(-18,18),ylim=(-3,3),xlabel="X_FF, scan conjugate (mm)",ylabel="Y_FF, line conjugate (mm)",title=title,aspect="auto")
            fig.colorbar(im,ax=ax,shrink=.8,label="|T0|^2" if ax is axs[row,1] else "I_FF / common I_FF,ref")
            for number,k in enumerate(m.active_orders(cfg,opt["choices"]["selected"]),1):
                pos=k*s["order_pitch_mm"]
                if abs(pos)<3:
                    ax.axhline(pos,color="#50b7dc",lw=.5,alpha=.65)
                    ax.text(12,pos+.04,f"R{number}",color="#50b7dc",fontsize=6)
    fig.suptitle("Two scan instants: through periodic dots (top), through a defect in a gap (bottom)")
    savefig(fig,"fourier_stage.png")
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    d=opt["choices"]["selected"];p=cfg["filter"]["tooth_pitch_mm"]
    xx=np.linspace(-1.5,1.5,2000);ph=abs(np.remainder(xx+p/2,p)-p/2)/(p/2)
    tip=cfg["filter"]["tooth_tip_minimum_mm"]/p;ph=np.clip(ph,tip,1-tip)
    edge=d["core_width_mm"]/2+d["b_mm"]*2/np.pi*np.arcsin(ph)
    for j in [-1,0,1]:axs[0].fill_between(xx,j*s["order_pitch_mm"]-edge,j*s["order_pitch_mm"]+edge,color="#333333")
    axs[0].set(xlabel="X_FF along rods (mm)",ylabel="Y_FF across rods (mm)",title="Binary opaque teeth: 0.75 mm pitch",ylim=(-1,1))
    yy=np.linspace(-.25,.25,2001);tt=m.mean_transmission(yy,cfg,d)
    axs[1].plot(yy,tt,label="Zero-order amplitude T0");axs[1].plot(yy,tt**2,label="Zero-order intensity |T0|^2")
    axs[1].set(xlabel="Y_FF around one rod (mm)",ylabel="Transmission",title="20 um flat tips + OD4 assumed")
    axs[1].legend();axs[1].grid(alpha=.2)
    savefig(fig,"manufacturing_and_zero_order.png")


def table(headers,rows):
    return "\n".join(["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,row))+"|" for row in rows])


def report(cfg,opt,validation):
    from pupil36_report import build
    build(cfg,opt,validation)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    cfg=m.config();opt=read("optimization.json")
    target=opt["effective_background_target_db"]
    NAMES["selected"]=f"Joint / >={target:g} dB"
    NAMES["best_hard"]=f"Hard / >={target:g} dB"
    val=validations(cfg,opt)
    optimization_figures(opt);profile_figures(cfg,opt);imaging_figures(cfg,opt)
    report(cfg,opt,val)
    print("Generated Ch3 figures, sensitivity/validation tables and MD report.")


if __name__=="__main__":main()
