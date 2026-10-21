"""Fixed-unit image displays and explicit finite physical rod placement.

Chart contract: scientific Python figures embedded in the user's existing MD.
Same I_ref, same +/-45 um ROI, equal physical aspect. Linear evidence is primary;
log views diagnose weak sidelobes without changing the unit or reference.
Image grayscale; blue circles/labels mark true defects, orange diamonds mark
predicted optical replicas (not new object points). No per-image peak scaling.
"""
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LogNorm,SymLogNorm
from matplotlib.patches import Circle
import ch3_linescan as m
from ideal_na06 import save_csv,save_json

LIMITS=(-45.,45.)
I_MAX=1.2
BLUE="#2870a6"
ORANGE="#c87928"


def display_extent(axis):
    half=float(axis[1]-axis[0])/2
    return [axis[0]-half,axis[-1]+half,axis[0]-half,axis[-1]+half]


def mark_targets(ax,indices=(0,1,2)):
    for k in indices:
        x,y,d=m.DEFECTS[k];x*=1000;y*=1000
        ax.add_patch(Circle((x,y),1.7,fill=False,color=BLUE,lw=.8))
        ax.text(x+2.3,y+1.3,f"D{k+1}",color=BLUE,fontsize=8,weight="bold",
                bbox=dict(facecolor="white",alpha=.7,edgecolor="none",pad=.4))


def crop_arrays(z):
    axis=z["axis_um"]
    ids=np.flatnonzero((axis>=LIMITS[0])&(axis<=LIMITS[1]))
    return ids,display_extent(axis[ids])


def scene_figures(cfg,opt):
    rows=[]
    for case in ["dots","stripes"]:
        with np.load(m.OUT/f"{case}_none_fine_scan.npz") as z0, np.load(m.OUT/f"{case}_selected_fine_scan.npz") as z:
            ids,extent=crop_arrays(z)
            cut=np.ix_(ids,ids)
            arrays=[z["object"][cut]**2,z0["actual"][cut],z["actual"][cut],
                    z["reference"][cut],z["defects_only"][cut],z["image_difference"][cut]]
            titles=["Ideal geometric reference |t|^2","Unfiltered camera image","Filtered camera image",
                    "Filtered periodic background only","All 3 defects, background absent","Actual minus background (signed)"]
            for log in [False,True]:
                fig,axs=plt.subplots(2,3,figsize=(13,9),layout="constrained")
                for k,(ax,a,title) in enumerate(zip(axs.flat,arrays,titles)):
                    if k==5:
                        norm=SymLogNorm(1e-5,vmin=-I_MAX,vmax=I_MAX) if log else Normalize(-I_MAX,I_MAX)
                        cmap="RdBu_r";unit="Delta I / I_ref (dimensionless)"
                    else:
                        norm=LogNorm(1e-6,I_MAX) if log else Normalize(0,I_MAX)
                        cmap="gray";unit="I / I_ref (dimensionless)"
                    # LogNorm masks exact zero as transparent by default; clamp
                    # only display input so zero is black, not white background.
                    display=np.maximum(a,1e-6) if log and k!=5 else a
                    im=ax.imshow(display,origin="lower",extent=extent,cmap=cmap,norm=norm,aspect="equal")
                    ax.set(xlim=LIMITS,ylim=LIMITS,title=title,xlabel="x: scan (um)",ylabel="y: camera line (um)")
                    if k!=3:mark_targets(ax)
                    fig.colorbar(im,ax=ax,shrink=.82,label=unit)
                fig.suptitle(f"{case} | optical pupil 36 mm, <=10 rods | {'LOG diagnostic' if log else 'LINEAR comparison'}\n"
                             "One fixed I_ref; all panels show the same field; D1/D2/D3 are real object locations")
                fig.savefig(m.OUT/f"{case}_{'log_diagnostic' if log else 'full_scan'}.png",dpi=170)
                plt.close(fig)

            isolated=z["isolated"]
            before=z0["isolated"]
            axis=z["axis_um"]
            fig,axs=plt.subplots(1,3,figsize=(13,4.7),layout="constrained")
            for k,ax in enumerate(axs):
                cx,cy,d=m.DEFECTS[k];cx*=1000;cy*=1000
                im=ax.imshow(np.maximum(isolated[k][cut],1e-6),origin="lower",extent=extent,cmap="gray",
                             norm=LogNorm(1e-6,I_MAX),aspect="equal")
                ax.set(xlim=LIMITS,ylim=LIMITS,xlabel="x: scan (um)",ylabel="y: camera line (um)",
                       title=f"D{k+1} only: {d*1000:g} um FWHM\ntrue centre ({cx:g}, {cy:g}) um")
                mark_targets(ax,(k,))
                ix=int(np.argmin(abs(axis-cx)));iy=int(np.argmin(abs(axis-cy)))
                centre=float(isolated[k,iy,ix])
                rows.append(dict(case=case,defect=k+1,x_um=cx,y_um=cy,fwhm_um=d*1000,
                                 order=0,response_at_expected_centre=float(isolated[k,iy,ix]),
                                 unfiltered_at_same_centre=float(before[k,iy,ix]),relative_to_own_centre=1.,
                                 in_common_display=bool(LIMITS[0]<=cx<=LIMITS[1] and LIMITS[0]<=cy<=LIMITS[1])))
                for order in [v for v in range(-5,6) if v!=0]:
                    yp=cy+order*cfg["object"]["pitch_mm"]*1000
                    if not LIMITS[0]<=yp<=LIMITS[1]:continue
                    ax.plot(cx,yp,"D",ms=7,mfc="none",mec=ORANGE,mew=.8)
                    ax.text(cx+2,yp,f"{order:+d}p",color=ORANGE,fontsize=7)
                    jy=int(np.argmin(abs(axis-yp)))
                    rows.append(dict(case=case,defect=k+1,x_um=cx,y_um=yp,fwhm_um=d*1000,
                                     order=order,response_at_expected_centre=float(isolated[k,jy,ix]),
                                     unfiltered_at_same_centre=float(before[k,jy,ix]),
                                     relative_to_own_centre=float(isolated[k,jy,ix])/max(centre,1e-30),
                                     in_common_display=True))
            fig.colorbar(im,ax=axs,label="I / I_ref (dimensionless; same scale for every defect)",shrink=.8)
            fig.suptitle(f"{case}: single-defect optical response | blue=true defect, orange=predicted +/-20 um replicas\n"
                         "No TDI and no background in these three calculations; replicas come from the optical filter")
            fig.savefig(m.OUT/f"{case}_defect_tracking.png",dpi=180);plt.close(fig)
    save_csv(m.OUT/"defect_tracking.csv",rows)
    save_json(m.OUT/"display_validation.json",dict(
        intensity_unit="I/I_ref (dimensionless)",linear_limits=[0,I_MAX],
        signed_difference_limits=[-I_MAX,I_MAX],log_limits=[1e-6,I_MAX],
        common_xy_limits_um=list(LIMITS),all_true_defects_in_frame=all(r["in_common_display"] for r in rows if r["order"]==0),
        true_defect_count=len(m.DEFECTS),reference_definition="same illumination, uniform unit-amplitude object, no rods",
        reference_value=m.camera_intensity_reference(cfg),
        columns=["ideal intensity","unfiltered","filtered","background","all defects without background","signed difference"]))


def rod_map(cfg,opt):
    d=opt["choices"]["selected"];s=m.scales(cfg)
    orders=m.active_orders(cfg,d);pitch=s["order_pitch_mm"];radius=s["filter_pupil_mm"]/2
    a=d["core_width_mm"]/2;b=d["b_mm"];tooth=cfg["filter"]["tooth_pitch_mm"]
    positions=[k*pitch for k in orders]
    zoom=max(abs(v) for v in positions)+max(.8,a+b)
    fig,axs=plt.subplots(1,2,figsize=(13,7.5),layout="constrained")
    x=np.linspace(-25,25,12001)
    phase=abs(np.remainder(x+tooth/2,tooth)-tooth/2)/(tooth/2)
    tip=cfg["filter"]["tooth_tip_minimum_mm"]/tooth
    phase=np.clip(phase,tip,1-tip)
    edge=a+b*(phase if d["profile"]=="linear" else 2/np.pi*np.arcsin(phase))
    table=[]
    for ax in axs:
        ax.add_patch(Circle((0,0),radius,fill=False,color=BLUE,lw=1.5))
        if ax is axs[0]:
            ax.add_patch(Circle((0,0),cfg["optics"]["filter_clear_diameter_mm"]/2,
                                fill=False,color="#888888",ls="--",lw=1))
        for number,(k,centre) in enumerate(zip(orders,positions),1):
            ax.fill_between(x,centre-edge,centre+edge,color="#333333",lw=0)
            if ax is axs[1]:
                ax.axhline(centre,color=ORANGE,ls="--",lw=.65)
                ax.text(zoom*.2,centre+.04,f"R{number}: n={k:+d}, Y={centre:+.4f} mm",
                        fontsize=8,color=BLUE,bbox=dict(facecolor="white",alpha=.85,pad=.6,edgecolor="none"))
        ax.set(aspect="equal",xlabel="X_FF along rods (mm)",ylabel="Y_FF across rods (mm)")
    axs[0].set(xlim=(-26,26),ylim=(-26,26),title="Full layout: blue=36 mm optical pupil\nGrey dashed=50 mm mechanical opening")
    axs[1].set(xlim=(-zoom,zoom),ylim=(-zoom,zoom),title="Central zoom: every physical rod labelled\nn=0 is the object zero-order blocker")
    for number,(k,centre) in enumerate(zip(orders,positions),1):
        table.append(dict(rod_id=f"R{number}",object_order_n=k,centre_y_mm=centre,
                          core_y_min_mm=centre-a,core_y_max_mm=centre+a,
                          envelope_y_min_mm=centre-a-b,envelope_y_max_mm=centre+a+b,
                          full_core_mm=2*a,single_side_b_mm=b,tooth_pitch_mm=tooth,
                          active_pupil_chord_at_centre_mm=2*np.sqrt(max(0,radius**2-centre**2))))
    fig.suptitle(f"Selected finite Fourier-rod layout: {len(orders)} rods (hard limit {cfg['filter']['maximum_rods']})\n"
                 f"f_FF={s['f_ff_mm']:g} mm; Y_n=n x {pitch:.4f} mm; core={2*a:g} mm, b={b:g} mm")
    fig.savefig(m.OUT/"fourier_rod_positions.png",dpi=190);plt.close(fig)
    save_csv(m.OUT/"fourier_rod_positions.csv",table)
