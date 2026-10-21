"""Ideal speed-matched TDI: move charge one row per one-row image motion.

This isolates electronic accumulation from optics/acceptance. Input is an
already formed intensity image, with constant shape along its trajectory.
No N-row spatial averaging, N-times pixel pitch, or N-dependent blur kernel.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tdi_compare import gate,config as tdi_config,OUT as OPTICAL_OUT
from ideal_na06 import save_json,save_csv

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"tdi_charge_pupil36_max10"


def charge_transfer_image(image,stages,row_weights=None):
    """Return raw accumulated charge and gain after complete pipeline fill.

    image[scan_index, along_line_pixel] is the reference output image.
    At clock t, image[t+N-1-r] illuminates row r. At the next clock that
    same sample moves to row r+1; its charge must move there too.
    Time t starts at -(N-1), so every output sample receives all N stages.
    Off-frame illumination is zero (no periodic roll/wraparound).
    """
    image=np.asarray(image,dtype=np.float64)
    if image.ndim!=2 or stages<1 or int(stages)!=stages:
        raise ValueError("Expected 2-D intensity and a positive integer stage count.")
    weights=np.ones(stages) if row_weights is None else np.asarray(row_weights,dtype=float)
    if weights.shape!=(stages,) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError("Nonnegative row weights with a positive sum are required.")
    charge=np.zeros((stages,image.shape[1]))
    result=np.zeros_like(image)
    row=np.arange(stages)
    for clock in range(-(stages-1),len(image)):
        charge[1:]=charge[:-1].copy()
        charge[0]=0
        index=clock+stages-1-row
        valid=(index>=0)&(index<len(image))
        charge[valid]+=weights[valid,None]*image[index[valid]]
        if clock>=0:result[clock]=charge[-1]
    return result,float(weights.sum())


def fixed_illumination_weights(stages,pitch,width):
    """Mean intensity in each row, for a fixed Gaussian image illumination.

    Scalar row-exposure weights only: this control does NOT replace the
    separate coherent model in which illumination changes the optical field.
    """
    centers=(np.arange(stages)-(stages-1)/2)*pitch
    norm=width*np.sqrt(np.pi)/(2*np.sqrt(np.log(2)))
    return gate(centers,pitch,width)*norm/pitch


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    # Chart contract: equal-scale spatial comparisons and profile overlays.
    # Question: does charge accumulation alone change geometry? Fixed optics,
    # one-row motion per clock, same output pitch, noise-free. Black reference,
    # blue/orange stage curves distinguished by dash styles; static Python/MD.
    dx=.25
    axis=(np.arange(257)-128)*dx
    xx=axis[:,None];yy=axis[None,:]
    wrapx=(xx+4)%8-4;wrapy=(yy+4)%8-4
    scene=.18*np.exp(-4*np.log(2)*(wrapx**2+wrapy**2)/1.7**2)
    scene+=np.exp(-4*np.log(2)*((xx-3.5)**2+(yy-2)**2)/.8**2)
    scene+=.6*np.exp(-4*np.log(2)*((xx+9)**2+(yy-5)**2)/1.2**2)
    # Also test a previously calculated optical point response, same input for N.
    z=np.load(OPTICAL_OUT/"tdi_point_responses.npz")
    x=z["x_um"];y=z["y_um"]
    ix=np.flatnonzero(abs(x)<=16)[::2];iy=np.flatnonzero(abs(y)<=16)[::2]
    point=z["25_all"][np.ix_(iy,ix)].T.copy()
    point/=point.max()
    rows=[];outputs={};tp=tdi_config()["tdi"]
    for case,input_image in [("prescribed_image",scene),("fixed_optical_point_image",point)]:
        for mode in ["equal_exposure_per_stage","fixed_25stage_exposure_envelope"]:
            for n in [1,25,256]:
                weights=None if mode=="equal_exposure_per_stage" else fixed_illumination_weights(n,.00025,tp["illumination_intensity_fwhm_object_mm"])
                q,gain=charge_transfer_image(input_image,n,weights)
                normalized=q/gain
                error=float(np.max(abs(normalized-input_image))/np.max(input_image))
                relative_l2=float(np.linalg.norm(normalized-input_image)/np.linalg.norm(input_image))
                assert error<2e-13
                rows.append(dict(input_case=case,exposure_mode=mode,stages=n,charge_gain=gain,
                                 normalized_max_error=error,normalized_l2_error=relative_l2,
                                 output_pitch_object_um=dx,motion_blur_added_pixels=0.0))
                if mode=="equal_exposure_per_stage":outputs[f"{case}_{n}"]=normalized
    save_csv(OUT/"registration_validation.csv",rows)
    save_json(OUT/"summary.json",dict(rows=rows,maximum_normalized_error=max(r["normalized_max_error"] for r in rows),
        finding="Same optical input + exact one-row/clock motion = same normalized image for 1,25,256 stages.",
        limits="Ideal matched motion, perfect charge transfer, no saturation/noise; optical acceptance evaluated separately."))
    np.savez_compressed(OUT/"registered_images.npz",axis_um=axis,point_scan_axis_um=x[ix],point_line_axis_um=y[iy],**outputs)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,3,figsize=(11,4.1),layout="constrained")
    for ax,n in zip(axs,[1,25,256]):
        im=ax.imshow(outputs[f"prescribed_image_{n}"].T,origin="lower",extent=[axis[0],axis[-1],axis[0],axis[-1]],
                     vmin=0,vmax=1,cmap="gray",aspect="equal")
        ax.set(xlim=(-12,12),ylim=(-12,12),xlabel="x: scan (um)",ylabel="y: camera line (um)",title=f"{n} stage(s), charge / gain")
    fig.suptitle("Matched-motion charge transfer | identical input optics, identical pixel pitch")
    fig.colorbar(im,ax=axs,label="Normalised intensity",shrink=.7)
    fig.savefig(OUT/"matched_tdi_images.png",dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for n,color,style in [(1,"#333333","-"),(25,"#2465a6","--"),(256,"#c56b23",":")]:
        a=outputs[f"fixed_optical_point_image_{n}"]
        profile=a[:,len(iy)//2]
        axs[0].plot(x[ix],profile,color=color,linestyle=style,label=f"{n} stage(s)")
        delta=a-outputs["fixed_optical_point_image_1"]
        axs[1].plot(x[ix],delta[:,len(iy)//2],color=color,linestyle=style,label=f"{n} - 1 stages")
    axs[0].set(xlim=(-1.5,1.5),xlabel="Scan coordinate (um)",ylabel="Charge / gain",title="Fixed optical point image")
    axs[1].set(xlim=(-1.5,1.5),xlabel="Scan coordinate (um)",ylabel="Normalised residual",title="Numerical registration residual")
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/"matched_tdi_profiles.png",dpi=170);plt.close(fig)
    print(f"Matched TDI regression max relative error: {max(r['normalized_max_error'] for r in rows):.3g}")


if __name__=="__main__":run()
