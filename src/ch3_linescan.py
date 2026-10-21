"""Normal incidence / central Ch3 collection, coherent line-scan Fourier filter.

The tooth diffraction order k is distinct from object lattice order m.
Only tooth k=0 is detected. Convolution kernels incorporate scanned line
illumination and finite detector-line height; no area-camera substitution.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import fft, special
import yaml

from ideal_na06 import fftc, ifftc, save_csv, save_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"ch3_pupil36_max10"
DEFECTS=((.009,.009,.0005),(-.009,.029,.001),(.031,-.031,.002))


def design_parameters(design):
    return {k:design[k] for k in ("core_width_mm","b_mm","profile","threshold","orders") if k in design}


def active_orders(cfg,design):
    if design.get("profile")=="none":
        return []
    if "orders" not in design:
        raise ValueError("Explicit physical rod orders required; no implicit infinite rod comb.")
    orders=sorted(set(int(k) for k in design["orders"]))
    if len(orders)>cfg["filter"]["maximum_rods"]:
        raise ValueError("Physical rod count exceeds maximum_rods.")
    return orders


def config():
    return yaml.safe_load((ROOT/"config"/"ch3_linescan.yml").read_text(encoding="utf-8"))


def scales(cfg):
    o=cfg["optics"]
    # A mechanical clear aperture is NOT a pupil magnifier.
    relay=o.get("pupil_relay_magnification",1.0)
    fff=o["objective_focal_length_mm"]*relay
    pupil_at_filter=2*fff*o["numerical_aperture"]
    if pupil_at_filter>o["filter_clear_diameter_mm"]+1e-9:
        raise ValueError("Mechanical aperture clips optical pupil; explicit clipped-pupil model required.")
    alpha=fff*o["wavelength_mm"]
    return dict(pupil_mm=2*o["objective_focal_length_mm"]*o["numerical_aperture"],
                relay_magnification=relay,f_ff_mm=fff,filter_pupil_mm=pupil_at_filter,
                mechanical_clear_diameter_mm=o["filter_clear_diameter_mm"],
                alpha_mm2=alpha,order_pitch_mm=alpha/cfg["object"]["pitch_mm"],
                cutoff_per_mm=o["numerical_aperture"]/o["wavelength_mm"],
                pixel_object_mm=cfg["camera"]["pixel_width_mm"]/o["magnification"],
                line_height_object_mm=cfg["camera"]["active_line_height_mm"]/o["magnification"],
                tooth_first_order_shift_um=1000*alpha/cfg["filter"]["tooth_pitch_mm"])


def mean_profile(distance,a,b,profile):
    if b==0:
        return (np.asarray(distance)>a).astype(float)
    q=np.clip((np.asarray(distance)-a)/b,0,1)
    return q if profile=="linear" else np.sin(np.pi*q/2)


def mean_transmission(ff,cfg,design,shift=0.0,width_error=0.0,leak=None,blunted=True):
    if design.get("profile")=="none":
        return np.ones_like(ff)
    pitch=scales(cfg)["order_pitch_mm"]
    order=np.rint((ff-shift)/pitch)
    distance=np.abs(ff-shift-order*pitch)
    active=np.isin(order,active_orders(cfg,design))
    a=(design["core_width_mm"]+width_error)/2
    b=design["b_mm"]
    t=mean_profile(distance,a,b,design["profile"])
    if b and blunted:
        # Explicit manufacturable flattened tips/valleys: transparent/opaque
        # slivers smaller than the minimum along-tooth feature are removed.
        duty=cfg["filter"]["tooth_tip_minimum_mm"]/cfg["filter"]["tooth_pitch_mm"]
        t=np.where(t<duty,0,np.where(t>1-duty,1,t))
    t=np.where(active,t,1)
    if leak is None:
        leak=cfg["filter"]["blocked_intensity_transmission"]
    return np.sqrt(leak)+(1-np.sqrt(leak))*t


def pixel_average_1d(i,dx,pixel):
    steps=int(round(pixel/dx))
    if steps<2:
        return i
    if steps%2:
        raise ValueError("Use even subdivisions across a centered camera pixel.")
    kernel=np.ones(steps+1)
    kernel[[0,-1]]=.5
    return np.convolve(i,kernel/kernel.sum(),mode="same")


def camera_intensity_reference(cfg,quadrature_order=12):
    """One fixed image-intensity reference, never a per-scene/defect peak.

    Uniform unit-amplitude object, same Gaussian illumination and line height,
    no Fourier rods. Analytic angular-spectrum integral over the circular pupil.
    Common magnification/irradiance factors cancel in I/I_ref.
    """
    nodes,weights=np.polynomial.legendre.leggauss(quadrature_order)
    sc=scales(cfg);width=cfg["camera"]["illumination_intensity_fwhm_mm"]
    u=nodes*sc["line_height_object_mm"]/2
    beta=np.pi*width/np.sqrt(2*np.log(2));phase=np.pi*u/beta
    field=np.exp(-phase**2)*np.real(special.erf(beta*sc["cutoff_per_mm"]+1j*phase))
    return float(np.sum(weights/2*field**2))


def measures(axis,e,pixel_mm,guard_mm,radial_order):
    i=pixel_average_1d(np.abs(e)**2,axis[1]-axis[0],pixel_mm)
    c=len(axis)//2
    normalized=i/max(i[c],1e-40)
    crosses=np.flatnonzero(normalized[c:]<=.5)
    k=c+int(crosses[0]) if len(crosses) else len(axis)-1
    half=np.interp(.5,[normalized[k],normalized[k-1]],[axis[k],axis[k-1]])
    cumul=np.cumsum(i[radial_order])
    total=cumul[-1]
    widths=[]
    for frac in (.8,.9):
        j=min(int(np.searchsorted(cumul,frac*total)),len(axis)-1)
        widths.append(2*abs(axis[radial_order[j]])*1000)
    outside=np.abs(axis)>guard_mm
    return dict(fwhm_um=float(2*half*1000),ee80_width_um=float(widths[0]),ee90_width_um=float(widths[1]),
                energy=float(total),halo_fraction=float(i[outside].sum()/max(total,1e-40)),
                halo_peak_fraction=float(i[outside].max()/max(i[c],1e-40)))


def search_arrays(cfg,n=None,window=None):
    n=n or cfg["numerics"]["search_n"]
    window=window or cfg["numerics"]["search_window_mm"]
    dx=window/n
    axis=(np.arange(n)-n//2)*dx
    nu=fft.fftshift(fft.fftfreq(n,d=dx))
    sc=scales(cfg)
    fxmax=np.sqrt(np.maximum(0,sc["cutoff_per_mm"]**2-nu**2))
    # Background is illuminated by the scanned Gaussian line, uniform locally
    # in scan direction. This is the center-line coherent pupil integral.
    bg_transfer=special.erf(np.pi*cfg["camera"]["illumination_intensity_fwhm_mm"]*fxmax/np.sqrt(2*np.log(2)))
    extent=cfg["object"]["field_mm"]/2
    taper=cfg["object"]["line_edge_taper_mm"]
    envelope=.5+.5*np.cos(np.pi*np.clip((np.abs(axis)-(extent-taper))/taper,0,1))
    pitch=cfg["object"]["pitch_mm"]
    wrap=np.remainder(axis+pitch/2,pitch)-pitch/2
    backgrounds=[]
    for w in [cfg["object"]["feature_width_mm"], .0049]:
        obj=(np.abs(wrap)<w/2)*envelope
        backgrounds.append(np.abs(fft.fftshift(fft.fft(fft.ifftshift(obj),norm="ortho"))*bg_transfer)**2)
    radial_order=np.argsort(np.abs(axis))
    targets=[]
    baselines=[]
    for d_um in cfg["object"]["defect_fwhm_um"]:
        d=d_um/1000
        wx=1/np.sqrt(1/d**2+1/cfg["camera"]["illumination_intensity_fwhm_mm"]**2)
        spectrum=np.exp(-np.pi**2*d*d*nu**2/(2*np.log(2)))*special.erf(np.pi*wx*fxmax/np.sqrt(2*np.log(2)))
        targets.append(spectrum)
        e=fft.fftshift(fft.ifft(fft.ifftshift(spectrum),norm="ortho"))
        baselines.append(measures(axis,e,sc["pixel_object_mm"],max(.00075,1.5*d),radial_order))
    return dict(axis=axis,nu=nu,ff=nu*sc["alpha_mm2"],backgrounds=backgrounds,targets=targets,
                baselines=baselines,radial_order=radial_order,dx=dx,window=window,n=n)


def evaluate_candidate(cfg,arr,design,details=False,transmission_function=None):
    sc=scales(cfg)
    transmission_function=transmission_function or mean_transmission
    t=transmission_function(arr["ff"],cfg,design)
    all_m=[]
    target_energy=[]
    fields=[]
    for j,(d,spec,base) in enumerate(zip(cfg["object"]["defect_fwhm_um"],arr["targets"],arr["baselines"])):
        e=fft.fftshift(fft.ifft(fft.ifftshift(spec*t),norm="ortho"))
        m=measures(arr["axis"],e,sc["pixel_object_mm"],max(.00075,1.5*d/1000),arr["radial_order"])
        m.update(defect_fwhm_um=d,retention=m["energy"]/base["energy"],fwhm_ratio=m["fwhm_um"]/base["fwhm_um"],
                 ee90_ratio=m["ee90_width_um"]/base["ee90_width_um"])
        all_m.append(m)
        target_energy.append(np.abs(spec)**2)
        if j==0 or details:
            fields.append(e)
    # Dense distance range includes Fourier-filter replicas, not just a few
    # favorable zero locations. Worst relative phase gives 2r+r² at a neighbor.
    sep=np.linspace(.001,.025,481)
    e0=fields[0]
    ratios=np.interp(sep,arr["axis"],np.abs(e0))/max(abs(e0[len(e0)//2]),1e-40)
    crosstalk=2*ratios+ratios**2
    worst_r=1.0
    worst_b=0.0
    tol=cfg["filter"]["lateral_tolerance_mm"]
    dw_tol=cfg["filter"]["core_full_width_tolerance_mm"]
    for shift,dw in [(0,0),(tol,-dw_tol),(-tol,-dw_tol),(0,dw_tol)]:
        tt=transmission_function(arr["ff"],cfg,design,shift,dw)
        worst_b=max(worst_b,max(float(np.sum(b*tt*tt)/np.sum(b)) for b in arr["backgrounds"]))
        worst_r=min(worst_r,min(float(np.sum(d*tt*tt)/np.sum(d)) for d in target_energy))
    s=-10*np.log10(max(worst_b,1e-30))
    r_nom=min(m["retention"] for m in all_m)
    fwhm=max(m["fwhm_ratio"] for m in all_m)
    halo=max(m["halo_fraction"] for m in all_m)
    ee=max(m["ee90_ratio"] for m in all_m)
    cross=float(crosstalk.max())
    o=cfg["optimization"]
    score=(s+10*np.log10(max(worst_r,1e-30))
           -10*np.log10(1+halo/o["halo_reference_fraction"])
           -10*np.log10(1+cross/o["crosstalk_reference_fraction"])
           -10*np.log10(1+max(ee-1,0)))
    feasible=worst_r>=o["minimum_retention"] and fwhm<=o["maximum_fwhm_ratio"]
    result=dict(**design,score_db=float(score),worst_background_suppression_db=float(s),
                nominal_min_retention=float(r_nom),worst_retention=float(worst_r),maximum_fwhm_ratio=float(fwhm),
                maximum_ee90_ratio=float(ee),maximum_halo_fraction=float(halo),
                worst_pair_intensity_perturbation=float(cross),worst_pair_separation_um=float(sep[crosstalk.argmax()]*1000),
                feasible=bool(feasible))
    if details:
        result["defect_metrics"]=all_m
        result["baseline_defect_metrics"]=arr["baselines"]
        result["pair_metrics"]=[dict(separation_um=float(d),field_ratio=float(r),worst_phase_intensity_perturbation=float(2*r+r*r))
                                  for d,r in zip(cfg["object"]["pair_separations_um"],np.interp(np.array(cfg["object"]["pair_separations_um"])/1000,arr["axis"],np.abs(e0))/abs(e0[len(e0)//2]))]
        view=np.abs(arr["axis"])<.04
        np.savez_compressed(OUT/(design.get("label","selected")+"_line_profiles.npz"),
                            y_um=arr["axis"][view]*1000,fields=np.asarray(fields)[:,view],
                            baseline_fields=np.asarray([fft.fftshift(fft.ifft(fft.ifftshift(f),norm="ortho"))[view] for f in arr["targets"]]),
                            ff=arr["ff"],transmission=t,sep_um=sep*1000,crosstalk=crosstalk)
    return result


def optimize(cfg):
    OUT.mkdir(parents=True,exist_ok=True)
    arr=search_arrays(cfg)
    rows=[]
    started=time.time()
    pool=cfg["optimization"]["order_pair_pool"]
    max_pairs=(cfg["filter"]["maximum_rods"]-1)//2
    layouts=[]
    for count in range(min(max_pairs,len(pool))+1):
        for pairs in itertools.combinations(pool,count):
            layouts.append(sorted([0]+list(pairs)+[-k for k in pairs]))
    for core in cfg["optimization"]["core_widths_mm"]:
        for b in cfg["optimization"]["transition_widths_mm"]:
            if core+2*b+.01>scales(cfg)["order_pitch_mm"]-cfg["filter"]["minimum_clear_gap_mm"]:
                continue
            for profile in (["hard"] if b==0 else cfg["optimization"]["profiles"]):
                for orders in layouts:
                    d=dict(core_width_mm=core,b_mm=b,profile=profile,threshold=0,
                           orders=orders,rod_count=len(orders))
                    rows.append(evaluate_candidate(cfg,arr,d))
        print(f"Ch3: core {core:.2f} mm, {len(rows)} candidates ({time.time()-started:.1f}s)",flush=True)
    feasible=[r for r in rows if r["feasible"]]
    if not feasible:
        save_csv(OUT/"candidate_scan.csv",rows)
        raise RuntimeError("No jointly feasible design; retain evidence and revise explicit bounds.")
    ring_priority=max(feasible,key=lambda r:r["score_db"])
    target=cfg["optimization"]["selected_background_target_db"]
    eligible=[r for r in feasible if r["worst_background_suppression_db"]>=target]
    target_met=bool(eligible)
    effective_target=target
    if not eligible:
        lower_targets=sorted([v for v in cfg["optimization"]["background_targets_db"] if v<target],reverse=True)
        for fallback in lower_targets:
            eligible=[r for r in feasible if r["worst_background_suppression_db"]>=fallback]
            if eligible:
                effective_target=fallback
                break
    selected=max(eligible,key=lambda r:r["score_db"]) if eligible else max(feasible,key=lambda r:(r["worst_background_suppression_db"],r["score_db"]))
    background_only=max(feasible,key=lambda r:r["worst_background_suppression_db"])
    hard_pool=[r for r in (eligible or feasible) if r["profile"]=="hard"]
    hard=max(hard_pool,key=lambda r:r["score_db"])
    choices={}
    selection=[("selected",selected),("background_only",background_only),("best_hard",hard),("ring_priority",ring_priority),
               ("none",dict(core_width_mm=0,b_mm=0,profile="none",threshold=0,orders=[]))]
    for target in cfg["optimization"]["background_targets_db"]:
        matches=[r for r in feasible if r["worst_background_suppression_db"]>=target]
        if matches:
            selection.append((f"target_{target:g}db",max(matches,key=lambda r:r["score_db"])))
    for label,r in selection:
        design=design_parameters(r)
        design["rod_count"]=len(active_orders(cfg,design))
        design["label"]=label
        choices[label]=evaluate_candidate(cfg,arr,design,details=True)
    save_csv(OUT/"candidate_scan.csv",rows)
    save_json(OUT/"optimization.json",dict(config=cfg,derived=scales(cfg),candidate_count=len(rows),
              feasible_count=len(feasible),choices=choices,elapsed_seconds=time.time()-started,
              maximum_rods=cfg["filter"]["maximum_rods"],layout_count=len(layouts),
              background_target_met=target_met,
              effective_background_target_db=effective_target if eligible else None,
              selection_rule="highest joint score above attainable declared target" if eligible else "target infeasible: highest robust background suppression within R/FWHM constraints",
              strict_ring_feasible_at_15db=sum(r["worst_background_suppression_db"]>=15 and r["maximum_halo_fraction"]<=.1 and r["worst_pair_intensity_perturbation"]<=.1 for r in feasible),
              caveat="1-D center-line prescreen; validate with full scanned-line kernel and finite sensor height."))
    print(json.dumps({k:{j:v for j,v in r.items() if not isinstance(v,list)} for k,r in choices.items()},ensure_ascii=False,indent=2),flush=True)
    return choices


def full_transfer(cfg,n,window,design,u):
    dx=window/n
    x=((np.arange(n)-n//2)*dx).astype(np.float32)
    nu=fft.fftshift(fft.fftfreq(n,d=dx)).astype(np.float32)
    sc=scales(cfg)
    pupil=(nu[:,None]**2+nu[None,:]**2<=sc["cutoff_per_mm"]**2).astype(np.float32)
    # rows=y (camera line); cols=x (scan). Shift h(x) by detector offset u,
    # multiply by scanned illumination A(x), transform back in x only.
    phase=np.exp(2j*np.pi*nu*u).astype(np.complex64)
    partial=fft.fftshift(fft.ifft(fft.ifftshift(pupil*phase[None,:],axes=1),axis=1,norm="ortho",workers=4),axes=1)
    partial*=np.exp(-2*np.log(2)*x*x/cfg["camera"]["illumination_intensity_fwhm_mm"]**2)[None,:]
    transfer=fft.fftshift(fft.fft(fft.ifftshift(partial,axes=1),axis=1,norm="ortho",workers=4),axes=1)
    # Preserve physical unit-input field scale here. The SAME uniform-object
    # camera intensity reference is applied after intensity/pixel integration.
    baseline_scale=1.0
    t=mean_transmission(nu*sc["alpha_mm2"],cfg,design).astype(np.float32)
    return transfer*t[:,None]/baseline_scale,x


def full_objects(cfg,x,case):
    xx=x[None,:]
    yy=x[:,None]
    p=cfg["object"]["pitch_mm"]
    wrapx=np.remainder(xx+p/2,p)-p/2
    wrapy=np.remainder(yy+p/2,p)-p/2
    radius=cfg["object"]["feature_width_mm"]/2
    if case=="dots":
        reference=(wrapx*wrapx+wrapy*wrapy<radius**2).astype(np.float32)
    elif case=="stripes":
        reference=np.broadcast_to(np.abs(wrapy)<radius,(len(x),len(x))).astype(np.float32).copy()
    else:
        raise ValueError(case)
    edge=cfg["object"]["field_mm"]/2
    taper=cfg["object"]["line_edge_taper_mm"]
    envelope=.5+.5*np.cos(np.pi*np.clip((np.abs(x)-(edge-taper))/taper,0,1))
    reference*=envelope[:,None]*envelope[None,:]
    # Deliberately non-lattice positions in gaps, Gaussian amplitude objects;
    # specified defect sizes mean their *object intensity* FWHM.
    defects=[]
    for cx,cy,d in DEFECTS:
        defects.append(np.exp(-2*np.log(2)*((xx-cx)**2+(yy-cy)**2)/d**2).astype(np.float32))
    return reference,defects


def full_simulation(cfg,design,case="dots",n=None,window=None,tag="selected",order=None):
    n=n or cfg["numerics"]["full_n"]
    window=window or cfg["numerics"]["full_window_mm"]
    dx=window/n
    if dx>cfg["optics"]["wavelength_mm"]/(2*cfg["optics"]["numerical_aperture"]):
        raise ValueError("NA undersampled")
    if not np.isclose(dx,scales(cfg)["pixel_object_mm"]) and not np.isclose(dx*2,scales(cfg)["pixel_object_mm"]):
        raise ValueError("Final camera binning expects 1x or 2x object sampling.")
    order=order or cfg["numerics"]["camera_quadrature_order"]
    nodes,weights=np.polynomial.legendre.leggauss(order)
    weights=weights/2
    offsets=nodes*scales(cfg)["line_height_object_mm"]/2
    x=((np.arange(n)-n//2)*dx).astype(np.float32)
    ref,defects=full_objects(cfg,x,case)
    fref=fftc(ref)
    fdefs=[fftc(d) for d in defects]
    iref=np.zeros_like(ref)
    actual=np.zeros_like(ref)
    defects_only=np.zeros_like(ref)
    isolated=[np.zeros_like(ref) for _ in defects]
    for j,(u,w) in enumerate(zip(offsets,weights)):
        g,_=full_transfer(cfg,n,window,design,float(u))
        e=ifftc(fref*g)
        iref+=w*np.abs(e)**2
        total=e.copy()
        for k,fd in enumerate(fdefs):
            ed=ifftc(fd*g)
            isolated[k]+=w*np.abs(ed)**2
            total+=ed
        actual+=w*np.abs(total)**2
        defects_only+=w*np.abs(total-e)**2
        del g,e,total,ed
        print(f"Ch3 {case}/{tag}: sensor-height sample {j+1}/{order}",flush=True)
    # Finite integration along the line (y). Scan direction is sampled, not
    # averaged as a full area-camera pixel; line height was already integrated.
    pixel=scales(cfg)["pixel_object_mm"]
    step=int(round(pixel/dx))
    for arr in [iref,actual,defects_only]+isolated:
        # Centered trapezoid quadrature within one y pixel; on coarse grid use
        # half-pixel linear interpolation (equivalent kernel .125,.75,.125).
        if step==1:
            arr[:]=.125*np.roll(arr,1,axis=0)+.75*arr+.125*np.roll(arr,-1,axis=0)
        else:
            arr[:]=.25*np.roll(arr,1,axis=0)+.5*arr+.25*np.roll(arr,-1,axis=0)
        # Use the SAME physical reference for every scene, rod layout and defect.
        arr/=camera_intensity_reference(cfg)
    half=cfg["object"]["field_mm"]/2
    ids=np.flatnonzero(np.abs(x)<=half)
    crop=(slice(ids[0],ids[-1]+1,step),slice(ids[0],ids[-1]+1,step))
    metrics=dict(case=case,tag=tag,n=n,window_mm=window,dx_um=dx*1000,sensor_quadrature_order=order,
                 intensity_unit="I/I_ref (dimensionless)",
                 intensity_reference=camera_intensity_reference(cfg),
                 reference_definition="uniform unit-amplitude object, no rods, same illumination and detector line height",
                 design=design_parameters(design),derived=scales(cfg),
                 reference_energy=float(iref.sum(dtype=np.float64)*dx*dx),
                 actual_energy=float(actual.sum(dtype=np.float64)*dx*dx),
                 defect_energies=[float(i.sum(dtype=np.float64)*dx*dx) for i in isolated])
    # Camera y long axis, scan x slow axis: stitched array is y rows, x cols.
    np.savez_compressed(OUT/f"{case}_{tag}_scan.npz",object=(ref+sum(defects))[crop],
                        reference=iref[crop],actual=actual[crop],isolated=np.asarray([i[crop] for i in isolated]),
                        defects_only=defects_only[crop],
                        image_difference=(actual-iref)[crop],axis_um=x[crop[0]]*1000)
    save_json(OUT/f"{case}_{tag}_full_metrics.json",metrics)
    return metrics


def export_rods(cfg,design):
    pitch=scales(cfg)["order_pitch_mm"]
    radius=scales(cfg)["filter_pupil_mm"]/2
    a=design["core_width_mm"]/2
    b=design["b_mm"]
    p=cfg["filter"]["tooth_pitch_mm"]
    rows=[]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="60mm" viewBox="-30 -30 60 60">',
         '<title>Ch3 line-scan, x is scan; black film on UV fused silica</title>','<g id="OPAQUE_PATTERN" fill="black">']
    dxf=['0','SECTION','2','HEADER','9','$INSUNITS','70','4','0','ENDSEC','0','SECTION','2','ENTITIES']
    # Invert t(u) to construct each edge as a repeating binary duty waveform.
    xs=np.linspace(-27,27,10801)
    phase=np.abs(np.remainder(xs+p/2,p)-p/2)/(p/2)
    tip=cfg["filter"]["tooth_tip_minimum_mm"]/p
    phase=np.clip(phase,tip,1-tip)
    ramp=phase if design["profile"]=="linear" else 2/np.pi*np.arcsin(phase)
    edge=a+b*ramp if b else np.full_like(xs,a)
    for m in active_orders(cfg,design):
        center=m*pitch
        if abs(center)>radius+a+b:
            raise ValueError("Selected rod does not intersect the optical pupil.")
        # Clipping the triangular phase gives flat tips/valleys and exactly
        # the duty-threshold mean used above, up to polygon sampling error.
        upper=np.column_stack([xs,center+edge])
        lower=np.column_stack([xs[::-1],center-edge[::-1]])
        points=np.vstack([upper,lower])
        svg.append('<polygon points="'+' '.join(f'{x:.5f},{-y:.5f}' for x,y in points)+'"/>')
        dxf+=['0','LWPOLYLINE','8','OPAQUE_BOUNDARY','90',str(len(points)),'70','1']
        for x,y in points:
            dxf+=['10',f'{x:.6f}','20',f'{y:.6f}']
        rows.append(dict(order=m,center_y_mm=center,core_width_mm=2*a,transition_b_mm=b,
                         outer_width_mm=2*(a+b),tooth_pitch_mm=p,profile=design["profile"]))
    mechanical=cfg["optics"]["filter_clear_diameter_mm"]/2
    svg+=['</g>',f'<circle id="OPTICAL_PUPIL_GUIDE_NOT_CHROME" r="{radius:g}" fill="none" stroke="#185abd" stroke-width="0.05"/>',
          f'<circle id="MECHANICAL_GUIDE_NOT_CHROME" r="{mechanical:g}" fill="none" stroke="#777777" stroke-dasharray="0.5 0.5" stroke-width="0.05"/>','</svg>']
    dxf+=['0','ENDSEC','0','EOF']
    (OUT/"selected_rods_checkplot.svg").write_text('\n'.join(svg),encoding="utf-8")
    (OUT/"selected_rods_checkplot.dxf").write_text('\n'.join(dxf)+'\n',encoding="ascii")
    save_csv(OUT/"selected_rod_dimensions.csv",rows)
    save_json(OUT/"manufacturing_check.json",dict(rod_count=len(rows),nominal_core_width_mm=2*a,transition_b_mm=b,
              outer_width_mm=2*(a+b),clear_gap_mm=pitch-2*(a+b),tooth_pitch_mm=p,
              tip_minimum_mm=cfg["filter"]["tooth_tip_minimum_mm"],substrate_mm=[60,60],
              optical_pupil_diameter_mm=2*radius,mechanical_clear_diameter_mm=2*mechanical,
              warning="Engineering checkplot with flattened tips; polygon sampling, corner rounding, coating OD/phase and substrate require DFM/optical revalidation. Not a released production drawing."))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["optimize","full","validate","export"],default="optimize")
    p.add_argument("--case",choices=["dots","stripes"],default="dots")
    p.add_argument("--choice",default="selected")
    p.add_argument("--n",type=int)
    p.add_argument("--window",type=float)
    p.add_argument("--order",type=int)
    p.add_argument("--tag")
    args=p.parse_args()
    cfg=config()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="optimize":
        optimize(cfg)
        return
    opt=json.loads((OUT/"optimization.json").read_text(encoding="utf-8"))
    design=dict(profile="none",core_width_mm=0,b_mm=0,threshold=0) if args.choice=="none" else opt["choices"][args.choice]
    if args.stage=="full":
        full_simulation(cfg,design,args.case,n=args.n,window=args.window,tag=args.tag or args.choice,order=args.order)
    elif args.stage=="export":
        export_rods(cfg,design)
    else:
        arr=search_arrays(cfg,65536,4.096)
        result=evaluate_candidate(cfg,arr,{**design_parameters(design),"label":"fine_search"},details=True)
        save_json(OUT/"fine_search_validation.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
