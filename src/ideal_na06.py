"""Ideal scalar sine-condition imaging with manufacturable Fourier stops.

Independent from the older scalar / real-Zemax models. All internal lengths mm.
Optimization uses Parseval energy sums, not an image-domain proxy. Finalists are
propagated with complex FFT/IFFT and separately checked in localized image ROIs.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft, ndimage, special
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ideal_na06"


def config():
    return yaml.safe_load((ROOT/"config"/"ideal_na06.yml").read_text(encoding="utf-8"))


def scales(cfg):
    op = cfg["optics"]
    dp = 2*op["objective_focal_length_mm"]*op["numerical_aperture"]
    mp = op["filter_clear_diameter_mm"]/dp
    eff = mp*op["objective_focal_length_mm"]
    return dict(original_pupil_diameter_mm=dp, pupil_relay_magnification=mp,
                effective_fourier_focal_length_mm=eff,
                fourier_scale_mm2=eff*op["wavelength_mm"],
                order_pitch_mm=eff*op["wavelength_mm"]/cfg["object"]["pitch_mm"],
                coherent_cutoff_per_mm=op["numerical_aperture"]/op["wavelength_mm"],
                nyquist_max_dx_um=1000*op["wavelength_mm"]/(2*op["numerical_aperture"]),
                rayleigh_radius_um=.61*op["wavelength_mm"]/op["numerical_aperture"]*1000)


@dataclass
class Grid:
    n: int
    window: float
    x: np.ndarray
    ff: np.ndarray
    pupil: np.ndarray
    cfg: dict

    @property
    def dx(self):
        return self.window/self.n

    @property
    def dff(self):
        return scales(self.cfg)["fourier_scale_mm2"]/self.window


def make_grid(cfg, n=None, window=None):
    n = n or cfg["grid"]["samples"]
    window = window or cfg["grid"]["window_mm"]
    dx = window/n
    if dx*1000 > scales(cfg)["nyquist_max_dx_um"]:
        raise ValueError("Object-plane sampling does not support requested NA.")
    x = ((np.arange(n)-n//2)*dx).astype(np.float32)
    freqs = fft.fftshift(fft.fftfreq(n, d=dx)).astype(np.float32)
    ff = freqs*scales(cfg)["fourier_scale_mm2"]
    radius = cfg["optics"]["filter_clear_diameter_mm"]/2
    pupil = (ff[:,None]**2+ff[None,:]**2 <= radius**2)
    return Grid(n, window, x, ff, pupil, cfg)


def illumination(grid, hard=False):
    r = np.hypot(grid.x[:,None], grid.x[None,:])
    radius = grid.cfg["grid"]["illuminated_diameter_mm"]/2
    taper = 0 if hard else grid.cfg["grid"]["illumination_edge_taper_mm"]
    if taper == 0:
        return (r < radius).astype(np.float32)
    u = np.clip((r-(radius-taper))/taper, 0, 1)
    return (.5+.5*np.cos(np.pi*u)).astype(np.float32)


def disk(grid, cx, cy, radius):
    return (grid.x[None,:]-cx)**2+(grid.x[:,None]-cy)**2 < radius**2


def object_fields(grid, case, hard=False):
    cfg = grid.cfg
    pitch = cfg["object"]["pitch_mm"]
    wrap = np.remainder(grid.x+pitch/2, pitch)-pitch/2
    ill = illumination(grid, hard)
    if case == "stripes":
        half = cfg["object"]["stripe_duty"]*pitch/2
        binary = np.broadcast_to(np.abs(wrap)[None,:] < half, (grid.n, grid.n)).copy()
        missing = (np.abs(grid.x[None,:]) < half) & (np.abs(grid.x[:,None]) < .010)
        extra = (np.abs(grid.x[None,:]-.009) < .0015) & (np.abs(grid.x[:,None]-.033) < .006)
        tiny = disk(grid, -.010, -.043, .0006)
        deltas = [-(missing & binary).astype(np.float32),
                  (extra & ~binary).astype(np.float32), (tiny & ~binary).astype(np.float32)]
        names = ["missing_7x20um", "extra_3x12um", "extra_diameter_1p2um"]
    elif case == "dots":
        r = cfg["object"]["dot_diameter_mm"]/2
        binary = wrap[:,None]**2+wrap[None,:]**2 < r*r
        original = disk(grid, 0, 0, r)
        moved = disk(grid, .005, .004, r)
        modified = binary.copy()
        modified[original] = False
        modified[moved] = True
        extra = disk(grid, .010, .030, .002)
        tiny = disk(grid, -.010, -.030, .0006)
        deltas = [modified.astype(np.float32)-binary.astype(np.float32),
                  (extra & ~binary).astype(np.float32), (tiny & ~binary).astype(np.float32)]
        names = ["shift_5um_4um", "extra_diameter_4um", "extra_diameter_1p2um"]
    else:
        raise ValueError(case)
    reference = binary.astype(np.float32)*ill
    deltas = [d*ill for d in deltas]
    return reference, deltas, names


def fftc(u, workers=4):
    return fft.fftshift(fft.fft2(fft.ifftshift(u), norm="ortho", workers=workers))


def ifftc(f, workers=4):
    return fft.fftshift(fft.ifft2(fft.ifftshift(f), norm="ortho", workers=workers))


def pupil_coordinates(grid):
    x = np.broadcast_to(grid.ff[None,:], grid.pupil.shape)[grid.pupil]
    y = np.broadcast_to(grid.ff[:,None], grid.pupil.shape)[grid.pupil]
    return x, y


def spectrum_energies(grid, case):
    ref, deltas, names = object_fields(grid, case)
    energies = []
    for u in [ref]+deltas:
        f = fftc(u)
        v = f[grid.pupil]
        energies.append((v.real*v.real+v.imag*v.imag).astype(np.float32))
        del f, v
    return energies, ["periodic_background"]+names


def geometry(x, y, cfg, case, shift=(0,0), rotation=0.0, scale=0.0):
    angle = np.deg2rad(rotation)
    u = (x-shift[0])*np.cos(angle)+(y-shift[1])*np.sin(angle)
    v = -(x-shift[0])*np.sin(angle)+(y-shift[1])*np.cos(angle)
    pitch = scales(cfg)["order_pitch_mm"]*(1+scale)
    m, ell = np.rint(u/pitch), np.rint(v/pitch)
    du, dv = u-m*pitch, v-ell*pitch
    if case == "stripes":
        strength = np.sinc(cfg["object"]["stripe_duty"]*m)**2
        active = ell == 0
    else:
        a = np.pi*cfg["object"]["dot_diameter_mm"]/cfg["object"]["pitch_mm"]*np.hypot(m,ell)
        amp = np.ones_like(a)
        nz = a != 0
        amp[nz] = 2*special.j1(a[nz])/a[nz]
        strength = amp*amp
        active = np.ones_like(strength, dtype=bool)
    return du, dv, strength, active


def transmission_at(x, y, cfg, case, design, shift=(0,0), width_error=0.0,
                    rotation=0.0, scale=0.0, leak=None):
    if design.get("shape") == "none":
        return np.ones(np.broadcast_shapes(np.shape(x),np.shape(y)), dtype=np.float32)
    du, dv, strength, active = geometry(x,y,cfg,case,shift,rotation,scale)
    if design["shape"] == "rods_reference":
        blocked = np.abs(du) <= design["width_mm"]/2
    else:
        active &= strength >= design["threshold"]
        w = np.maximum(cfg["manufacturing"]["minimum_feature_mm"],
                       design["width_mm"]*strength**design["beta"])+width_error
        if design["shape"] == "circle":
            distance = np.hypot(du,dv)
        else:
            distance = np.maximum(np.abs(du),np.abs(dv))
        blocked = active & (distance <= w/2)
    if leak is None:
        leak = cfg["manufacturing"]["assumed_blocked_intensity_transmission"]
    return np.where(blocked, np.sqrt(leak), 1).astype(np.float32)


def metrics_energy(energies, trans):
    t2 = trans*trans
    ratios = [float(np.sum(e*t2,dtype=np.float64)/np.sum(e,dtype=np.float64)) for e in energies]
    return dict(background_suppression_db=float(-10*np.log10(max(ratios[0],1e-30))),
                defect_retention=ratios[1:], minimum_retention=min(ratios[1:]),
                background_fraction=ratios[0])


def candidate_curves(energies, du, dv, strength, active, cfg, shape, beta, threshold, widths, width_error):
    distance = np.hypot(du,dv) if shape == "circle" else np.maximum(np.abs(du),np.abs(dv))
    enabled = active & (strength >= threshold)
    gain = np.maximum(strength,1e-30)**beta
    # First candidate diameter that blocks each sample. Histogram + cumulative
    # sum computes ALL candidate energies exactly on this sampled Fourier grid.
    twice_distance = 2*distance-width_error
    q = twice_distance/gain
    q[twice_distance <= cfg["manufacturing"]["minimum_feature_mm"]] = 0
    q[~enabled] = np.inf
    bins = np.searchsorted(widths,q,side="left")
    leak = cfg["manufacturing"]["assumed_blocked_intensity_transmission"]
    curves = []
    for energy in energies:
        total = np.sum(energy,dtype=np.float64)
        accum = np.cumsum(np.bincount(bins,weights=energy,minlength=len(widths)+1))[:len(widths)]
        curves.append(np.maximum(0,total-(1-leak)*accum)/total)
    return np.asarray(curves)


def optimize(grid, case):
    started = time.time()
    cfg = grid.cfg
    energies,names = spectrum_energies(grid,case)
    x,y = pupil_coordinates(grid)
    op,man = cfg["optimization"],cfg["manufacturing"]
    widths = np.round(np.arange(op["width_min_mm"],op["width_max_mm"]+1e-7,op["width_step_mm"]),8)
    tol,err = man["lateral_alignment_mm"],man["full_width_tolerance_mm"]
    scenarios = [("nominal",(0,0),0), ("shift_x_undersize",(tol,0),-err),
                 ("shift_y_undersize",(0,tol),-err), ("oversize",(0,0),err)]
    groups = [(s,b,t) for s in op["shapes"] for b in op["strength_exponents"]
              for t in op["coefficient_power_thresholds"]]
    stores = {g:[] for g in groups}
    for tag,shift,dw in scenarios:
        du,dv,strength,active = geometry(x,y,cfg,case,shift)
        for shape,beta,threshold in groups:
            curves = candidate_curves(energies,du,dv,strength,active,cfg,shape,beta,threshold,widths,dw)
            stores[(shape,beta,threshold)].append(curves)
        print(f"{case}: optimization scenario {tag} complete ({time.time()-started:.1f}s)",flush=True)
    rows = []
    for (shape,beta,threshold),curves in stores.items():
        cube = np.asarray(curves)
        for k,w in enumerate(widths):
            bg = cube[:,0,k]
            retention = cube[:,1:,k]
            minimum = float(retention.min())
            feasible = minimum >= op["minimum_defect_energy_retention"]
            worst = float(-10*np.log10(max(bg.max(),1e-30)))
            rows.append(dict(case=case,shape=shape,beta=beta,threshold=threshold,width_mm=float(w),
                             nominal_suppression_db=float(-10*np.log10(cube[0,0,k])),
                             worst_suppression_db=worst,nominal_minimum_retention=float(cube[0,1:,k].min()),
                             worst_minimum_retention=minimum,feasible=bool(feasible),
                             worst_energy_ratio_gain_db=worst+10*np.log10(max(minimum,1e-30))))
    valid = [r for r in rows if r["feasible"]]
    if not valid:
        raise RuntimeError(f"No feasible filter for {case} at the predefined retention threshold")
    valid.sort(key=lambda r:(r["worst_suppression_db"],r["worst_minimum_retention"]),reverse=True)
    # Retest top candidates over a richer deterministic tolerance set. All
    # shortlisted points remain in the CSV; this is not a global robust proof.
    scenarios_full = tolerance_scenarios(cfg)
    shortlist = valid[:12]
    robust_results=[]
    for rank,design in enumerate(shortlist):
        results=[]
        for scenario in scenarios_full:
            kwargs = {k:v for k,v in scenario.items() if k != "name"}
            t = transmission_at(x,y,cfg,case,design,**kwargs)
            results.append(dict(scenario=scenario["name"],**metrics_energy(energies,t)))
        worst_s=min(r["background_suppression_db"] for r in results)
        worst_r=min(r["minimum_retention"] for r in results)
        robust_results.append(dict(design=design,results=results,worst_suppression_db=worst_s,
                                   worst_minimum_retention=worst_r,feasible=worst_r>=op["minimum_defect_energy_retention"]))
        print(f"{case}: finalist {rank+1}/{len(shortlist)} worst S={worst_s:.3f} dB R={worst_r:.4f}",flush=True)
    robust_valid=[r for r in robust_results if r["feasible"]]
    if not robust_valid:
        raise RuntimeError("Shortlist has no robust feasible member; broaden search explicitly.")
    winner=max(robust_valid,key=lambda r:(r["worst_suppression_db"],r["worst_minimum_retention"]))
    result=dict(case=case,design=winner["design"],robust_worst_suppression_db=winner["worst_suppression_db"],
                robust_worst_minimum_retention=winner["worst_minimum_retention"],
                tolerance_results=winner["results"],shortlist=robust_results,
                energy_components=names,candidate_count=len(rows),feasible_count=len(valid),
                grid=dict(n=grid.n,window_mm=grid.window,dx_um=grid.dx*1000,dff_mm=grid.dff),
                elapsed_seconds=time.time()-started)
    save_csv(OUT/f"{case}_candidate_scan.csv",rows)
    save_json(OUT/f"{case}_optimization.json",result)
    del energies,x,y
    gc.collect()
    return result


def tolerance_scenarios(cfg):
    t=cfg["manufacturing"]["lateral_alignment_mm"]
    dw=cfg["manufacturing"]["full_width_tolerance_mm"]
    rot=cfg["manufacturing"]["rotation_tolerance_deg"]
    scale=cfg["manufacturing"]["scale_tolerance_fraction"]
    return [dict(name="nominal")]+[
        dict(name=f"xy_{i}",shift=(t*np.cos(i*np.pi/4),t*np.sin(i*np.pi/4)),width_error=-dw,
             rotation=rot if i%2 else -rot,scale=scale if i%3 else -scale) for i in range(8)]+[
        dict(name="oversize",width_error=dw),dict(name="undersize",width_error=-dw)]


def save_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_transmission(grid,case,design,**kwargs):
    # Work row-wise to bound peak memory on 8192² validation grids.
    result=np.zeros((grid.n,grid.n),dtype=np.float32)
    for first in range(0,grid.n,128):
        sl=slice(first,min(first+128,grid.n))
        result[sl]=transmission_at(grid.ff[None,:],grid.ff[sl,None],grid.cfg,case,design,**kwargs)
    result*=grid.pupil
    return result


def crop_view(array,grid,half=.255,target=1200):
    idx=np.flatnonzero(np.abs(grid.x)<=half)
    crop=array[idx[0]:idx[-1]+1,idx[0]:idx[-1]+1]
    stride=max(1,int(np.ceil(crop.shape[0]/target)))
    return crop[::stride,::stride].copy()


def detector_image(intensity,grid):
    cfg=grid.cfg
    equiv=cfg["detector"]["pixel_size_mm"]/cfg["optics"]["magnification"]
    block=int(round(equiv/grid.dx))
    if block<1 or not np.isclose(block*grid.dx,equiv,rtol=1e-5):
        raise ValueError("Detector integration requires integer sampling ratio.")
    count=cfg["detector"]["pixels"]
    width=block*count
    if width>grid.n:
        raise ValueError("Computational window too small for detector")
    lo=(grid.n-width)//2
    image=intensity[lo:lo+width,lo:lo+width].reshape(count,block,count,block).mean(axis=(1,3))
    # True inversion and intensity scaling from object-equivalent to image side.
    return np.roll(np.flip(image,axis=(0,1)),1,axis=(0,1))/cfg["optics"]["magnification"]**2


def propagate(grid,case,design,tag="nominal",hard=False,leak=None,apodization=False,save_views=True):
    ref,deltas,names=object_fields(grid,case,hard)
    trans=full_transmission(grid,case,design,leak=leak)
    pupil=grid.pupil.astype(np.float32)
    if apodization:
        sine2=(grid.ff[:,None]**2+grid.ff[None,:]**2)/scales(grid.cfg)["effective_fourier_focal_length_mm"]**2
        apod=np.maximum(0,1-sine2)**.25
        pupil*=apod
        trans*=apod
        del apod,sine2
    fref=fftc(ref)
    eref0=ifftc(fref*pupil)
    eref=ifftc(fref*trans)
    iref0=np.abs(eref0)**2
    iref=np.abs(eref)**2
    total0=float(np.sum(iref0,dtype=np.float64))
    total=float(np.sum(iref,dtype=np.float64))
    valid_roi=(np.hypot(grid.x[:,None],grid.x[None,:])<.18)
    center_supp=-10*np.log10(max(float(np.sum(iref[valid_roi],dtype=np.float64)/np.sum(iref0[valid_roi],dtype=np.float64)),1e-30))
    combined0=eref0.copy()
    combined=eref.copy()
    records=[]
    delta_view=None
    for name,d in zip(names,deltas):
        fd=fftc(d)
        e0=ifftc(fd*pupil)
        e=ifftc(fd*trans)
        combined0+=e0
        combined+=e
        roi=ndimage.maximum_filter(np.abs(d)>0,size=int(round(.004/grid.dx))*2+1)
        e0sq=np.abs(e0)**2
        esq=np.abs(e)**2
        records.append(dict(name=name,energy_retention=float(np.sum(esq,dtype=np.float64)/np.sum(e0sq,dtype=np.float64)),
                            roi_energy_retention=float(np.sum(esq[roi],dtype=np.float64)/np.sum(e0sq[roi],dtype=np.float64)),
                            halo_fraction=float(np.sum(esq[~roi],dtype=np.float64)/max(np.sum(esq,dtype=np.float64),1e-30)),
                            unfiltered_halo_fraction=float(np.sum(e0sq[~roi],dtype=np.float64)/max(np.sum(e0sq,dtype=np.float64),1e-30)),
                            halo_energy_over_unfiltered_total=float(np.sum(esq[~roi],dtype=np.float64)/max(np.sum(e0sq,dtype=np.float64),1e-30)),
                            halo_peak_over_core_peak=float(np.max(esq[~roi])/max(np.max(esq[roi]),1e-30)),
                            image_difference_abs_integral=float(np.sum(np.abs(np.abs(eref+e)**2-iref)[roi],dtype=np.float64)),
                            actual_roi_peak_over_reference_mean=float(np.max(np.abs(eref+e)[roi]**2)/max(np.mean(iref[roi]),1e-30)),
                            roi_pixels=int(roi.sum())))
        if name==names[0]:
            delta_view=crop_view(esq,grid,half=.06,target=800)
        del fd,e0,e,roi,e0sq,esq
    actual0=np.abs(combined0)**2
    actual=np.abs(combined)**2
    row=dict(case=case,tag=tag,grid_n=grid.n,window_mm=grid.window,dx_um=grid.dx*1000,dff_mm=grid.dff,
             background_suppression_db=float(-10*np.log10(max(total/total0,1e-30))),
             central_background_suppression_db=float(center_supp),
             minimum_retention=min(r["energy_retention"] for r in records),
             minimum_roi_retention=min(r["roi_energy_retention"] for r in records),
             actual_total_throughput=float(np.sum(actual,dtype=np.float64)/np.sum(actual0,dtype=np.float64)),
             defect_results=records,hard_illumination=hard,apodization=apodization,
             blocked_intensity_transmission=grid.cfg["manufacturing"]["assumed_blocked_intensity_transmission"] if leak is None else leak,
             parseval_relative_error=float(abs(total-float(np.sum(np.abs(fref*trans)**2,dtype=np.float64)))/max(total,1e-30)))
    if save_views:
        views=dict(object=crop_view(ref+sum(deltas),grid),unfiltered=crop_view(actual0,grid),
                   filtered=crop_view(actual,grid),reference_filtered=crop_view(iref,grid),
                   zoom_unfiltered=crop_view(actual0,grid,half=.06,target=800),
                   zoom_filtered=crop_view(actual,grid,half=.06,target=800),isolated_delta=delta_view,
                   spectrum=np.abs(fref[::max(1,grid.n//1200),::max(1,grid.n//1200)])**2,
                   filter=trans[::max(1,grid.n//1200),::max(1,grid.n//1200)],
                   detector=detector_image(actual,grid),
                   image_difference=crop_view(actual-iref,grid),
                   ff_extent=np.array([grid.ff[0],grid.ff[-1]]))
        np.savez_compressed(OUT/f"{case}_{tag}_views.npz",**views)
    save_json(OUT/f"{case}_{tag}_propagation.json",row)
    print(f"{case} {tag}: S={row['background_suppression_db']:.3f} dB, min R={row['minimum_retention']:.4f}, ROI={row['minimum_roi_retention']:.4f}",flush=True)
    return row


def export_design(cfg,case,design):
    pitch=scales(cfg)["order_pitch_mm"]
    radius=cfg["optics"]["filter_clear_diameter_mm"]/2
    limit=int(np.ceil((radius+design["width_mm"])/pitch))
    centers=[]
    for m in range(-limit,limit+1):
        for ell in ([0] if case=="stripes" else range(-limit,limit+1)):
            x,y=m*pitch,ell*pitch
            _,_,strength,active=geometry(np.array(x),np.array(y),cfg,case)
            if not active or strength < design["threshold"]:
                continue
            w=max(cfg["manufacturing"]["minimum_feature_mm"],design["width_mm"]*float(strength)**design["beta"])
            if np.hypot(x,y)>radius+w/np.sqrt(2):
                continue
            centers.append(dict(m=m,ell=ell,x_mm=x,y_mm=y,width_mm=w,shape=design["shape"],coefficient_power=float(strength)))
    save_csv(OUT/f"{case}_manufacturing_centers.csv",centers)
    # 60 mm substrate drawing, 50 mm clear beam shown on separate guide layer.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="60mm" viewBox="-30 -30 60 60">',
         '<title>Patterned opaque stops on UV fused silica; black = blocking film</title>',
         '<g id="OPAQUE_PATTERN" fill="black">']
    dxf=['0','SECTION','2','HEADER','9','$INSUNITS','70','4','0','ENDSEC','0','SECTION','2','ENTITIES']
    for c in centers:
        x,y,w=c["x_mm"],c["y_mm"],c["width_mm"]
        if design["shape"]=="circle":
            svg.append(f'<circle cx="{x:.8f}" cy="{-y:.8f}" r="{w/2:.8f}"/>')
            dxf+=['0','CIRCLE','8','OPAQUE_BOUNDARY','10',str(x),'20',str(y),'30','0','40',str(w/2)]
        else:
            svg.append(f'<rect x="{x-w/2:.8f}" y="{-y-w/2:.8f}" width="{w:.8f}" height="{w:.8f}"/>')
            vertices=[(x-w/2,y-w/2),(x+w/2,y-w/2),(x+w/2,y+w/2),(x-w/2,y+w/2)]
            for a,b in zip(vertices,vertices[1:]+vertices[:1]):
                dxf+=['0','LINE','8','OPAQUE_BOUNDARY','10',str(a[0]),'20',str(a[1]),'11',str(b[0]),'21',str(b[1])]
    svg+=['</g>','<g id="GUIDES_NOT_CHROME" fill="none" stroke="#185abd" stroke-width="0.03">',
          '<circle r="25"/><rect x="-30" y="-30" width="60" height="60"/></g>','</svg>']
    dxf+=['0','CIRCLE','8','GUIDE_50MM_NOT_CHROME','10','0','20','0','40','25','0','ENDSEC','0','EOF']
    (OUT/f"{case}_mask.svg").write_text('\n'.join(svg),encoding="utf-8")
    (OUT/f"{case}_mask.dxf").write_text('\n'.join(dxf)+'\n',encoding="ascii")
    widths=[c["width_mm"] for c in centers]
    qa=dict(number_of_stops=len(centers),minimum_size_mm=min(widths),maximum_size_mm=max(widths),
            conservative_clear_gap_mm=pitch-max(widths),minimum_feature_required_mm=cfg["manufacturing"]["minimum_feature_mm"],
            minimum_gap_required_mm=cfg["manufacturing"]["minimum_clear_gap_mm"],
            substrate_size_mm=[60,60],effective_beam_diameter_mm=50,
            dxf_units="mm",black_svg_layer="OPAQUE_PATTERN",guides_are_not_fabrication_patterns=True)
    qa["passes_nominal_drc"]=qa["minimum_size_mm"]>=qa["minimum_feature_required_mm"]-1e-8 and qa["conservative_clear_gap_mm"]>=qa["minimum_gap_required_mm"]
    save_json(OUT/f"{case}_manufacturing_check.json",qa)
    return qa


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["optimize","propagate","validate","all"],default="all")
    parser.add_argument("--case",choices=["stripes","dots","all"],default="all")
    args=parser.parse_args()
    cfg=config()
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/"model_parameters.json",dict(config=cfg,derived=scales(cfg)))
    cases=cfg["run"]["cases"] if args.case=="all" else [args.case]
    for case in cases:
        grid=make_grid(cfg)
        if args.stage in ("optimize","all"):
            result=optimize(grid,case)
        else:
            result=json.loads((OUT/f"{case}_optimization.json").read_text(encoding="utf-8"))
        design=result["design"]
        export_design(cfg,case,design)
        if args.stage in ("propagate","all"):
            propagate(grid,case,design)
        if args.stage in ("validate","all"):
            validations=[]
            for tag,kw in [("hard_edge",dict(hard=True)),("ideal_opacity",dict(leak=0)),
                           ("OD3",dict(leak=.001)),("apodized",dict(apodization=True))]:
                validations.append(propagate(grid,case,design,tag,save_views=False,**kw))
                gc.collect()
            for tag,n,window in [("fine_object",8192,1.024),("fine_fourier",8192,2.048)]:
                vg=make_grid(cfg,n,window)
                validations.append(propagate(vg,case,design,tag,save_views=False))
                del vg
                gc.collect()
            save_json(OUT/f"{case}_validation.json",validations)
        del grid
        gc.collect()


if __name__=="__main__":
    main()
