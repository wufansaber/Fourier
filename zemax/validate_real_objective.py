"""Export real Zemax ray directions, Fourier intercepts and full-system OPD."""
import argparse
import csv
import json
import math
import numpy as np
from real_objective_filter import OUT, ray, WAVE_UM
from zosapi_utils import ZOSApplication

def batch(zos, surface, samples, hy=0.):
    t=zos.system.Tools.OpenBatchRayTrace()
    try:
        d=zos.ZOSAPI.Tools.RayTrace.IRayTraceNormUnpolData(
            t.CreateNormUnpol(len(samples),zos.ZOSAPI.Tools.RayTrace.RaysType.Real,surface))
        mode=(zos.ZOSAPI.Tools.RayTrace.OPDMode.CurrentAndChief if surface==28
              else getattr(zos.ZOSAPI.Tools.RayTrace.OPDMode,'None'))
        for px,py in samples:d.AddRay(1,0.,hy,float(px),float(py),mode)
        t.RunAndWaitForCompletion()
        d.StartReadingResults()
        values=[]
        while True:
            row=d.ReadNextResult(0,0,0,*([0.]*11))
            if not row[0]:break
            values.append([float(v) for v in row[1:]])
        return np.array(values)
    finally:t.Close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grid',type=int,default=65)
    a=p.parse_args()
    with ZOSApplication() as zos:
        s=zos.system
        s.LoadFile(str(OUT/'models'/'real_collector_relay_tube_optimized.zos'),False)
        rows=[]
        for i in range(s.LDE.NumberOfSurfaces):
            r=s.LDE.GetSurfaceAt(i)
            rows.append(dict(surface=i,type=str(r.Type),radius_mm=float(r.Radius),
                             thickness_mm=float(r.Thickness),material=str(r.Material),
                             semi_diameter_mm=float(r.SemiDiameter),conic=float(r.Conic),comment=str(r.Comment)))
        (OUT/'real_surface_prescription.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        with (OUT/'real_surface_prescription.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        grid=np.linspace(-1.,1.,a.grid)
        samples=[(x,y) for y in grid for x in grid if x*x+y*y<=1.]
        # Columns: ray_number,error,vignette,x,y,z,l,m,n,ln,mn,nn,opd,intensity.
        fields=[]
        for hy in (0.,.5,1.):
            data={}
            for surf in (0,25,28):
                arr=batch(zos,surf,samples,hy)
                print('batch',hy,surf,arr.shape,'first',arr[0],flush=True)
                data['s%d'%surf]=arr
            np.savez_compressed(OUT/('rays_field_%g.npz'%hy),samples=np.array(samples),**data)
            valid=(data['s28'][:,1]==0)&(data['s28'][:,2]==0)
            opd=data['s28'][valid,12]
            fields.append(dict(hy=hy,rays=len(samples),valid=int(valid.sum()),
                               opd_rms_about_mean_waves=float(np.std(opd)),
                               opd_pv_waves=float(np.ptp(opd))))
        # Dense meridional calibration; actual sine direction at object, not pupil coordinate.
        fan=[]
        for hy in (0.,.5,1.):
            samples=[(0.,float(p)) for p in np.linspace(-1.,1.,401)]
            ob=batch(zos,0,samples,hy);fp=batch(zos,25,samples,hy);im=batch(zos,28,samples,hy)
            for j,(_,py) in enumerate(samples):
                fan.append(dict(hy=hy,py=py,sin_object=ob[j,7],sin_image=im[j,7],
                                fourier_y_mm=fp[j,4],image_y_mm=im[j,4],
                                opd_waves=im[j,12],error=im[j,1],vignette=im[j,2]))
        with (OUT/'sine_fan.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fan[0].keys());w.writeheader();w.writerows(fan)
        orders=[]
        for hy in (0.,.5,1.):
            for m in range(-4,5):
                target=m*WAVE_UM/5.
                # Existing normalized pupil runs opposite to the object direction.
                lo,hi=-1.,1.
                for j in range(38):
                    py=(lo+hi)/2
                    ob=ray(zos,0,hy=hy,py=py)
                    if ob['m']>target:lo=py
                    else:hi=py
                fp=ray(zos,25,hy=hy,py=py)
                orders.append(dict(hy=hy,order=m,py=py,sin_object=ob['m'],
                                   fourier_y_mm=fp['y'],ideal_sine_y_mm=-20.000029217895*target,
                                   error=fp['error'],vignette=fp['vignette']))
        with (OUT/'diffraction_order_calibration.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=orders[0].keys());w.writeheader();w.writerows(orders)
        (OUT/'ray_validation_summary.json').write_text(json.dumps(fields,indent=2),encoding='utf-8')
        print('summary',fields,flush=True)

if __name__=='__main__':main()
