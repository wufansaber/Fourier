"""Check that the saved real Zemax model physically blocks the grating orders."""
import csv
import json
from real_objective_filter import OUT,ray,WAVE_UM
from zosapi_utils import ZOSApplication

def main():
    checks=[]
    with ZOSApplication() as zos:
        zos.system.LoadFile(str(OUT/'models'/'real_collector_relay_tube_optimized.zos'),False)
        apertures=[dict(surface=i,type=str(zos.system.LDE.GetSurfaceAt(i).ApertureData.CurrentType),
                        semi_diameter_mm=float(zos.system.LDE.GetSurfaceAt(i).SemiDiameter)) for i in range(29)]
        outside=[dict(py=p,trace=ray(zos,28,py=p)) for p in (1.,1.01,1.1,1.2)]
        (OUT/'aperture_audit.json').write_text(json.dumps(dict(surfaces=apertures,outside_pupil_rays=outside),indent=2))
        print('outside pupil',outside,flush=True)
        samples=[]
        for q in [float(i) for i in range(-4,5)]+[i+.5 for i in range(-4,4)]:
            target=q*WAVE_UM/5.
            lo,hi=-1.,1.
            for _ in range(36):
                px=(lo+hi)/2
                ob=ray(zos,0,px=px)
                if ob['l']>target:lo=px
                else:hi=px
            samples.append((q,px))
        for kind in ('open','rods'):
            zos.system.LoadFile(str(OUT/'models'/('real_regular_'+kind+'.zos')),False)
            for order,px in samples:
                r=ray(zos,28,px=px)
                checks.append(dict(model=kind,order=order,px=px,error=r['error'],vignette=r['vignette']))
        expected=all(c['error']==0 and c['vignette']==(25 if c['model']=='rods' and c['order'].is_integer() else 0) for c in checks)
        with (OUT/'uda_ray_blocking.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=checks[0].keys());w.writeheader();w.writerows(checks)
        (OUT/'uda_validation.json').write_text(json.dumps(dict(all_expected=expected,checks=34),indent=2))
        print('UDA expected blocking',expected,flush=True)
        assert expected,checks

if __name__=='__main__':main()
