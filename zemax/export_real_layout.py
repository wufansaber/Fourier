"""Export actual ray paths for a scientific lens-layout figure."""
import csv
from real_objective_filter import OUT
from validate_real_objective import batch
from zosapi_utils import ZOSApplication

def main():
    rows=[]
    with ZOSApplication() as zos:
        s=zos.system;s.LoadFile(str(OUT/'models'/'real_collector_relay_tube_optimized.zos'),False)
        z=0.
        for surf in range(29):
            if surf>0:z+=s.LDE.GetSurfaceAt(surf-1).Thickness
            for hy in (0.,1.):
                a=batch(zos,surf,[(0.,-.9),(0.,0.),(0.,.9)],hy)
                for k,r in enumerate(a):
                    rows.append(dict(surface=surf,hy=hy,pupil=(-.9,0.,.9)[k],z_global_mm=z+r[5],
                                     y_mm=r[4],error=r[1],vignette=r[2]))
        with (OUT/'layout_rays.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
        print('exported layout rays',len(rows),flush=True)

if __name__=='__main__':main()
