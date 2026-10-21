"""Read and perturb a standalone in-memory copy to check stop clipping; no save."""
import json
from real_objective_filter import OUT,ray
from zosapi_utils import ZOSApplication

def main():
    data=[]
    with ZOSApplication() as zos:
        s=zos.system;s.LoadFile(str(OUT/'models'/'real_collector_relay_tube_optimized.zos'),False)
        stop=s.LDE.GetSurfaceAt(17)
        for hy in (0.,1.):
            for py in (-1.01,-1.,1.,1.01):
                r=ray(zos,17,hy=hy,py=py)
                data.append(dict(hy=hy,py=py,stop_y_mm=r['y'],stop_radius_mm=float(stop.SemiDiameter)))
        print(data,flush=True)
        (OUT/'stop_footprint_check.json').write_text(json.dumps(data,indent=2))

if __name__=='__main__':main()
