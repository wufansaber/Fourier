"""Native OpticStudio POP with explicit interface enums and audited grid export."""
import argparse
import json
import math
import time
import numpy as np
from real_objective_filter import OUT
from zosapi_utils import ZOSApplication

def run(zos,model,end,n,width=1.024):
    s,api=zos.system,zos.ZOSAPI
    s.LoadFile(str(OUT/'models'/(model+'.zos')),False)
    for i in range(1,s.LDE.NumberOfSurfaces):
        d=s.LDE.GetSurfaceAt(i).PhysicalOpticsData
        d.DoNotRescaleBeamSizeUsingRayData=True
        d.AutoResample=False
        d.ResampleAfterRefraction=False
    pop=s.Analyses.New_Analysis_SettingsFirst(api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
    try:
        cfg=api.Analysis.PhysicalOptics.IAS_PhysicalOpticsPropagation(pop.GetSettings())
        cfg.StartSurface.SetSurfaceNumber(1);cfg.EndSurface.SetSurfaceNumber(end)
        cfg.Wavelength.SetWavelengthNumber(1);cfg.Field.SetFieldNumber(1)
        cfg.BeamType=api.Analysis.PhysicalOptics.POPBeamTypes.TopHat
        cfg.SetParameterValue(0,.55);cfg.SetParameterValue(1,.55)
        cfg.SetParameterValue(2,0.);cfg.SetParameterValue(3,0.)
        cfg.XSampling=getattr(api.Analysis.SampleSizes,'S_%dx%d'%(n,n))
        cfg.YSampling=cfg.XSampling
        cfg.XWidth=width;cfg.YWidth=width
        cfg.UseTotalPower=True;cfg.TotalPower=1.
        cfg.UsePolarization=False;cfg.SeparateXY=False
        cfg.DataType=api.Analysis.PhysicalOptics.POPDataTypes.Irradiance
        cfg.ZoomIn=api.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
        directory=OUT/'native_pop';directory.mkdir(exist_ok=True)
        tag='%s_s%d_n%d'%(model,end,n)
        cfg.SaveTo(str(directory/(tag+'.CFG')))
        start=time.time();print('START',tag,flush=True)
        pop.ApplyAndWaitForCompletion();r=pop.GetResults()
        messages=[str(r.GetMessageAt(i).Text) for i in range(r.NumberOfMessages)]
        r.GetTextFile(str(directory/(tag+'.txt')))
        result=dict(model=model,end_surface=end,sampling=n,input_window_mm=width,
                    beam='TopHat',waist_x_mm=.55,waist_y_mm=.55,polarization=False,
                    messages=messages,elapsed_s=time.time()-start,data_grids=int(r.NumberOfDataGrids))
        if r.NumberOfDataGrids:
            g=r.GetDataGrid(0);arr=np.array(list(g.Values)).reshape(g.Ny,g.Nx)
            x=g.MinX+np.arange(g.Nx)*g.Dx;y=g.MinY+np.arange(g.Ny)*g.Dy
            np.savez_compressed(directory/(tag+'.npz'),irradiance=arr,x=x,y=y)
            result.update(nx=int(g.Nx),ny=int(g.Ny),dx_mm=float(g.Dx),dy_mm=float(g.Dy),
                          power_W=float(arr.sum()*g.Dx*g.Dy),peak_W_per_mm2=float(arr.max()),
                          x_min_mm=float(x[0]),x_max_mm=float(x[-1]))
        (directory/(tag+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('DONE',json.dumps(result),flush=True)
        return result
    finally:pop.Close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--end',type=int,default=28);p.add_argument('--grid',type=int,default=2048)
    a=p.parse_args()
    with ZOSApplication() as zos:run(zos,a.model,a.end,a.grid)

if __name__=='__main__':main()
