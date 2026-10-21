"""Small, bounded native POP and ray API diagnostic for the real objective."""
import argparse
from pathlib import Path
import numpy as np
from zosapi_utils import ZOSApplication
from real_objective_filter import OUT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--end',type=int,default=1)
    p.add_argument('--sampling',type=int,default=8)
    p.add_argument('--width',type=float,default=1.)
    p.add_argument('--model',default='real_collector_relay_tube_optimized.zos')
    p.add_argument('--no-ray-rescale',action='store_true')
    p.add_argument('--scan',action='store_true')
    a=p.parse_args()
    with ZOSApplication() as zos:
        s,api=zos.system,zos.ZOSAPI
        s.LoadFile(str(OUT/'models'/a.model),False)
        if a.no_ray_rescale:
            for i in range(1,s.LDE.NumberOfSurfaces):
                d=s.LDE.GetSurfaceAt(i).PhysicalOpticsData
                d.DoNotRescaleBeamSizeUsingRayData=True
                d.AutoResample=False
                d.ResampleAfterRefraction=False
        t=s.Tools.OpenBatchRayTrace()
        data=t.CreateNormUnpol(2,api.Tools.RayTrace.RaysType.Real,28)
        print('ReadNextResult',data.ReadNextResult.__doc__,flush=True)
        print('OPDMode',list(n for n in dir(api.Tools.RayTrace.OPDMode) if not n.startswith('_')),flush=True)
        t.Close()
        pop=s.Analyses.New_Analysis_SettingsFirst(api.Analysis.AnalysisIDM.PhysicalOpticsPropagation)
        cfg=api.Analysis.PhysicalOptics.IAS_PhysicalOpticsPropagation(pop.GetSettings())
        print('settings',','.join(n for n in dir(cfg) if not n.startswith('_')),flush=True)
        cfg.StartSurface.SetSurfaceNumber(1)
        cfg.EndSurface.SetSurfaceNumber(a.end)
        cfg.Wavelength.SetWavelengthNumber(1)
        cfg.Field.SetFieldNumber(1)
        cfg.BeamType=api.Analysis.PhysicalOptics.POPBeamTypes.TopHat
        for i in range(cfg.NumberOfParameters):
            print('parameter',i,cfg.GetParameterName(i),cfg.GetParameterValue(i),flush=True)
        cfg.SetParameterValue(0,.55)
        cfg.SetParameterValue(1,.55)
        sampling=getattr(api.Analysis.SampleSizes,'S_%dx%d'%(2**(a.sampling+4),2**(a.sampling+4)))
        cfg.XSampling=sampling
        cfg.YSampling=sampling
        cfg.XWidth=a.width
        cfg.YWidth=a.width
        cfg.UseTotalPower=True
        cfg.TotalPower=1.
        cfg.UsePolarization=False
        cfg.ZoomIn=api.Analysis.PhysicalOptics.POPZoomTypes.NoZoom
        print('power mode',cfg.UseTotalPower,cfg.UsePeakIrradiance,'data',cfg.DataType,'zoom',cfg.ZoomIn,flush=True)
        diag=OUT/'diagnostics'
        diag.mkdir(exist_ok=True)
        cfg.SaveTo(str(diag/'real_pop_settings.CFG'))
        for end in ((1,2,3,5,10,20,25,28) if a.scan else (a.end,)):
            cfg.EndSurface.SetSurfaceNumber(end)
            print('RUN surface',end,flush=True)
            pop.ApplyAndWaitForCompletion()
            r=pop.GetResults()
            r.GetTextFile(str(diag/('scan_s%d.txt'%end)))
            for i in range(r.NumberOfMessages):
                m=r.GetMessageAt(i)
                print('message',m.ErrorCode,m.Text,flush=True)
            if r.NumberOfDataGrids:
                g=r.GetDataGrid(0)
                arr=np.array(list(g.Values)).reshape(g.Ny,g.Nx)
                print('grid',g.Nx,g.Ny,g.MinX,g.Dx,g.MinY,g.Dy,'power',arr.sum()*g.Dx*g.Dy,'max',arr.max(),flush=True)
                np.savez_compressed(diag/('scan_s%d.npz'%end),irradiance=arr,x=g.MinX+np.arange(g.Nx)*g.Dx,y=g.MinY+np.arange(g.Ny)*g.Dy)
            else:break
        pop.Close()

if __name__=='__main__':main()
