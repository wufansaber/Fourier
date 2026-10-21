"""Add finite periodic objects and full-size 1 mm rods to the real prescription."""
import csv
import json
import math
import shutil
from pathlib import Path
from real_objective_filter import OUT
from zosapi_utils import ZOSApplication

def rect(x0,x1,y0,y1):return [(x0,y1),(x1,y1),(x1,y0),(x0,y0)]

def objects(defect):
    polygons=[]
    for m in range(-49,50):
        c=m*.005
        x0,x1=c-.00125,c+.00125
        if m==0 and defect:
            polygons.extend([rect(x0,x1,-.25,-.025),rect(x0,x1,.025,.25),
                             rect(x0+.0015,x1+.0015,-.025,.025)])
        else:
            xs=[x0+(x1-x0)*j/8 for j in range(9)]
            polygons.append([(x,math.sqrt(.25**2-x*x)) for x in xs]+
                            [(x,-math.sqrt(.25**2-x*x)) for x in reversed(xs)])
    return polygons

def write_uda(path,polys):
    with path.open('w',encoding='ascii') as f:
        f.write('! Real objective 584.601 nm; dimensions in mm\n')
        for p in polys:
            for x,y in p:f.write('%.12g %.12g\n'%(x,y))
            f.write('0 0\n')

def aperture(zos,surf,name):
    a=surf.ApertureData.CreateApertureTypeSettings(zos.ZOSAPI.Editors.LDE.SurfaceApertureTypes.UserAperture)
    a._S_UserAperture.ApertureFile=name
    a._S_UserAperture.UDAScale=1.
    surf.ApertureData.ChangeApertureTypeSettings(a)

def main():
    rows=list(csv.DictReader((OUT/'diffraction_order_calibration.csv').open(encoding='utf-8-sig')))
    centers=sorted(float(r['fourier_y_mm']) for r in rows if float(r['hy'])==0.)
    gaps=[]
    bounds=[(-16.,centers[0]-.5)]+[(a+.5,b-.5) for a,b in zip(centers[:-1],centers[1:])]+[(centers[-1]+.5,16.)]
    for a,b in bounds:
        assert b>a
        gaps.append(rect(a,b,-16.,16.))
    uda=OUT/'uda';uda.mkdir(exist_ok=True)
    files={}
    for name,polys in [('OBJECT_REGULAR',objects(False)),('OBJECT_DEFECT',objects(True)),('FILTER_1MM',gaps)]:
        filename='REALOBJ5846_'+name+'.UDA'
        write_uda(uda/filename,polys);files[name]=filename
    with ZOSApplication() as zos:
        dest=Path(zos.application.ObjectsDir)/'Apertures'
        for filename in files.values():
            target=dest/filename
            if target.exists() and target.read_bytes()!=(uda/filename).read_bytes():
                raise RuntimeError('Will not overwrite a differing existing UDA: '+str(target))
            shutil.copy2(uda/filename,target)
        zos.application.UpdateFileLists()
        s=zos.system
        for defect in (False,True):
            for mask in (False,True):
                s.LoadFile(str(OUT/'models'/'real_collector_relay_tube_optimized.zos'),False)
                aperture(zos,s.LDE.GetSurfaceAt(1),files['OBJECT_DEFECT' if defect else 'OBJECT_REGULAR'])
                if mask:aperture(zos,s.LDE.GetSurfaceAt(25),files['FILTER_1MM'])
                name='real_%s_%s'%('defect' if defect else 'regular','rods' if mask else 'open')
                s.SystemData.TitleNotes.Title=name+' | 584.601 nm, real refracting surfaces'
                s.SystemData.TitleNotes.Notes='Temporary real-objective test; original objective fixed. Surface 1 is amplitude target, surface 25 is 1 mm opaque rods at measured Fourier orders. Nominal 10x. Object pitch 5 um, circular FOV diameter 0.5 mm. Additional conics are custom, not catalog components. Scalar propagation only.'
                s.SaveAs(str(OUT/'models'/(name+'.zos')))
                s.SaveAs(str(OUT/'models'/(name+'.zmx')))
                print('saved',name,flush=True)
        metadata=dict(object_pitch_um=5.,bar_width_um=2.5,object_diameter_mm=.5,
                      defect_shift_um=1.5,defect_length_um=50.,rod_width_mm=1.,
                      rod_centers_mm=centers,filter_surface=25,object_surface=1,image_surface=28,
                      installed_aperture_directory=str(dest),uda_files=files)
        (OUT/'filter_definition.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

if __name__=='__main__':main()
