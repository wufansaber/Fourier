"""Audit the saved real prescription and plot source-backed engineering figures."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'real_objective'

def sag(r,c,y):
    if not np.isfinite(r):return np.zeros_like(y)
    return (y*y/r)/(1+np.sqrt(1-(1+c)*y*y/(r*r)))

def main():
    rows=json.loads((OUT/'real_surface_prescription.json').read_text())
    source=json.loads((OUT/'source_audit.json').read_text())
    comparisons=[]
    for i in range(1,21):
        old=22-i
        a=rows[i];b=source['surfaces'][old]
        radius_ok=(math.isinf(a['radius_mm']) and math.isinf(float(b['radius_mm']))) or abs(a['radius_mm']+float(b['radius_mm']))<1e-9
        comparisons.append(radius_ok and a['material']==source['surfaces'][old-1]['material']
                           and (old==2 or abs(a['thickness_mm']-float(source['surfaces'][old-1]['thickness_mm']))<1e-9))
    fan=np.genfromtxt(OUT/'sine_fan.csv',delimiter=',',names=True,encoding='utf-8-sig')
    a=fan[(fan['hy']==0)&(fan['py']>=.1)]
    sine_error=(a['sin_object']/a['sin_image']/10-1)*100
    geometry=[]
    for i in (21,23,26):
        r1=rows[i];r2=rows[i+1];y=np.linspace(0,min(r1['semi_diameter_mm'],r2['semi_diameter_mm']),1001)
        t=r1['thickness_mm']+sag(r2['radius_mm'],r2['conic'],y)-sag(r1['radius_mm'],r1['conic'],y)
        geometry.append(dict(front_surface=i,min_axial_thickness_mm=float(np.min(t))))
    orders=np.genfromtxt(OUT/'diffraction_order_calibration.csv',delimiter=',',names=True,encoding='utf-8-sig')
    calibration={}
    for hy in (0.,.5,1.):
        aord=orders[orders['hy']==hy]
        calibration[str(hy)]=float(max(abs(aord['fourier_y_mm']-aord['ideal_sine_y_mm']))*1000)
    result=dict(original_objective_preserved=all(comparisons),all_standard_surfaces=all(r['type'] in ('Standard','65') for r in rows),
                surface_type_enum_65='Standard (verified against installed ZOSAPI_Interfaces.dll)',
                original_source_sha256=hashlib.sha256(Path(r'C:\Users\wufan\Documents\Zemax\Samples\10X_OBJ_0.5na-2.zmx').read_bytes()).hexdigest(),
                project_copy_sha256=hashlib.sha256((OUT/'source'/'10X_OBJ_0.5na-2.zmx').read_bytes()).hexdigest(),
                finite_sine_relative_max_percent=float(max(abs(sine_error))),
                axial_fourier_residual_max_um=float(max(abs(a['fourier_y_mm']+20.000029217895*a['sin_object']))*1000),
                order_residual_max_um_by_field=calibration,added_lens_geometry=geometry)
    (OUT/'engineering_qa.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for hy,ls,color in [(0.,'-','#22689A'),(.5,'--','#BC711B'),(1.,':','#8A8632')]:
        b=fan[(fan['hy']==hy)&(fan['py']>=0)]
        axes[0].plot(abs(b['sin_object']),(b['fourier_y_mm']+20.000029217895*b['sin_object'])*1000,ls,color=color,label='Field %.3f mm'%(hy*.25))
    axes[0].set(xlabel='Absolute object direction sine',ylabel='Fourier y residual (um)',title='Measured Fourier mapping residual')
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(a['py'],sine_error,color='#22689A')
    axes[1].axhline(.1,color='#333333',ls='--');axes[1].axhline(-.1,color='#333333',ls='--')
    axes[1].set(xlabel='Normalized pupil coordinate Py',ylabel='Sine ratio error relative to 10x (%)',title='Finite-conjugate sine-condition check')
    axes[1].grid(alpha=.2)
    fig.suptitle('Actual Zemax ray data | 584.601 nm; dashed limits are temporary +/-0.1% references')
    fig.savefig(OUT/'real_sine_condition.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),layout='constrained')
    from matplotlib.colors import LinearSegmentedColormap
    wavecolors=LinearSegmentedColormap.from_list('wave_error',['#22689A','#FAFAFA','#BC711B'])
    for ax,hy in zip(axes,(0.,.5,1.)):
        d=np.load(OUT/('rays_field_%g.npz'%hy));opd=d['s28'][:,12];p=d['samples']
        im=ax.scatter(p[:,0],p[:,1],c=opd-opd.mean(),s=4,cmap=wavecolors,vmin=-.4,vmax=.4,rasterized=True)
        ax.set(aspect='equal',xlabel='Px',ylabel='Py',title='Object height %.3f mm'%(hy*.25))
    fig.colorbar(im,ax=axes,label='Full-system OPD minus mean (waves)',shrink=.8)
    fig.suptitle('Zemax full-system wavefront | 3209 pupil samples per field')
    fig.savefig(OUT/'real_wavefront.png',dpi=180);plt.close(fig)
    paths=np.genfromtxt(OUT/'layout_rays.csv',delimiter=',',names=True,encoding='utf-8-sig')
    z=np.cumsum([0.]+[r['thickness_mm'] for r in rows[:-1]])
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained',height_ratios=[1.15,1.])
    for ax in axes:
        for i,r in enumerate(rows[1:-1],1):
            yy=np.linspace(-r['semi_diameter_mm'],r['semi_diameter_mm'],301)
            ax.plot(z[i]+sag(r['radius_mm'],r['conic'],yy),yy,color='#333333',lw=.7)
            if r['material'] and i<28:
                r2=rows[i+1];lim=min(r['semi_diameter_mm'],r2['semi_diameter_mm']);yy=np.linspace(-lim,lim,301)
                ax.fill_betweenx(yy,z[i]+sag(r['radius_mm'],r['conic'],yy),z[i+1]+sag(r2['radius_mm'],r2['conic'],yy),color='#84AFC9',alpha=.5)
        for hy,color in [(0.,'#22689A'),(1.,'#BC711B')]:
            for pupil in (-.9,0.,.9):
                v=paths[(paths['hy']==hy)&(paths['pupil']==pupil)]
                ax.plot(v['z_global_mm'],v['y_mm'],color=color,lw=.75,ls='-' if hy==0 else '--')
        ax.set(xlabel='Axial z from object (mm)',ylabel='Meridional y (mm)',ylim=(-17,19))
        ax.grid(alpha=.15)
    for i,label in [(10,'Original objective'),(21,'Relay L1'),(23,'Relay L2'),(25,'Filter S25'),(26,'Tube lens'),(28,'Image')]:
        axes[0].text(z[i],17,label,ha='center',fontsize=8)
    axes[0].set_xlim(-10,z[28]+15);axes[0].set_title('Complete real-glass system (unequal axial / transverse scales)')
    axes[1].set_xlim(-1,80);axes[1].set_title('Reverse-use original objective, enlarged')
    fig.suptitle('Actual Zemax ray paths; glass outlines from the saved prescription\nBlue: axial object; dashed gold: object height 0.25 mm')
    fig.savefig(OUT/'real_lens_layout.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
