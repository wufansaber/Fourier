"""Scalar coherent filtering with the measured real-Zemax pupil, not native POP.

Uses the axial, local shift-invariant approximation. No fitted ideal-lens phase
replaces the exported full-system OPD. Object-space coordinates are used for FFT;
nominal physical image coordinates are 10 times these coordinates.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'real_objective'
WL=.000584601
F=20.000029217895

def rectangle_fraction(x,center,width,dx):
    return np.clip((width/2+dx/2-np.abs(x-center))/dx,0,1)

def model(n,window,rod_width=1.,apodization=False,save=False):
    dx=window/n
    x=(np.arange(n)-n/2)*dx
    xx=x[None,:];yy=x[:,None]
    # Exact pixel averages along bar edges; circular boundary has <= 1 pixel error.
    dist=np.abs((x+.0025)%.005-.0025)
    bars=np.clip((.00125+dx/2-dist)/dx,0,1)
    bars*=np.abs(x)<.24625+dx/2
    disk=(xx**2+yy**2<.25**2)
    regular=(disk*bars[None,:]).astype('float32')
    delta=(rectangle_fraction(x,.0015,.0025,dx)-rectangle_fraction(x,0.,.0025,dx))[None,:]
    delta=(delta*rectangle_fraction(x,0.,.050,dx)[:,None]).astype('float32')
    defect=regular+delta
    fan=np.genfromtxt(OUT/'sine_fan.csv',delimiter=',',names=True,encoding='utf-8-sig')
    a=fan[(fan['hy']==0)&(fan['sin_object']>=0)]
    a=np.sort(a,order='sin_object')
    freq=fft.fftfreq(n,dx)
    sx=(freq*WL)[None,:];sy=(freq*WL)[:,None]
    rho=np.sqrt(sx*sx+sy*sy)
    pupil=rho<=a['sin_object'].max()
    opd=np.interp(rho,a['sin_object'],a['opd_waves']).astype('float32')
    mapradius=np.interp(rho,a['sin_object'],np.abs(a['fourier_y_mm']))
    fp_x=-mapradius*np.divide(sx,rho,out=np.zeros_like(rho),where=rho>0)
    definition=json.loads((OUT/'filter_definition.json').read_text())
    mask=np.ones((n,n),dtype=bool)
    for center in definition['rod_centers_mm']:
        mask&=np.abs(fp_x-center)>rod_width/2
    amp=pupil.astype('float32')
    if apodization:amp*=np.maximum(1-rho*rho,.5)**(-.25)
    transfer=(amp*np.exp(2j*np.pi*opd)).astype('complex64')
    spectra={'regular':fft.fft2(regular,workers=4),'defect':fft.fft2(defect,workers=4)}
    outputs={}
    for name,spectrum in spectra.items():
        outputs[name+'_open']=fft.ifft2(spectrum*transfer,workers=4)
        outputs[name+'_rods']=fft.ifft2(spectrum*transfer*mask,workers=4)
    roi=(xx*xx+yy*yy)<.20**2
    local=(np.abs(xx)<.006)&(np.abs(yy)<.025)
    background=roi&~((np.abs(xx)<.02)&(np.abs(yy)<.06))
    metrics=[]
    for kind in ('open','rods'):
        ur=outputs['regular_'+kind];ud=outputs['defect_'+kind]
        ir=np.abs(ur)**2;idef=np.abs(ud)**2
        signal=np.abs(ud-ur)**2
        metrics.append(dict(kind=kind,periodic_power_roi=float(ir[roi].sum()*dx*dx),
                            defect_field_energy=float(signal.sum()*dx*dx),
                            defect_peak_to_regular_background=float(idef[local].max()/ir[background].mean()),
                            intensity_difference_rms=float(np.sqrt(np.mean((idef[roi]-ir[roi])**2)))))
    b,r=metrics
    summary=dict(grid=n,window_mm=window,dx_um=dx*1000,rod_width_mm=rod_width,
                 apodization=apodization,metrics=metrics,
                 suppression_db=float(10*np.log10(b['periodic_power_roi']/r['periodic_power_roi'])),
                 defect_field_energy_retention_percent=100*r['defect_field_energy']/b['defect_field_energy'])
    if save:
        image_data={k:np.abs(v[::2,::2])**2 for k,v in outputs.items()}
        # Decimate only display exports; all metrics use the original calculation.
        np.savez_compressed(OUT/'coherent_images.npz',x=x[::2],regular=regular[::2,::2],
                            defect=defect[::2,::2],**image_data)
        spectrum=np.abs(fft.fftshift(spectra['defect']))**2
        fpcoord=fft.fftshift(freq)*WL*F
        sel=np.abs(fpcoord)<10.5
        np.savez_compressed(OUT/'coherent_spectrum.npz',x=fpcoord[sel],
                            before=spectrum[np.ix_(sel,sel)],
                            after=(spectrum*fft.fftshift(mask*pupil))[np.ix_(sel,sel)])
    return summary

def plots():
    d=np.load(OUT/'coherent_images.npz');x=d['x'];n=len(x)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    # Rows are object and actual-Zemax-pupil outputs, not screenshots of POP.
    for ax,key,title in zip(axes.flat,['regular','regular_open','regular_rods','defect','defect_open','defect_rods'],
                            ['Regular amplitude target','Unfiltered intensity','Filtered intensity',
                             'Target with shifted bar','Unfiltered intensity','Filtered intensity']):
        im=d[key]
        vmax=1. if key in ('regular','defect') else (1.3 if key.endswith('open') else .25)
        ax.imshow(im,extent=[x[0]*10,x[-1]*10,x[0]*10,x[-1]*10],origin='lower',cmap='inferno',vmin=0,vmax=vmax)
        ax.set(xlim=(-.20,.20),ylim=(-.40,.40),title=title,xlabel='Nominal image x (mm)',ylabel='Nominal image y (mm)')
    fig.suptitle('Real Zemax pupil + Python coherent propagation | 10x image, central crop\nColor scale: targets 0-1; open images 0-1.3; filtered images 0-0.25')
    fig.savefig(OUT/'real_filter_comparison.png',dpi=180);plt.close(fig)
    s=np.load(OUT/'coherent_spectrum.npz');xf=s['x'];base=s['before'].max()
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key,title in zip(axes,['before','after'],['Before rods','After 1 mm rods']):
        z=np.log10(np.maximum(s[key]/base,1e-8))
        im=ax.imshow(z,origin='lower',extent=[xf[0],xf[-1],xf[0],xf[-1]],vmin=-8,vmax=0,cmap='inferno')
        ax.set(title=title,xlabel='Nominal Fourier x (mm)',ylabel='Nominal Fourier y (mm)',ylim=(-1,1))
        ax.set_aspect('auto')
        fig.colorbar(im,ax=ax,label='log10 intensity / common pre-mask peak')
    fig.suptitle('Scalar spectrum of finite periodic target + defect\nCoordinates use nominal f=20 mm; mask evaluation uses measured Zemax mapping')
    fig.savefig(OUT/'real_fourier_spectrum.png',dpi=180);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=2048)
    p.add_argument('--window',type=float,default=1.024);p.add_argument('--plots',action='store_true')
    p.add_argument('--rod-width',type=float,default=1.)
    p.add_argument('--apodization',action='store_true')
    p.add_argument('--no-save',action='store_true')
    a=p.parse_args()
    r=model(a.grid,a.window,a.rod_width,a.apodization,save=not a.no_save)
    name='coherent_metrics_%d_%g_w%g_a%d.json'%(a.grid,a.window,a.rod_width,a.apodization)
    (OUT/name).write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
    if a.plots:plots()

if __name__=='__main__':main()
