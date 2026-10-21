"""Compare native POP grids on their physical image coordinates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/'results'/'real_objective'

def main():
    names=['real_regular_open','real_regular_rods','real_defect_open','real_defect_rods']
    available=[name for name in names if (OUT/'native_pop'/(name+'_s28_n2048.npz')).exists()]
    fig,axes=plt.subplots(2,2,figsize=(10,8),squeeze=False,layout='constrained')
    common={kind:max(float(np.load(OUT/'native_pop'/(name+'_s28_n2048.npz'))['irradiance'].max())
                         for name in available if name.endswith(kind)) for kind in ('open','rods')}
    summaries={}
    for ax,name in zip(axes.flat,available):
        d=np.load(OUT/'native_pop'/(name+'_s28_n2048.npz'));x=d['x'];y=d['y'];a=d['irradiance']
        roi=(x[None,:]**2+y[:,None]**2)<2.**2
        central=(np.abs(x)[None,:]<.06)&(np.abs(y)[:,None]<.25)
        peak=common['open' if name.endswith('open') else 'rods']
        im=ax.imshow(a,origin='lower',extent=[x[0],x[-1],y[0],y[-1]],cmap='inferno',vmin=0,vmax=peak)
        ax.set(xlim=(-.2,.2),ylim=(-.4,.4),xlabel='Image x (mm)',ylabel='Image y (mm)',title=name.replace('real_','').replace('_',' '))
        fig.colorbar(im,ax=ax,label='Native POP irradiance (W/mm2)',shrink=.75)
        summaries[name]=dict(power_W=float(a.sum()*(x[1]-x[0])*(y[1]-y[0])),
                             roi_power_W=float(a[roi].sum()*(x[1]-x[0])*(y[1]-y[0])),
                             central_peak_W_per_mm2=float(a[central].max()),nx=len(x),dx_um=float((x[1]-x[0])*1000))
    fig.suptitle('Native Zemax POP | real objective + real relay + 1 mm rods\n2048 x 2048; common absolute irradiance scale within each column; central image crop')
    fig.savefig(OUT/'native_pop_comparison.png',dpi=180);plt.close(fig)
    if 'real_regular_open' in summaries and 'real_regular_rods' in summaries:
        summaries['regular_suppression_db']=float(10*np.log10(summaries['real_regular_open']['roi_power_W']/summaries['real_regular_rods']['roi_power_W']))
        summaries['central_peak_ratio_defect_to_regular_filtered']=summaries['real_defect_rods']['central_peak_W_per_mm2']/summaries['real_regular_rods']['central_peak_W_per_mm2']
    fine={}
    for name in ('real_regular_open','real_regular_rods'):
        p=OUT/'native_pop'/(name+'_s28_n4096.npz')
        if p.exists():
            d=np.load(p);x=d['x'];y=d['y'];a=d['irradiance']
            roi=x[None,:]**2+y[:,None]**2<4.
            fine[name]=dict(roi_power_W=float(a[roi].sum()*(x[1]-x[0])*(y[1]-y[0])),
                            power_W=float(a.sum()*(x[1]-x[0])*(y[1]-y[0])),nx=len(x),dx_um=float((x[1]-x[0])*1000))
    if len(fine)==2:
        fine['regular_suppression_db']=float(10*np.log10(fine['real_regular_open']['roi_power_W']/fine['real_regular_rods']['roi_power_W']))
        fine['change_from_2048_db']=fine['regular_suppression_db']-summaries['regular_suppression_db']
    summaries['fine_sampling']=fine
    p=OUT/'native_pop'/'real_defect_open_s25_n2048.npz'
    if p.exists():
        from scipy.signal import find_peaks
        d=np.load(p);x=d['x'];y=d['y'];a=d['irradiance'];s=a.sum(axis=0)
        peaks,_=find_peaks(s,height=s.max()*.001,distance=25)
        summaries['native_fourier_peaks_mm']=x[peaks].tolist()
        centers=json.loads((OUT/'filter_definition.json').read_text())['rod_centers_mm']
        mask=np.ones(len(x),dtype=bool)
        for c in centers:mask&=np.abs(x-c)>.5
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,z,title in zip(axes,[a,a*mask[None,:]],['Native POP before filter, S25','Same-plane mask product (derived, not a second POP run)']):
            im=ax.imshow(np.log10(np.maximum(z/a.max(),1e-8)),extent=[x[0],x[-1],y[0],y[-1]],origin='lower',cmap='inferno',vmin=-8,vmax=0)
            ax.set(xlim=(-10.5,10.5),ylim=(-1,1),xlabel='Fourier x (mm)',ylabel='Fourier y (mm)',title=title)
            ax.set_aspect('auto');fig.colorbar(im,ax=ax,label='log10 irradiance / common pre-mask peak')
        fig.suptitle('Actual real-objective Fourier-plane irradiance | 584.601 nm, 5 um object pitch')
        fig.savefig(OUT/'native_fourier_plane.png',dpi=180);plt.close(fig)
    (OUT/'native_pop'/'image_metrics.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
