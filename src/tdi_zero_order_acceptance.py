"""TDI scan-direction acceptance required to keep serration nonzero orders out.

The calculation is geometric in the detector image plane. Gaussian illumination
has infinite support, so exclusion is reported at explicit encircled-energy
fractions rather than described as mathematically zero leakage.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.special import erfinv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/tdi_zero_order_acceptance'

LAM_MM=0.000355
F_FF_MM=30.0
MAG=20.0
PITCH_MM=0.75
ROW_UM=5.0
N25=25
N256=256
ILLUM_FWHM_IMAGE_UM=125.0
LINE_PIXELS=2048
LINE_LENGTH_UM=LINE_PIXELS*ROW_UM
IMAGE_FIELD_X_UM=0.5*MAG*1000.0


def full_width_for_central_energy(fwhm_um,fraction):
    # I(x)=exp[-4 ln(2) x^2/FWHM^2].
    return fwhm_um*erfinv(fraction)/np.sqrt(np.log(2.0))


def order_shift_image_um(pitch_mm):
    return 1000.0*MAG*LAM_MM*F_FF_MM/pitch_mm


def max_tdi_height_um(delta_um,w_eff_um,margin_um=0.0):
    return 2.0*(delta_um-margin_um)-w_eff_um


def gaussian(x,center,fwhm):
    return np.exp(-4*np.log(2)*(x-center)**2/fwhm**2)


def draw_window(ax,h_um,title,delta,w_eff):
    x=np.linspace(-750,750,8000)
    for k,color in [(-2,'#9aa0a6'),(-1,'#b14b5f'),(0,'#177e65'),(1,'#b14b5f'),(2,'#9aa0a6')]:
        ax.plot(x,gaussian(x,k*delta,ILLUM_FWHM_IMAGE_UM),color=color,lw=1.4 if abs(k)<=1 else .9,
                label='0 order' if k==0 else ('+/-1 orders' if k==-1 else None))
        if abs(k)<=1:
            ax.axvspan(k*delta-w_eff/2,k*delta+w_eff/2,color=color,alpha=.08)
    ax.axvspan(-h_um/2,h_um/2,color='#3156a3',alpha=.16,label=f'TDI active height H={h_um:.1f} um')
    ax.axvline(-h_um/2,color='#3156a3',lw=.9);ax.axvline(h_um/2,color='#3156a3',lw=.9)
    ax.set(xlim=(-700,700),ylim=(0,1.08),xlabel='scan-direction image coordinate (um)',ylabel='relative envelope',title=title)
    ax.grid(alpha=.15);ax.legend(fontsize=7,loc='upper right')


def direction_audit(delta):
    fig,axs=plt.subplots(1,2,figsize=(15,6.4),layout='constrained')
    # Fourier-filter plane: camera long edge is Y. The rod is perpendicular to
    # it, so the rod axis and serration repetition are along scan X.
    ax=axs[0]
    x=np.linspace(-5,5,1800);core=.65;depth=.42;period=1.25
    tooth=depth*(2/np.pi)*np.arcsin(np.sin(2*np.pi*x/period))
    upper=core+tooth;lower=-core-tooth
    ax.fill_between(x,lower,upper,color='#30343b',alpha=.88)
    ax.plot(x,upper,color='#9b2853',lw=1);ax.plot(x,lower,color='#9b2853',lw=1)
    ax.annotate('',xy=(3.2,1.8),xytext=(-3.2,1.8),arrowprops=dict(arrowstyle='->',lw=2,color='#9b2853'))
    ax.text(0,1.98,'serration repetition / grating vector along X',ha='center',color='#8a2349',fontsize=9)
    ax.annotate('',xy=(0,-1.8),xytext=(0,1.4),arrowprops=dict(arrowstyle='->',lw=2,color='#177e65'))
    ax.text(.15,-1.65,'rod-width normal along Y',color='#116b56',fontsize=9,rotation=90,va='bottom')
    ax.set(xlim=(-5.2,5.2),ylim=(-2.2,2.2),aspect='equal',xlabel='Fourier-filter X (rod long axis / scan)',ylabel='Fourier-filter Y (rod width / camera long edge)',
           title='Confirmed geometry: rod is perpendicular to camera long edge Y')
    ax.grid(alpha=.12)

    # Image/camera plane: long line is Y, TDI charge transfer and scan are X.
    ax=axs[1]
    line_half=LINE_LENGTH_UM/2
    ax.add_patch(plt.Rectangle((-ILLUM_FWHM_IMAGE_UM/2,-line_half),ILLUM_FWHM_IMAGE_UM,LINE_LENGTH_UM,
                               facecolor='#177e65',alpha=.10,edgecolor='#177e65',lw=1.0,label='illumination line along y (125 um FWHM across x)'))
    ax.add_patch(plt.Rectangle((-N25*ROW_UM/2,-line_half),N25*ROW_UM,LINE_LENGTH_UM,
                               facecolor='#3156a3',alpha=.13,edgecolor='#3156a3',lw=1.2,label='25-row TDI active area'))
    for xxrow in np.linspace(-N25*ROW_UM/2,N25*ROW_UM/2,26):ax.plot([xxrow,xxrow],[-line_half,line_half],color='#3156a3',lw=.25,alpha=.35)
    for k in range(-2,3):
        ax.scatter(k*delta,0,s=70 if k==0 else 40,color='#177e65' if k==0 else '#9b2853',zorder=4)
        ax.text(k*delta,115,str(k),ha='center',fontsize=8,color='#177e65' if k==0 else '#8a2349')
    yy=np.linspace(-1400,1400,700);halo=np.exp(-abs(yy)/240)
    ax.fill_betweenx(yy,-75*halo,75*halo,color='#e9a03b',alpha=.28,label='continuous rod-width diffraction along y')
    ax.annotate('',xy=(-270,2100),xytext=(-270,-2100),arrowprops=dict(arrowstyle='->',lw=2,color='#202124'))
    ax.text(-330,0,'camera line / pixel array y',ha='center',va='center',rotation=90,fontsize=9)
    ax.annotate('',xy=(330,-4300),xytext=(-330,-4300),arrowprops=dict(arrowstyle='->',lw=2,color='#3156a3'))
    ax.text(0,-4500,'TDI scan and charge transfer x',ha='center',va='top',color='#3156a3',fontsize=9)
    ax.set(xlim=(-700,700),ylim=(-5400,5400),xlabel='image x scan direction (um)',ylabel='image y along TDI camera line (um)',
           title=f'Illumination line and camera long edge are both along y\nserration orders lie along scan x; first-order spacing = {delta:.0f} um')
    ax.grid(alpha=.12);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Diffraction-direction audit: camera long edge Y, TDI scan X, rod perpendicular to camera\n'
                 'Discrete tooth orders follow scan X; rod-width sinc/soft-edge response follows camera-line Y',fontsize=13)
    fig.savefig(OUT/'rod_serration_tdi_direction_audit.png',dpi=210)
    fig.savefig(OUT/'rod_serration_tdi_direction_audit.svg')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    delta=order_shift_image_um(PITCH_MM)
    fractions=[.99,.999,.9999]
    rows=[]
    for frac in fractions:
        w=full_width_for_central_energy(ILLUM_FWHM_IMAGE_UM,frac)
        for margin in [0.0,10.0,20.0]:
            h=max_tdi_height_um(delta,w,margin)
            rows.append(dict(tooth_pitch_mm=PITCH_MM,order_shift_object_um=delta/MAG,
                             order_shift_image_um=delta,illumination_fwhm_image_um=ILLUM_FWHM_IMAGE_UM,
                             central_energy_fraction=frac,effective_illumination_width_um=w,
                             one_side_margin_um=margin,max_tdi_active_height_um=h,
                             max_rows_at_5um=h/ROW_UM,passes_25_rows=h>=N25*ROW_UM,
                             passes_256_rows=h>=N256*ROW_UM))
    with (OUT/'tdi_linewidth_limits.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)

    pitch_rows=[]
    frac=.99;w99=full_width_for_central_energy(ILLUM_FWHM_IMAGE_UM,frac)
    for pitch in [.5,.625,.75,1.0,1.5]:
        d=order_shift_image_um(pitch);h=max_tdi_height_um(d,w99,0)
        pitch_rows.append(dict(tooth_pitch_mm=pitch,order_shift_image_um=d,
                               max_height_99pct_um=h,max_rows_99pct=h/ROW_UM))
    with (OUT/'tooth_pitch_linewidth_trade.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=pitch_rows[0]);wr.writeheader();wr.writerows(pitch_rows)

    w99=full_width_for_central_energy(ILLUM_FWHM_IMAGE_UM,.99)
    hmax99=max_tdi_height_um(delta,w99,0)
    fig,axs=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    draw_window(axs[0,0],N25*ROW_UM,'25 stages: +/-1 order 99% intervals remain outside',delta,w99)
    draw_window(axs[0,1],hmax99,'Geometric limit: first-order 99% interval just touches TDI edge',delta,w99)
    draw_window(axs[1,0],N256*ROW_UM,'256 stages: +/-1 and +/-2 centres enter the active height',delta,w99)
    p=np.linspace(.45,1.5,500);d=order_shift_image_um(p);h=2*d-w99
    ax=axs[1,1];ax.plot(p,h,color='#9b2853',lw=2,label='Hmax, 99% illumination interval')
    ax.axhline(N25*ROW_UM,color='#177e65',ls='--',label='25 rows = 125 um')
    ax.axhline(N256*ROW_UM,color='#3156a3',ls='--',label='256 rows = 1280 um')
    ax.scatter([PITCH_MM],[hmax99],s=55,color='#9b2853',zorder=3)
    ax.text(PITCH_MM+.025,hmax99+25,f'0.75 mm: Hmax={hmax99:.1f} um\n({hmax99/ROW_UM:.1f} rows)',fontsize=8)
    ax.set(xlim=(.45,1.5),ylim=(-50,1350),xlabel='serration pitch ps (mm)',ylabel='maximum TDI active height Hmax (um)',
           title='Smaller tooth pitch sends nonzero orders farther away')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Confirmed orientation: tooth repetition and discrete orders are parallel to TDI scan x\n'
                 '355 nm, fFF=30 mm, 20x, row pitch=5 um; Gaussian exclusion stated at a 99% energy interval',fontsize=13)
    fig.savefig(OUT/'tdi_zero_order_linewidth_condition.png',dpi=210)
    fig.savefig(OUT/'tdi_zero_order_linewidth_condition.svg')
    plt.close(fig)

    direction_audit(delta)
    summary=dict(wavelength_nm=355,ff_focal_length_mm=F_FF_MM,magnification=MAG,tooth_pitch_mm=PITCH_MM,
                 order_shift_object_um=delta/MAG,order_shift_image_um=delta,row_pitch_um=ROW_UM,
                 current_serration_order_direction='image x, parallel to TDI scan/charge transfer',
                 current_rod_width_diffraction_direction='image y, parallel to the 2048-pixel TDI camera line',
                 illumination_line_direction='image y, parallel to the TDI camera long edge',
                 illumination_line_width_direction='image x, parallel to TDI scan/charge transfer',
                 camera_line_length_um=LINE_LENGTH_UM,image_field_x_um=IMAGE_FIELD_X_UM,
                 illumination_fwhm_image_um=ILLUM_FWHM_IMAGE_UM,
                 illumination_99pct_width_um=w99,no_margin_Hmax_99pct_um=hmax99,
                 no_margin_rows_99pct=hmax99/ROW_UM,
                 max_illumination_width_for_25_rows_no_margin_um=2*delta-N25*ROW_UM,
                 note='Camera long edge is Y and scan is X. Rod is perpendicular to camera, so X-periodic teeth diffract along scan X and Hmax applies. Rod-width diffraction is continuous along line Y and is not removed by the TDI scan width.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
