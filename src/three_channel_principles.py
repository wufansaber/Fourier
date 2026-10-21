"""Exact direction-cosine coverage for Ch1/Ch2/Ch3 collection."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/three_channel_principles'
LAM_MM=.000355; NA=.6; F_MM=30.; PHI=np.deg2rad(45); INC=np.deg2rad(75); P_MM=.020
COLORS={'Ch1':'#202124','Ch3':'#167a5b','Ch2':'#3156a3'}


def axes():
    # Local e1 is in the on-screen radial plane; e2 points out of screen.
    return {
      'Ch1':(np.array([np.sin(PHI),0,np.cos(PHI)]),np.array([np.cos(PHI),0,-np.sin(PHI)]),np.array([0,1,0])),
      'Ch3':(np.array([0.,0,1.]),np.array([1.,0,0]),np.array([0,1,0])),
      'Ch2':(np.array([-np.sin(PHI),0,np.cos(PHI)]),np.array([np.cos(PHI),0,np.sin(PHI)]),np.array([0,1,0]))}


def boundary(channel,incident,n=1000):
    c,e1,e2=axes()[channel];t=np.linspace(0,2*np.pi,n)
    u=NA*np.cos(t);v=NA*np.sin(t)
    d=np.sqrt(1-u*u-v*v)[:,None]*c+u[:,None]*e1+v[:,None]*e2
    return (d[:,:2]-incident[:2])/LAM_MM


def accepted_points(incident):
    orders=np.arange(-120,121);m,n=np.meshgrid(orders,orders,indexing='xy')
    q=np.stack([m.ravel()/P_MM,n.ravel()/P_MM],axis=1)
    tang=incident[:2]+LAM_MM*q;valid=np.sum(tang*tang,axis=1)<=1
    d=np.c_[tang,np.sqrt(np.maximum(0,1-np.sum(tang*tang,axis=1)))]
    dots=np.column_stack([d@axes()[k][0] for k in ['Ch1','Ch3','Ch2']])
    valid &= np.max(dots,axis=1)>=np.sqrt(1-NA**2)
    return q[valid]


def plot():
    incidents=[('Normal illumination',np.array([0.,0.,-1.])),
               ('75 deg illumination, projection out of screen',np.array([0.,np.sin(INC),-np.cos(INC)]))]
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,(title,inc) in zip(axs,incidents):
        pts=accepted_points(inc);ax.scatter(pts[:,0],pts[:,1],s=2,c='#b7bcc2',alpha=.35,label='accepted 20 um grid orders')
        for ch in ['Ch1','Ch3','Ch2']:
            c=axes()[ch][0];center=(c[:2]-inc[:2])/LAM_MM;b=boundary(ch,inc)
            ax.plot(b[:,0],b[:,1],color=COLORS[ch],lw=1.7,label=f'{ch} NA boundary')
            ax.scatter(*center,s=38,color=COLORS[ch],zorder=4)
            ax.text(center[0],center[1]+120,ch,ha='center',color=COLORS[ch],fontsize=10,weight='bold')
        ax.axhline(0,color='.75',lw=.7);ax.axvline(0,color='.75',lw=.7)
        ax.set(aspect='equal',xlabel='object spatial frequency qx (cycles/mm)',ylabel='object spatial frequency qy (cycles/mm)',title=title)
        ax.grid(alpha=.12)
    axs[0].set(xlim=(-3700,3700),ylim=(-2000,2000));axs[1].set(xlim=(-3700,3700),ylim=(-4700,-700))
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',fontsize=8)
    fig.suptitle('Three collection channels in object-frequency space | 355 nm, NA 0.6, axes at 0 and +/-45 deg',fontsize=13)
    for ext in ['png','svg']:fig.savefig(OUT/f'three_channel_frequency_coverage.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)


def tilt_translation_by_channel():
    """Show the same incidence-induced frequency shift for each receiver separately."""
    normal=np.array([0.,0.,-1.])
    oblique=np.array([0.,np.sin(INC),-np.cos(INC)])
    shift=np.array([0.,-np.sin(INC)/LAM_MM])
    fig,axs=plt.subplots(1,3,figsize=(16,5.8),layout='constrained')
    for ax,ch in zip(axs,['Ch1','Ch3','Ch2']):
        c=axes()[ch][0]
        qn=(c[:2]-normal[:2])/LAM_MM
        qo=(c[:2]-oblique[:2])/LAM_MM
        bn=boundary(ch,normal);bo=boundary(ch,oblique)
        ax.plot(bn[:,0],bn[:,1],color='#72777d',lw=1.5,label='normal incidence support')
        ax.plot(bo[:,0],bo[:,1],color='#9b2853',lw=2.0,label='75 deg incidence support')
        ax.scatter(*qn,s=48,color='#3c4043',zorder=4)
        ax.scatter(*qo,s=48,color='#9b2853',zorder=4)
        ax.annotate('',xy=qo,xytext=qn,arrowprops=dict(arrowstyle='-|>',lw=2,color='#a12a55'))
        ax.text((qn[0]+qo[0])/2+80,(qn[1]+qo[1])/2,
                f'$\\Delta q_y$ = {shift[1]:.1f}\ncycles/mm',color='#8c244b',fontsize=9)
        ax.text(qn[0]+80,qn[1]+100,'normal center',fontsize=8,color='#3c4043')
        ax.text(qo[0]+80,qo[1]-650,'75 deg center',fontsize=8,color='#9b2853')
        # Plot only the local 20 um reciprocal-lattice neighbourhood; no other receiver is drawn.
        mc=int(round(qo[0]*P_MM));nc=int(round(qo[1]*P_MM))
        mm,nn=np.meshgrid(np.arange(mc-7,mc+8),np.arange(nc-7,nc+8),indexing='xy')
        q=np.c_[mm.ravel()/P_MM,nn.ravel()/P_MM]
        mask=channel_accepts(q,oblique,ch)
        ax.scatter(q[mask,0],q[mask,1],s=12,facecolors='none',edgecolors=COLORS[ch],linewidths=.6,
                   label='accepted 20 um orders')
        ax.set(xlim=(qn[0]-2100,qn[0]+2100),ylim=(-4700,1900),aspect='equal',
               xlabel='$q_x$ (cycles/mm)',ylabel='$q_y$ (cycles/mm)',
               title=f'{ch} calculated independently')
        ax.grid(alpha=.12);ax.legend(fontsize=7,loc='lower right')
    fig.suptitle('75 deg illumination translates every receiver support by the same object-frequency vector\n'
                 r'$\Delta\mathbf{q}=(0,-\sin75^\circ/\lambda)=(0,-2720.9)$ cycles/mm; no channel-overlap assumption',fontsize=13)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'tilt_spectrum_translation_by_channel.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)


def arrow3d(ax,start,end,color,label,lw=2.2,ls='-'):
    start=np.asarray(start);vec=np.asarray(end)-start
    ax.quiver(*start,*vec,color=color,linewidth=lw,arrow_length_ratio=.10,linestyle=ls)
    ax.text(*(np.asarray(end)*1.08),label,color=color,fontsize=9,ha='center')


def wavevector_geometry():
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig=plt.figure(figsize=(10,8),layout='constrained');ax=fig.add_subplot(111,projection='3d')
    u=np.linspace(0,2*np.pi,80);v=np.linspace(0,np.pi,40)
    x=np.outer(np.cos(u),np.sin(v));y=np.outer(np.sin(u),np.sin(v));z=np.outer(np.ones_like(u),np.cos(v))
    ax.plot_wireframe(x,y,z,rstride=6,cstride=8,color='#a9adb2',alpha=.20,linewidth=.45)
    # Physical source is above the wafer at -KY; propagation points toward the origin.
    rsrc=np.array([0.,-np.sin(INC),np.cos(INC)])
    ki=np.array([0.,np.sin(INC),-np.cos(INC)])
    k0=np.array([0.,np.sin(INC),np.cos(INC)])
    arrow3d(ax,rsrc,[0,0,0],'#972b55','physical incident ray',2.5)
    arrow3d(ax,[0,0,0],ki,'#d24a77',r'$\mathbf{k}_i/k$',1.8,'--')
    arrow3d(ax,[0,0,0],k0,'#e26d19',r'specular $\mathbf{k}_0/k$',2.2)
    for ch in ['Ch1','Ch3','Ch2']:arrow3d(ax,[0,0,0],axes()[ch][0],COLORS[ch],ch,2.8)
    ax.scatter(*rsrc,s=42,c='#972b55');ax.text(*(rsrc+np.array([0,-.08,.08])),'source location\n(-KY side)',color='#972b55',ha='center',fontsize=9)
    ax.plot([-1.05,1.05],[0,0],[0,0],color='.5',lw=.8);ax.plot([0,0],[-1.05,1.05],[0,0],color='.5',lw=.8)
    ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),zlim=(-1.05,1.12),xlabel='$K_X/k$',ylabel='$K_Y/k$',zlabel='$K_Z/k$')
    ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-55)
    ax.set_title('Direction convention: source location is -KY, incident propagation is +KY and -KZ\n75 deg from sample normal; Ch1 is +KX and Ch2 is -KX')
    for ext in ['png','svg']:fig.savefig(OUT/f'wavevector_direction_convention.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)


def channel_accepts(q,incident,ch):
    tang=incident[:2]+LAM_MM*np.asarray(q)
    good=np.sum(tang*tang,axis=-1)<=1
    z=np.sqrt(np.maximum(0,1-np.sum(tang*tang,axis=-1)))
    d=np.concatenate([tang,np.expand_dims(z,-1)],axis=-1)
    return good & ((d@axes()[ch][0])>=np.sqrt(1-NA**2))


def local_bfp_xy(q,incident,ch):
    """Exact sine-coordinate intercepts in one receiver's local Fourier plane."""
    q=np.atleast_2d(np.asarray(q,dtype=float))
    tang=incident[:2]+LAM_MM*q
    valid=np.sum(tang*tang,axis=1)<=1
    z=np.sqrt(np.maximum(0,1-np.sum(tang*tang,axis=1)))
    d=np.c_[tang,z]
    _,e1,e2=axes()[ch]
    xy=F_MM*np.c_[d@e1,d@e2]
    return xy,valid


def oblique_local_peak_spacing():
    """Plot exact 75-degree peak lattices in each receiver's own BFP coordinates."""
    inc=np.array([0.,np.sin(INC),-np.cos(INC)])
    centres={'Ch1':(40,-54),'Ch3':(0,-54),'Ch2':(-40,-54)}
    rows=[]
    fig,axs=plt.subplots(1,3,figsize=(16,5.8),layout='constrained')
    for ax,ch in zip(axs,['Ch1','Ch3','Ch2']):
        mc,nc=centres[ch]
        ms=np.arange(mc-5,mc+6);ns=np.arange(nc-5,nc+6)
        mm,nn=np.meshgrid(ms,ns,indexing='xy');orders=np.c_[mm.ravel(),nn.ravel()]
        q=orders/P_MM;xy,prop=local_bfp_xy(q,inc,ch)
        accept=prop & channel_accepts(q,inc,ch)
        ax.scatter(xy[accept,0],xy[accept,1],s=22,facecolors='white',edgecolors=COLORS[ch],linewidths=.8)
        # Highlight the central row and column so both adjacent spacings are visible.
        rowmask=accept & (orders[:,1]==nc);colmask=accept & (orders[:,0]==mc)
        ax.plot(xy[rowmask,0],xy[rowmask,1],color=COLORS[ch],lw=.8,alpha=.55)
        ax.plot(xy[colmask,0],xy[colmask,1],color=COLORS[ch],lw=.8,alpha=.55)
        q0=np.array([[mc/P_MM,nc/P_MM]])
        x0=local_bfp_xy(q0,inc,ch)[0][0]
        xm=local_bfp_xy(np.array([[(mc-1)/P_MM,nc/P_MM]]),inc,ch)[0][0]
        xp=local_bfp_xy(np.array([[(mc+1)/P_MM,nc/P_MM]]),inc,ch)[0][0]
        yp=local_bfp_xy(np.array([[mc/P_MM,(nc+1)/P_MM]]),inc,ch)[0][0]
        dxm=x0[0]-xm[0];dxp=xp[0]-x0[0];dy=yp[1]-x0[1]
        ax.scatter(*x0,s=65,color='#9b2853',zorder=5)
        ax.annotate('',xy=(xp[0],x0[1]),xytext=(x0[0],x0[1]),arrowprops=dict(arrowstyle='<->',color='#9b2853',lw=1.4))
        ax.annotate('',xy=(x0[0],yp[1]),xytext=(x0[0],x0[1]),arrowprops=dict(arrowstyle='<->',color='#9b2853',lw=1.4))
        ax.text(x0[0]+.18,x0[1]+.15,f'central order\n(m,n)=({mc},{nc})',fontsize=8,color='#7d2143')
        ax.text(.03,.03,f'$\\Delta X_-$={dxm:.4f} mm\n$\\Delta X_+$={dxp:.4f} mm\n$\\Delta Y$={dy:.4f} mm\ncenter=({x0[0]:+.4f}, {x0[1]:+.4f}) mm',
                transform=ax.transAxes,fontsize=8,bbox=dict(boxstyle='round',fc='white',ec='.75',alpha=.9))
        ax.axhline(0,color='.65',lw=.7);ax.axvline(0,color='.65',lw=.7)
        ax.set(aspect='equal',xlim=(-4,4),ylim=(-3.5,3.8),xlabel='local $X_F$ (mm)',ylabel='local $Y_F$ (mm)',
               title=f'{ch}: exact 75 deg local BFP')
        ax.grid(alpha=.15)
        rows.append(dict(channel=ch,central_m=mc,central_n=nc,central_XF_mm=x0[0],central_YF_mm=x0[1],
                         delta_X_minus_mm=dxm,delta_X_plus_mm=dxp,delta_Y_mm=dy,
                         nominal_f_lambda_over_p_mm=F_MM*LAM_MM/P_MM))
    fig.suptitle('20 um 2-D pattern peaks after 75 deg illumination | each receiver calculated in its own sine-coordinate BFP\n'
                 'f=30 mm, wavelength=355 nm, NA=0.6; zoom around the local optical axis (full pupil radius = 18 mm)',fontsize=13)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'oblique75_local_bfp_peak_spacing.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)
    with (OUT/'oblique75_local_bfp_peak_spacing.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return rows


def pattern_cases():
    orders=np.arange(-120,121)
    cases=[('1-D qx comb\n(no KY reciprocal component)',np.c_[orders/P_MM,np.zeros_like(orders)]),
           ('1-D qy comb\n(can cancel incident +KY)',np.c_[np.zeros_like(orders),orders/P_MM])]
    m,n=np.meshgrid(orders,orders,indexing='xy');cases.append(('2-D reciprocal lattice\n(can reach all three channels)',np.c_[m.ravel()/P_MM,n.ravel()/P_MM]))
    illumination=[('Normal illumination',np.array([0.,0.,-1.]),(-2000,2000)),
                  ('75 deg illumination from source on -KY side',np.array([0.,np.sin(INC),-np.cos(INC)]),(-4700,300))]
    fig,axs=plt.subplots(2,3,figsize=(16,10),layout='constrained')
    for row,(rowtitle,inc,ylim) in enumerate(illumination):
      for col,(title,q) in enumerate(cases):
        ax=axs[row,col];keep=(abs(q[:,0])<=3800)&(q[:,1]>=ylim[0])&(q[:,1]<=ylim[1]);ax.scatter(q[keep,0],q[keep,1],s=3,c='#c3c6c9',alpha=.3)
        for ch in ['Ch1','Ch3','Ch2']:
            b=boundary(ch,inc);mask=channel_accepts(q,inc,ch)
            ax.plot(b[:,0],b[:,1],color=COLORS[ch],lw=1.3)
            ax.scatter(q[mask,0],q[mask,1],s=9,facecolors='none',edgecolors=COLORS[ch],linewidths=.55,label=ch)
        ax.set(xlim=(-3700,3700),ylim=ylim,aspect='equal',xlabel='$q_x$ (cycles/mm)',ylabel='$q_y$ (cycles/mm)',title=rowtitle+'\n'+title);ax.grid(alpha=.12)
    fig.suptitle('Normal versus 75 deg illumination | 20 um pattern orders entering Ch1 / Ch2 / Ch3',fontsize=13)
    handles,labels=axs[-1,-1].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='lower center')
    for ext in ['png','svg']:fig.savefig(OUT/f'normal_and_oblique_pattern_entry.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)


def bfp_spacing():
    # Panel 1 uses a common sample-normal sine coordinate. It displays absolute
    # orders and therefore must not put the side-channel spectral centres at zero.
    fig,axs=plt.subplots(2,1,figsize=(13,7.2),layout='constrained')
    ax=axs[0];step=F_MM*LAM_MM/P_MM
    groups=[('Ch2',-40,-F_MM*np.sin(PHI),.065),('Ch3',0,0.,.065),('Ch1',40,F_MM*np.sin(PHI),.065)]
    for ch,mc,axis_x,label_y in groups:
        orders=np.arange(mc-4,mc+5);x=orders*step
        ax.scatter(x,np.zeros_like(x,dtype=float),s=38,color=COLORS[ch],zorder=3,label=f'{ch} absolute orders')
        for xx,k in zip(x,orders):ax.text(xx,label_y,str(k),ha='center',fontsize=7,color=COLORS[ch])
        ax.axvline(axis_x,color=COLORS[ch],lw=1.1,ls='--')
        ax.axvspan(axis_x-1,axis_x+1,color=COLORS[ch],alpha=.08)
        ax.text(axis_x,-.085,f'axis {axis_x:+.3f} mm',ha='center',fontsize=7,color=COLORS[ch])
    ax.axhline(0,color='.5',lw=.7)
    ax.set(xlim=(-25,25),ylim=(-.18,.18),yticks=[0],yticklabels=['$Y_G=0$'],xlabel=r'common sample-normal sine coordinate $X_G=f s_x$ (mm)',
           title='Normal illumination: all X-diffraction peaks share Y_G=0; only their absolute X order changes')
    ax.grid(axis='x',alpha=.15);ax.legend(fontsize=7,ncol=3,loc='lower center')

    # Panel 2 shows the orthogonal 75-degree translation and the compensating integer orders.
    ax=axs[1]
    n=np.arange(-58,-50);y=F_MM*(np.sin(INC)+n*LAM_MM/P_MM)
    ax.scatter(y,np.ones_like(y),s=42,color='#9b2853',zorder=3,label='75 deg illumination, absolute n orders')
    for yy,k in zip(y,n):ax.text(yy,1.08,str(k),ha='center',fontsize=7,color='#7d2143')
    ax.scatter([0,F_MM*np.sin(INC)],[.72,.72],s=50,c=['#3c4043','#d66a1f'],zorder=4)
    ax.annotate('',xy=(F_MM*np.sin(INC),.72),xytext=(0,.72),arrowprops=dict(arrowstyle='<->',lw=1.4,color='#a12a55'))
    ax.text(F_MM*np.sin(INC)/2,.62,f'fixed-order shift = f sin75 deg = {F_MM*np.sin(INC):.3f} mm',ha='center',fontsize=8)
    ax.axvspan(-1,1,color='#b64040',alpha=.16,label='2.0 mm rigid-rod envelope near compensated centre')
    ax.axvline(0,color='.45',lw=.8)
    ax.set(xlim=(-3,32),ylim=(.48,1.22),yticks=[],xlabel=r'common out-of-screen sine coordinate $Y_G=f s_y$ (mm)',
           title='75 deg illumination: n=0 shifts to +28.978 mm; n=-54 returns to +0.223 mm')
    ax.grid(axis='x',alpha=.15);ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Global absolute diffraction-order positions | 20 um period, f=30 mm, 355 nm\n'
                 'Shaded bands are 2 mm rod envelopes; local-objective zero is a different coordinate convention',fontsize=13)
    for ext in ['png','svg']:fig.savefig(OUT/f'local_bfp_spacing_vs_rigid_rod.{ext}',dpi=210 if ext=='png' else None)
    plt.close(fig)


def table():
    rows=[]
    for case,inc in [('normal',np.array([0.,0.,-1.])),('oblique75_out_of_screen',np.array([0.,np.sin(INC),-np.cos(INC)]))]:
        for ch in ['Ch1','Ch3','Ch2']:
            c=axes()[ch][0];q=(c[:2]-inc[:2])/LAM_MM
            rows.append(dict(case=case,channel=ch,qx_cycles_per_mm=q[0],qy_cycles_per_mm=q[1],
              nearest_m_for_20um=round(q[0]*P_MM),nearest_n_for_20um=round(q[1]*P_MM),
              local_BFP_step_screen_mm=F_MM*LAM_MM/P_MM/(np.cos(PHI) if ch!='Ch3' else 1),
              local_BFP_step_out_of_screen_mm=F_MM*LAM_MM/P_MM))
    with (OUT/'channel_centres_and_spacing.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return rows


def translation_table():
    step=LAM_MM/P_MM
    rows=[]
    normal=np.array([0.,0.,-1.]);oblique=np.array([0.,np.sin(INC),-np.cos(INC)])
    for ch in ['Ch1','Ch3','Ch2']:
        c=axes()[ch][0]
        qn=(c[:2]-normal[:2])/LAM_MM;qo=(c[:2]-oblique[:2])/LAM_MM
        rows.append(dict(
            channel=ch,
            normal_qx_cycles_per_mm=qn[0],normal_qy_cycles_per_mm=qn[1],
            tilt75_qx_cycles_per_mm=qo[0],tilt75_qy_cycles_per_mm=qo[1],
            delta_qx_cycles_per_mm=qo[0]-qn[0],delta_qy_cycles_per_mm=qo[1]-qn[1],
            normal_m_20um=qn[0]*P_MM,normal_n_20um=qn[1]*P_MM,
            tilt75_m_20um=qo[0]*P_MM,tilt75_n_20um=qo[1]*P_MM,
            fixed_order_BFP_shift_mm=F_MM*np.sin(INC),
            nearest_cancel_n=round(-np.sin(INC)/step),
            nearest_cancel_residual_BFP_y_mm=F_MM*(np.sin(INC)+round(-np.sin(INC)/step)*step)))
    with (OUT/'tilt_translation_by_channel.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True);wavevector_geometry();plot();tilt_translation_by_channel();pattern_cases();bfp_spacing();oblique_rows=oblique_local_peak_spacing();rows=table();translations=translation_table()
    print('written',OUT)
    for r in rows:print(r)
    for r in translations:print(r)
    for r in oblique_rows:print(r)


if __name__=='__main__':main()
