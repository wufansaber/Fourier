"""Dimensioned 2-D drawings for the independently movable serrated rods."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint_adjustable/rod_2d'
W=.100; B=.050; P=.750; TIP=.020; T=.050
STATIONS=[31.,41.,51.,61.,71.]


def profile(x):
    q=np.abs(np.remainder(x+P/2,P)-P/2)/(P/2)
    q=np.clip(q,TIP/P,1-TIP/P)
    return W/2+B*q


def polygon(length,center=0.,samples_per_pitch=20):
    n=max(401,int(length/P*samples_per_pitch)+1)
    x=np.linspace(-length/2,length/2,n);h=profile(x)
    return np.r_[x,x[::-1]],np.r_[center+h,(center-h)[::-1]]


def dimension(ax,a,b,y,text,offset=.08):
    ax.annotate('',(a,y),(b,y),arrowprops=dict(arrowstyle='<->',lw=.8,color='#333'))
    ax.plot([a,a],[y-offset,y+offset],color='#333',lw=.7);ax.plot([b,b],[y-offset,y+offset],color='#333',lw=.7)
    ax.text((a+b)/2,y+offset*1.15,text,ha='center',va='bottom',fontsize=8)


def part_drawing():
    fig=plt.figure(figsize=(14,8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[.9,1.35])
    ax=fig.add_subplot(gs[0,:]);x,y=polygon(62)
    ax.fill(x,y,color='#273746');ax.axvspan(-25,25,color='#2e86c1',alpha=.08,label='50 mm mechanical clear opening')
    ax.axvspan(-18,18,color='#17a589',alpha=.10,label='36 mm optical pupil')
    ax.scatter([-31,31],[0,0],s=28,c='#c0392b',zorder=4,label='paired support/slider center')
    dimension(ax,-31,31,.35,'R1 support span L1 = 62 mm',.035)
    dimension(ax,-25,25,-.36,'mechanical opening = 50 mm',.035)
    ax.set(xlim=(-34,34),ylim=(-.55,.55),xlabel='X along rod (mm)',ylabel='Y (mm)',title='A. Full part plan view (R1 shortest member; serration continues across span)')
    ax.legend(ncol=3,fontsize=8,loc='lower center');ax.grid(axis='x',alpha=.15)

    ax=fig.add_subplot(gs[1,0]);x=np.linspace(-.85,.85,1201);h=profile(x)
    ax.fill_between(x,-h,h,color='#273746');ax.axhline(W/2,color='#c0392b',ls=':',lw=.8);ax.axhline(-W/2,color='#c0392b',ls=':',lw=.8)
    dimension(ax,-.375,.375,.15,'tooth pitch P = 0.750 mm',.012)
    ax.annotate('one-side depth\nb = 0.050 mm',xy=(0,h[np.argmin(abs(x))]),xytext=(.28,.115),ha='center',fontsize=8,arrowprops=dict(arrowstyle='->',lw=.8))
    ax.annotate('flat tip/valley\n>= 0.020 mm',xy=(.375,profile(np.array([.375]))[0]),xytext=(.62,-.12),ha='center',fontsize=8,arrowprops=dict(arrowstyle='->',lw=.8))
    ax.annotate('',(-.73,-W/2),(-.73,W/2),arrowprops=dict(arrowstyle='<->',lw=.8));ax.text(-.70,0,'core w = 0.100 mm',rotation=90,va='center',fontsize=8)
    ax.set(xlim=(-.85,.85),ylim=(-.18,.18),aspect='equal',xlabel='X (mm)',ylabel='Y (mm)',title='B. Serrated edge detail (nominal)');ax.grid(alpha=.15)

    ax=fig.add_subplot(gs[1,1]);scale=5
    ax.add_patch(Rectangle((-.2,-T/2),.4,T,facecolor='#273746'))
    ax.annotate('',(.26,-T/2),(.26,T/2),arrowprops=dict(arrowstyle='<->',lw=.8));ax.text(.29,0,'foil t = 0.050 mm',va='center',fontsize=8)
    ax.plot([-.8,.8],[-2,-2],color='#236e95',lw=4,label='Layer A, z = -2 mm');ax.plot([-.8,.8],[2,2],color='#b38226',lw=4,label='Layer B, z = +2 mm')
    ax.annotate('',(.65,-2),(.65,2),arrowprops=dict(arrowstyle='<->',lw=.8));ax.text(.69,0,'layer gap = 4.0 mm',rotation=90,va='center',fontsize=8)
    ax.set(xlim=(-1,1.15),ylim=(-2.8,2.8),xlabel='local X (schematic)',ylabel='Z (mm)',title='C. Foil thickness and two-deck spacing');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.12)
    fig.suptitle('Independently movable full-serrated rod | nominal dimensions, not released fabrication drawing',fontsize=14)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'serrated_rod_dimensioned.{ext}',dpi=220 if ext=='png' else None)
    plt.close(fig)


def assembly_drawing(data):
    selected=next(x['selected'] for x in data['presets'] if x['period_um']==20 and x['mode']=='balanced')
    d=selected['design'];fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    ax=axs[0];ax.add_patch(Circle((0,0),25,fill=False,ls='--',color='.4',label='50 mm opening'))
    ax.add_patch(Circle((0,0),18,fill=False,color='#222',label='36 mm optical pupil'))
    colors=['#236e95','#b38226']
    for i,(cy,layer,on) in enumerate(zip(d['centres_mm'],d['layers'],d['active_flags'])):
        L=2*STATIONS[i//2];x,y=polygon(L,cy,8)
        ax.fill(x,y,color=colors[layer],alpha=.90 if on else .32,lw=0)
        for sign in [-1,1]:ax.add_patch(Rectangle((sign*STATIONS[i//2]-4,cy-4),8,8,facecolor=colors[layer],alpha=.15,edgecolor=colors[layer],lw=.7))
        station=STATIONS[i//2]
        if on:
            label_y=8 if layer==0 else -8
            ax.plot([station,station],[cy,label_y],color=colors[layer],lw=.45,alpha=.7)
            ax.text(station,label_y,f'R{i+1}\nY={cy:+.4f}',fontsize=6.3,ha='center',
                    va='bottom' if label_y>0 else 'top',rotation=90,color=colors[layer])
        else:
            label_y=24 if cy>0 else -24
            ax.plot([station,station],[cy,label_y],color=colors[layer],lw=.45,alpha=.7)
            ax.text(station,label_y,f'R{i+1} park\nY={cy:+.4f}',fontsize=6.3,ha='center',
                    va='bottom' if label_y>0 else 'top',rotation=90,color=colors[layer])
    ax.set(xlim=(-79,83),ylim=(-31,31),aspect='equal',xlabel='X (mm)',ylabel='Y (mm)',title='A. 20 um-period balanced recipe / independent paired-end Y slides');ax.legend(fontsize=8,loc='lower left')
    ax=axs[1]
    for layer,z,color in [(0,-2,colors[0]),(1,2,colors[1])]:
        ids=[i for i,k in enumerate(d['layers']) if k==layer]
        ax.plot([-25,25],[z,z],color=color,lw=2,label=f'Layer {"AB"[layer]}: '+', '.join(f'R{i+1}' for i in ids))
        for x0 in STATIONS:
            ax.add_patch(Rectangle((x0-4,(-6 if layer==0 else 6)-2),8,4,facecolor=color,alpha=.18,edgecolor=color))
    ax.axvspan(-25,25,color='.6',alpha=.08);ax.axvline(25,color='.3',ls='--');ax.text(24.2,0,'opening edge',rotation=90,ha='right',va='center',fontsize=8)
    ax.set(xlim=(-28,77),ylim=(-10,10),xlabel='X (mm)',ylabel='Z (mm)',title='B. Side-view allocation / slider bodies outside aperture');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('9 installed rods: 5 on A + 4 on B | each rod has its own Y axis and paired end supports',fontsize=14)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'nine_rod_adjustable_assembly.{ext}',dpi=220 if ext=='png' else None)
    plt.close(fig)


def dxf_polyline(lines,layer,x,y):
    lines.extend(['0','LWPOLYLINE','8',layer,'90',str(len(x)),'70','1'])
    for a,b in zip(x,y):lines.extend(['10',f'{a:.7f}','20',f'{b:.7f}'])


def cad_exports(data):
    lines=['0','SECTION','2','HEADER','9','$INSUNITS','70','4','0','ENDSEC','0','SECTION','2','ENTITIES']
    rows=[]
    for i,station in enumerate(STATIONS,1):
        L=2*station;x,y=polygon(L,0,24);dxf_polyline(lines,f'L{int(L)}MM',x,y)
        rows.append(dict(part_variant=f'L{i}',support_span_mm=L,nominal_core_mm=W,one_side_depth_mm=B,
            ideal_envelope_mm=W+2*B,tooth_pitch_mm=P,minimum_flat_mm=TIP,foil_thickness_mm=T,
            quantity=2 if i<5 else 1,release_status='concept / DFM required'))
    lines+=['0','ENDSEC','0','EOF'];(OUT/'serrated_rod_length_variants.dxf').write_text('\n'.join(lines)+'\n',encoding='ascii')
    with (OUT/'serrated_rod_dimensions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Exact current 20-um recipe at physical filter-plane coordinates.
    d=next(x['selected']['design'] for x in data['presets'] if x['period_um']==20 and x['mode']=='balanced')
    lines=['0','SECTION','2','HEADER','9','$INSUNITS','70','4','0','ENDSEC','0','SECTION','2','ENTITIES']
    for i,(cy,layer) in enumerate(zip(d['centres_mm'],d['layers'])):
        x,y=polygon(2*STATIONS[i//2],cy,20);dxf_polyline(lines,f'{"AB"[layer]}_R{i+1}',x,y)
    for radius,layer in [(18,'GUIDE_OPTICAL_PUPIL_NOT_PART'),(25,'GUIDE_OPENING_NOT_PART')]:
        lines+=['0','CIRCLE','8',layer,'10','0','20','0','30','0','40',str(radius)]
    lines+=['0','ENDSEC','0','EOF'];(OUT/'nine_rod_20um_balanced_layout.dxf').write_text('\n'.join(lines)+'\n',encoding='ascii')


def note():
    text='''# 独立可调锯齿挡杆二维结构图\n\n本图对应联合优化报告的推荐硬件：9根装机、A/B双层5+4，每根挡杆独立Y向移动且两端同步支承。零件名义截面为核心全宽0.100 mm、单侧齿深0.050 mm、理论最大包络0.200 mm、三角齿周期0.750 mm、平顶/谷不小于0.020 mm、薄片厚度0.050 mm。\n\n![挡杆零件尺寸图](results/joint_adjustable/rod_2d/serrated_rod_dimensioned.png)\n\n![九杆独立调节装配图](results/joint_adjustable/rod_2d/nine_rod_adjustable_assembly.png)\n\n## 输出文件\n\n- [零件尺寸矢量图](results/joint_adjustable/rod_2d/serrated_rod_dimensioned.svg)；[PDF](results/joint_adjustable/rod_2d/serrated_rod_dimensioned.pdf)\n- [九杆机构矢量图](results/joint_adjustable/rod_2d/nine_rod_adjustable_assembly.svg)；[PDF](results/joint_adjustable/rod_2d/nine_rod_adjustable_assembly.pdf)\n- [5种长度挡杆DXF](results/joint_adjustable/rod_2d/serrated_rod_length_variants.dxf)：支承跨距62、82、102、122、142 mm，对应数量2、2、2、2、1。\n- [20 µm折中配方九杆DXF](results/joint_adjustable/rod_2d/nine_rod_20um_balanced_layout.dxf)\n- [挡杆尺寸和数量CSV](results/joint_adjustable/rod_2d/serrated_rod_dimensions.csv)\n\n## 使用边界\n\nDXF单位为mm。圆形光瞳和机械开口图层只是辅助线，不是加工轮廓。当前图是与光学、制造和机械包络模型一致的二维概念图，不是放行图；夹持孔、张紧端、实际电机/导轨、圆角补偿、公差基准、材料牌号及表面处理仍需完成DFM与正式机械设计。不同周期只改变每根杆的Y坐标，不改变上述齿形尺寸。\n'''
    (ROOT/'挡杆二维结构图.md').write_text(text,encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    data=json.loads((ROOT/'results/joint_adjustable/final_presets.json').read_text(encoding='utf-8'))
    part_drawing();assembly_drawing(data);cad_exports(data);note()
    print(f'2-D rod drawings written to {OUT}')


if __name__=='__main__':main()
