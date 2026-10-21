"""Three-domain decision report, engineering envelopes, and rod recipes."""
import csv
import itertools
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import joint_adjustable as j
import ch3_linescan as m
from ideal_na06 import save_json,save_csv

FIG=j.OUT/'figures'
BLUE='#236e95';GOLD='#b38226';INK='#30343a'


def read(name):return json.loads((j.OUT/name).read_text(encoding='utf-8'))
def save(fig,name):fig.savefig(FIG/f'{name}.png',dpi=180);plt.close(fig)
def reduce_percent(db):return 100*(1-10**(-db/10))


def get_preset(data,mode,period=20):
    return next(v['selected'] for v in data['presets'] if v['mode']==mode and v['period_um']==period)


def final_mechanics(data):
    configs=[v for v in data['presets'] if v['selected']];paths=[];rods=[];mc=j.config()['mechanics']
    for item in configs:
        d=item['selected']['design']
        for index,(y,active,layer) in enumerate(zip(d['centres_mm'],d['active_flags'],d['layers'])):
            rods.append(dict(mode=item['mode'],period_um=item['period_um'],rod_id=f'R{index+1}',layer='AB'[layer],
                y_mm=y,z_mm=d['z_mm'][layer],active=active,station_x_abs_mm=mc['station_x_mm'][index//2],
                core_mm=.1,depth_mm=.05,foil_mm=.05,tooth_pitch_mm=.75))
    for a,b in itertools.combinations(configs,2):
        da=a['selected']['design'];db=b['selected']['design']
        bound=min(j.mechanical(d)['minimum_same_layer_clearance_mm'] for d in [da,db]);sampled=1e9
        for u in np.linspace(0,1,101):
            moved={**da,'centres_mm':((1-u)*np.array(da['centres_mm'])+u*np.array(db['centres_mm'])).tolist()}
            sampled=min(sampled,j.mechanical(moved)['minimum_same_layer_clearance_mm'])
        assert sampled>=bound-1e-10 and bound>=.1
        paths.append(dict(start=f'{a["mode"]}_{a["period_um"]}',end=f'{b["mode"]}_{b["period_um"]}',
            minimum_gap_mm=sampled,analytic_lower_bound_mm=bound))
    # Actual all-travel AABB test for 18 slider envelopes, not just centers.
    boxes=[]
    for index in range(9):
        layer=index%2;z=(-1 if layer==0 else 1)*(j.config()['axial_gap_mm']/2+mc['body_z_outset_mm'])
        for sign in [-1,1]:
            x=sign*mc['station_x_mm'][index//2]
            center=np.array([x,0,z]);half=np.array([mc['stage_width_x_mm']/2,mc['stroke_half_y_mm']+mc['body_length_y_mm']/2,mc['body_thickness_z_mm']/2])+mc['body_position_tolerance_mm']
            boxes.append((f'R{index+1}_{sign}',center-half,center+half))
    separations=[]
    for a,b in itertools.combinations(boxes,2):
        sep=np.maximum(b[1]-a[2],a[1]-b[2]);assert np.any(sep>0),(a[0],b[0])
        separations.append(float(max(sep)))
    result=dict(recipes=len(configs),motion_paths=len(paths),path_checks=paths,
        minimum_rod_gap_mm=min(p['minimum_gap_mm'] for p in paths),slider_swept_boxes=len(boxes),
        box_pair_checks=len(separations),collisions=0,minimum_slider_separating_gap_mm=min(separations),
        minimum_opening_clearance_mm=mc['station_x_mm'][0]-mc['stage_width_x_mm']/2-mc['body_position_tolerance_mm']-25,
        note='Full swept-Y boxes for allocated sliders; real motors/cables/fasteners not yet specified. Affine coordinated paths only, not arbitrary asynchronous moves.')
    save_json(j.OUT/'final_mechanical_checks.json',result);save_csv(j.OUT/'all_position_recipes.csv',rods)
    return result,rods


def mechanical_figure(data):
    d=get_preset(data,'balanced')['design'];mc=j.config()['mechanics']
    fig,axs=plt.subplots(1,3,figsize=(15,5.4),layout='constrained',gridspec_kw={'width_ratios':[1,1.35,1]})
    ax=axs[0]
    for radius,ls,label in [(18,'-','36 mm optical pupil'),(25,'--','50 mm clear opening')]:ax.add_patch(Circle((0,0),radius,fill=False,ls=ls,color=INK,label=label))
    parked={}
    for i,(y,k,active) in enumerate(zip(d['centres_mm'],d['layers'],d['active_flags']),1):
        ax.plot([-25,25],[y,y],color=BLUE if k==0 else GOLD,lw=1.2 if active else .8,ls='-' if active else ':')
        if not active:parked.setdefault(y,[]).append(f'R{i}')
    for y,labels in parked.items():ax.text(26,y,' / '.join(labels),fontsize=7,va='center')
    ax.set(xlim=(-28,39),ylim=(-28,28),aspect='equal',title='Filter-plane allocation at 20 um period',xlabel='X (mm)',ylabel='Y (mm)')
    ax.legend(loc='lower left',fontsize=7)
    ax=axs[1]
    ax.add_patch(Circle((0,0),25,fill=False,color=INK,ls='--'))
    for x in mc['station_x_mm']:
        for sign in [-1,1]:ax.add_patch(Rectangle((sign*x-4,-28),8,56,fill=False,edgecolor=BLUE,lw=.8,ls='--'))
    for index in range(0,9,2):
        x=mc['station_x_mm'][index//2];y=d['centres_mm'][index]
        ax.plot([-x,x],[y,y],color=BLUE,lw=1)
        for sign in [-1,1]:ax.add_patch(Rectangle((sign*x-4,y-4),8,8,facecolor=BLUE,alpha=.22,edgecolor=BLUE))
        ax.text(x,y+4.4,f'R{index+1}',fontsize=7,ha='center')
    ax.set(xlim=(-78,78),ylim=(-38,38),aspect='equal',xlabel='X (mm)',ylabel='Y (mm)',
        title='Layer A: staggered X stations / paired Y sliders')
    ax.text(0,-35,'Dashed boxes = entire ±24 mm travel envelope',ha='center',fontsize=8)
    ax=axs[2]
    for layer,z,color in [(0,-2,BLUE),(1,2,GOLD)]:
        ax.plot([20,45],[z,z],color=color,lw=1.5,label=f'Layer {"AB"[layer]}, z={z:+g} mm')
        bodyz=-6 if layer==0 else 6
        for x in [31,41]:
            ax.add_patch(Rectangle((x-4,bodyz-2),8,4,facecolor=color,alpha=.22,edgecolor=color))
            lo=min(z,bodyz);ax.add_patch(Rectangle((x-.75,lo),1.5,abs(z-bodyz),facecolor=color,alpha=.55))
    ax.axvline(25,color=INK,ls='--');ax.text(24.5,0,'50 mm opening edge',rotation=90,ha='right',va='center',fontsize=7)
    ax.set(xlim=(20,46),ylim=(-11,11),aspect='equal',xlabel='X (mm)',ylabel='Z (mm)',title='Side view: hardware stays outside aperture')
    ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Concept envelopes, not released CAD | 9 independent Y DOF / 18 paired end sliders',fontsize=12)
    save(fig,'01_mechanical_envelopes')


def tradeoff(data,search):
    rows=[r for r in search['rows'] if r['period_um']==20 and r['mechanics']['dynamic_model_ok'] and r['worst_retention']>=.6]
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for process,color,marker in [('precision',BLUE,'o'),('standard',GOLD,'s')]:
        rs=[r for r in rows if r['process']==process]
        axs[0].scatter([r['worst_background_suppression_db'] for r in rs],[r['worst_retention']*100 for r in rs],c=color,marker=marker,s=12,alpha=.4,label=process)
        axs[1].scatter([r['maximum_halo_fraction']*100 for r in rs],[r['worst_pair_intensity_perturbation']*100 for r in rs],c=color,marker=marker,s=12,alpha=.4)
    for mode,marker in [('metrology','*'),('balanced','D')]:
        v=get_preset(data,mode);r=v['worst']
        axs[0].scatter(r['suppression_db'],r['minimum_retention']*100,marker=marker,s=90,c=INK,label=mode+' / tested')
        axs[1].scatter(r['maximum_halo']*100,r['maximum_crosstalk']*100,marker=marker,s=90,c=INK,label=mode)
    axs[0].set(xlabel='Background suppression (dB)',ylabel='Worst defect energy retained (%)',title='20 um period: process-qualified optical candidates')
    axs[1].set(xlabel='Halo energy (%)',ylabel='Worst pair perturbation (%)',title='Ringing and pair interference')
    axs[1].axhline(15,color='0.5',ls='--');axs[1].axvline(20,color='0.5',ls='--')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'02_tradeoffs')


def profile_plot(data,ref):
    detection=next(r for r in ref['rows'] if r['design']['process']=='precision' and r['design']['period_um']==20 and r['design']['active_rods']==9)
    files=[('metrology',j.OUT/'final_metrology_20_profiles.npz',BLUE,'-'),
           ('balanced',j.OUT/'final_balanced_20_profiles.npz',GOLD,'--'),
           ('background priority',j.OUT/f'profiles_{detection["search_id"]}.npz',INK,':')]
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,path,color,ls in files:
        z=np.load(path);x=z['axis_mm']*1000
        for i,col in enumerate([2,6]):
            val=m.pixel_average_1d(abs(z['fields'][:,col])**2,(x[1]-x[0])/1000,.00025)/m.camera_intensity_reference(m.config())
            axs[i].semilogy(x,np.maximum(val,1e-12),label=name,color=color,ls=ls,lw=1.2)
    for ax,size in zip(axs,[.5,2]):
        ax.set(xlim=(-45,45),ylim=(1e-6,1.2),xlabel='y relative to defect (um)',ylabel='I / I_ref',title=f'{size:g} um defect / common intensity reference')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    save(fig,'03_profile_modes')


def report(data,checks,rods,search,ref,robust):
    b=get_preset(data,'balanced');met=get_preset(data,'metrology')
    detect=next(v for v in ref['rows'] if v['design']['process']=='precision' and v['design']['period_um']==20 and v['design']['active_rods']==9)
    dr=next(v['worst'] for v in robust['rows'] if v['search_id']==detect['search_id'])
    rows=[]
    for name,nom,worst in [('低振铃/计量',met['nominal'],met['worst']),('保守折中',b['nominal'],b['worst']),('强背景抑制对照',detect,dr)]:
        rows.append(f'| {name} | {nom["suppression_db"]:.3f} → {worst["suppression_db"]:.3f} | {reduce_percent(worst["suppression_db"]):.1f}% | {worst["minimum_retention"]*100:.1f}% | {worst["maximum_halo"]*100:.1f}% | {worst["maximum_crosstalk"]*100:.1f}% | {worst["maximum_ee90_ratio"]:.2f}× |')
    recipes=[]
    for item in data['presets']:
        v=item['selected']
        if not v:continue
        r=v['worst'];recipes.append(f'| {item["period_um"]} | {"计量" if item["mode"]=="metrology" else "折中"} | {v["design"]["active_rods"]}/9 | {", ".join(map(str,v["design"]["object_orders"]))} | {r["suppression_db"]:.3f} | {r["minimum_retention"]*100:.1f}% | {r["maximum_halo"]*100:.1f}% | {r["maximum_crosstalk"]*100:.1f}% | {r["maximum_ee90_ratio"]:.2f}× |')
    d=b['design'];positions='\n'.join(f'| R{i+1} | {"AB"[k]} | {d["z_mm"][k]:+.1f} | {y:+.4f} | {"工作" if active else "瞳外停放"} | ±{j.config()["mechanics"]["station_x_mm"][i//2]:g} |' for i,(y,k,active) in enumerate(zip(d['centres_mm'],d['layers'],d['active_flags'])))
    hardware_b=next(v['best'] for v in read('selection.json')['winners'] if v['process']=='standard' and v['gate']=='balanced')
    std_id=next(v['search_id'] for v in hardware_b['recipes'] if v['period_um']==20);std=next(v for v in ref['rows'] if v['search_id']==std_id)
    base_mech=j.mechanical(j.design(.1,.05,'precision',[-5,-3,-2,-1,0,1,2,3,5],20))
    maxs=max(r['worst_background_suppression_db'] for r in search['rows'] if r['period_um']==20 and r['process']=='precision'
             and r['worst_retention']>=.6 and r['maximum_halo_fraction']<=.1 and r['worst_pair_intensity_perturbation']<=.1 and r['maximum_ee90_ratio']<=2)
    txt=rf'''# 独立可调全锯齿滤杆：光学、加工、机械三维联合方案

## 技术结论：推荐一种硬件、两种经误差抽查的工作配方

**推荐的硬件路线是：9根同规格薄片锯齿杆、双层布置、每根独立Y向移动、两端联动导向和张紧，宽滑台在50 mm通光孔外沿X错位排列。** 同一套硬件固定使用0.100 mm名义核心全宽、0.050 mm单侧齿深、0.200 mm名义全包络、0.750 mm齿周期、0.050 mm薄片厚度；两层位于傅里叶面±2 mm。不是上一版B层0.0375 mm齿深，也不是固定梳形片。

但**不存在本轮已证实的“高背景抑制、低振铃、不损失小缺陷、容易加工、机械动态零串扰”同时最优点**。推荐的是有条件的工程折中路线，不是加工放行或全局最优证明。尤其不能用主峰FWHM不变来代替缺陷尺寸保真；本轮把EE90能量宽度单列，并对计量配方要求不超过无滤波的2倍。

以20 µm物体周期为例，保守折中配方在35组指定误差/偏场检查中最差抑制{b['worst']['suppression_db']:.3f} dB，即背景积分能量减少{reduce_percent(b['worst']['suppression_db']):.1f}%；目标最差积分保留{b['worst']['minimum_retention']*100:.1f}%，晕圈{b['worst']['maximum_halo']*100:.1f}%，双缺陷扰动上界{b['worst']['maximum_crosstalk']*100:.1f}%。这个配方适合折中检出，**不保证尺寸计量**。

若首要任务是防止缺陷能量轮廓变大，应切换同硬件的低振铃配方：20 µm条件下最差背景抑制仅{met['worst']['suppression_db']:.3f} dB，但最差能量保留{met['worst']['minimum_retention']*100:.1f}%，EE90最大{met['worst']['maximum_ee90_ratio']:.2f}倍。不能同时把强抑制配方约10 dB的效果算到这个低振铃配方上。

## 1. 三个维度如何进入决策

| 维度 | 评价量 | 本轮决策方式 |
|---|---|---|
| 光学 | 背景积分抑制S、最小缺陷积分保留R、保护区外能量H、双缺陷最坏相位扰动C、FWHM及EE90 | 保留原始指标；设置公开阈值与误差裕量，再比较背景收益，不只最大化目标亮度 |
| 可加工性 | 薄片厚度、误差后最窄核心、齿深/轮廓误差比、尖谷平台、圆角及材料强度 | 不满足规则先淘汰；同硬件所有杆相同截面，换周期不换杆宽/齿深 |
| 机械 | 杆与连接指净空、滑台全行程包络、瞳外停放、切换路径、张紧挠度 | 几何及行程作硬约束；力学仅为明确假设下的初估，动态实测单列为未完成 |

本轮真实计算{search['candidate_count']:,}组对称候选，加{data['extension_count']:,}组允许非对称/偶数活动杆的补充候选。两个候选族有重复，**不能相加声称是去重后的总数**。30组初选经过4 mm实体层距传播及每组35个误差状态复核；保守配方又单独复核。新增计算均为Python，不冒充新Zemax结果。

图中圆点/方点分别表示高精度与标准工艺档；叠加的实心标记为经过误差检查的最终配方。图说明可行候选之间存在明显取舍，不是一个指标领先就代表三维都最优。

![三维筛选后的光学权衡](results/joint_adjustable/figures/02_tradeoffs.png)

## 2. 固定光学系统与评价基准

355 nm、NA=0.6、f=30 mm、20×、Ch3中路法向照明及收集；**实际光学瞳36 mm，机械通光孔50 mm**，没有引入瞳放大。主线扫照明强度FWHM为物方1 µm，像元/有效线高物方0.25 µm。孔径有限，即使无像差也仍有衍射。

$$
\alpha=\lambda f=0.01065\ \mathrm{{mm}}^2,\qquad
Y_n=\frac{{\lambda f}}{{p_{{\rm object}}}}n.
$$

用户尚未给出最终周期上下限，暂验证10、15、20、30、40 µm。各周期用两种条宽占空比0.35和0.245取较差背景；20 µm时就是原来的7/4.9 µm条宽。缺陷是0.5、0.75、1、1.5、2、3 µm高斯强度FWHM，R取6种尺寸的最小积分比；这不是只挑0.5 µm小点，也不是峰值保留率。不同占空比、反射率或真实周期范围需要重新算配方。

$$
S=-10\log_{{10}}(E_{{B,\rm filtered}}/E_{{B,0}}),\qquad
\eta_{{\rm suppression}}=1-10^{{-S/10}}.
$$

H统计半径max(0.75 µm,1.5倍缺陷尺寸)外的能量；C在正负1～25 µm、0.05 µm间距上取0.5 µm双缺陷的最坏相位扰动上界。EE90是以原目标位置为中心、包含90%目标能量的全宽。所有轮廓图共用I/I_ref，其中I_ref={m.camera_intensity_reference(m.config()):.12f}，不逐图或逐缺陷归一化，不能直接解释成W/mm²或DN。

双层复场传播保留−k、+k重新回到总零级的通道，不把两层各自零级简单相乘。初筛8192点、复核16384点、窗口1.024 mm；复核至±24层间谐波。沿杆方向假定只接收主线扫总零级；有限TDI接收仍见第8节限制。

## 3. 同一硬件的三种光学取舍，不能混用成绩

下面“最差”指本次指定的35个状态中取最坏，不是所有可能误差的数学证明：名义状态、8个共同核心/齿尖/位置误差角点、2个±0.20 mm偏场状态、24个逐杆独立随机误差状态（种子355）。制造误差为边缘±10 µm、位置总预算每杆±25 µm；形状误差保持核心边和齿尖边各自受限。

| 20 µm周期配方 | 抑制dB：名义→抽查最差 | 最差背景抑制率 | 最差目标保留R | 最大晕圈H | 最大串扰C | 最大EE90倍率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

低振铃档使用H≤10%、C≤10%、R≥60%、EE90≤2倍；折中档使用H≤20%、C≤15%、R≥60%，没有EE90硬上限；强抑制档只是对照。上述数值是本轮公开的工程评价假设，不是用户已经确认的验收规格。最终配方先保留名义裕量，再通过35状态检查；**失败的原初选没有被写成合格结果**。

例如对称初选的“背景优先”方案名义约10.467 dB，误差后H可到28.0%、C约20.1%；名义“折中”方案误差后H约22.2%，也不能按20%门槛放行。补查非对称布置后，才得到本表的保守折中配方。对于计量，连H/C通过也不够，还要看EE90。

图中统一强度画出0.5和2 µm目标的外围能量。强背景抑制的副像更明显；这些是滤波器光学响应，不是TDI电荷转移把原像拉长。

![同一硬件的缺陷能量轮廓](results/joint_adjustable/figures/03_profile_modes.png)

## 4. 工艺：50 µm张紧薄片，比直接照搬毫米级杆更匹配本系统

| 项目 | 推荐条件 | 说明 |
|---|---:|---|
| 名义核心全宽 / 单侧齿深 / 全包络 | 0.100 / 0.050 / 0.200 mm | 所有杆同规格，避免更换周期时偷偷改尺寸 |
| 薄片厚度 / 三角齿周期 | 0.050 / 0.750 mm | 齿周期满足≥0.5 mm；厚度不是核心宽度 |
| 轮廓误差目标 | ±0.010 mm | 必须由加工供应商对这张图确认，非已取得承诺 |
| 误差后最窄核心 | 0.080 mm | 与制造允许的最小连筋50 µm比较 |
| 单侧齿深 / 单边误差 | 5 | 本项目要求至少4；上一版37.5 µm齿深在本档下低于这个裕量 |
| 尖谷平台 / 圆角评估半径 | 0.020 / 0.050 mm | 浅角度三角齿可放入圆角；圆角不能凭“齿深小于半径”判失败 |
| 工作张应力 / 材料证明应力条件 | 100 / ≥300 MPa | 指误差后最窄核心；必须取得具体材料及热处理状态数据 |

公开工艺能力不能互相混用：[Precision Micro](https://www.precisionmicro.com/chemical-etching/technical-guidelines/)给出的薄片标准公差档约±25 µm；[micrometal](https://www.micrometal.de/en/service-portfolio/our-technologies/)公开了更细的薄片特征和约±7 µm级最低公差指导值。由此本轮设置“±10 µm高精度”与“±25 µm标准”两档；这只是建模依据，不是询价、交期或质量保证。

若只能保证±25 µm，按同一4倍齿深裕量，齿深至少0.10 mm、平台至少0.05 mm，不能继续拿±10 µm硬件的光学成绩作为预期。标准工艺档的跨周期折中硬件在20 µm条件下名义抑制仅{std['suppression_db']:.3f} dB。PPT毫米级外包络还会进一步损失缺陷频谱，原尺寸对照已在[上一轮报告](全锯齿双层傅里叶滤波_优化结果.md)保留。

## 5. 机械：每杆独立移动，但不能任意穿过邻杆

采用9个独立Y自由度、18个两端滑台。每根两端Y位置同步，其中一端允许X向弹性张紧；不采用长细杆单端悬臂。A层5根、B层4根，层距4 mm。未使用的杆停在Y≈±20/±21 mm，仍计入装机9根。

### 5.1 宽滑台在光瞳外错位，不挤在频谱峰间距上

暂分配每层5个X工位：±31、±41、±51、±61、±71 mm。每工位滑台X宽8 mm、相邻节距10 mm；宽滑台退到滤杆层外侧，窄连接指宽Y=0.25 mm到达滤杆面。最大跨距142 mm，建议预留机构概念包络约156×76×20 mm，不含线缆和未选定电机端部。

**这里的8 mm是允许的滑台/导向包络，不是已找到的商用电机型号尺寸。** 实际电机、螺钉和线缆加入后必须重跑碰撞检查。如果把8 mm宽滑台直接装在约1 mm杆间距的同一个X位置上，会干涉；只画细杆并不能证明机构成立。

图左包含工作杆和瞳外停放杆；中图展示A层两端滑台的整个移动包络；右图显示两层及支承在通光孔外的Z向退让。图是可检查的概念包络，不是正式装配CAD。

![独立移动机构与避让包络](results/joint_adjustable/figures/01_mechanical_envelopes.png)

### 5.2 已检查的几何净空

| 检查对象 | 本轮结果 | 覆盖范围 |
|---|---:|---|
| 最终配方之间的连续切换路径 | {checks['motion_paths']}条 | 每条101点，并用中心距凸组合给出解析下界 |
| 路径最小同层杆/连接指净空 | {checks['minimum_rod_gap_mm']:.4f} mm | 已计每杆±25 µm位置及轮廓误差包络，要求≥0.10 mm |
| 滑台全行程包络 | {checks['slider_swept_boxes']}个，{checks['box_pair_checks']}对 | Y全行程±24 mm |
| 滑台包络碰撞数 / 最小分离量 | 0 / {checks['minimum_slider_separating_gap_mm']:.3f} mm | 所定义包围盒含0.10 mm安装误差，不含未建模零件 |
| 滑台对50 mm开口最小净空 | {checks['minimum_opening_clearance_mm']:.3f} mm | 不仅避开36 mm光学瞳，也避开50 mm开口 |
| 两层滤杆轴向净空预算 | {base_mech['cross_layer_rod_clearance_mm']:.3f} mm | 4 mm层距减板厚、轴向误差和挠曲预算 |

同层杆号始终保持Y顺序，位置切换采用共同进度s：

$$
Y_i(s)=(1-s)Y_i^{{(A)}}+sY_i^{{(B)}},\quad 0\le s\le1.
$$

相邻中心距同样是端点距离的凸组合，因此两个端点均满足净空时，全程不小于较小端点净空。**这不保证任意异步单杆动作安全**：单杆调节必须实时检查左右邻杆；配方切换要联动规划，不能让一根杆穿越同层另一根。

### 5.3 动态串扰不是0，张紧方案还需实测

锯齿增加质量，必须用平均宽度与最窄核心之比修正张紧弦估计：

$$
\beta=\frac{{w+b}}{{w-2\epsilon_{{\rm edge}}}}=1.875,\quad
\delta_{{\max}}=\frac{{\rho\beta aL^2}}{{8\sigma}},\quad
f_1=\frac1{{2L}}\sqrt{{\frac{{\sigma}}{{\rho\beta}}}}.
$$

暂取ρ=8000 kg/m³、工作应力100 MPa、最长L=142 mm。每杆预张力约{base_mech['force_per_rod_n']:.3f} N；最长杆2g等效静载挠度约{base_mech['maximum_installed_2g_deflection_um']:.2f} µm，张紧弦最低频率约{base_mech['minimum_string_frequency_hz']:.0f} Hz。10g估计挠度再乘5，**不能满足10 µm预算**。这些不是共振附近的动态上界，必须核实滤杆实际振动输入和支承模态。

两端Y位置差若控制在4 µm内，按最短62 mm跨距，18 mm瞳边缘的偏航误差约1.16 µm。共架反力若按1 N、允许邻杆移动2 µm估算，所需交叉方向等效刚度至少500 N/mm；目前尚未完成FEA或实测确认。一端张紧弹簧还需验证热漂移和行程。因此结论是**几何包络无干涉已检查；机械动态零串扰未证明**。

## 6. 20 µm保守折中配方：逐根位置

| 杆号 | 层 | z mm | Y mm | 状态 | 两端支承X mm |
|---|---|---:|---:|---|---:|
{positions}

该配方可使用非对称/偶数活动杆，但物理总数始终9。停放杆在瞳外，不等于从机械系统移除了。

## 7. 同一硬件适配不同周期的结果

下表全部为35状态抽查最差；只改变位置，不改变硬件截面。计量配方对EE90有2倍限制，折中配方没有，因此两者用途不同。所有最终入选配方均通过各自声明的抽查门槛，但这不是连续全参数误差保证。

| 周期µm | 用途 | 工作/装机杆数 | 工作衍射级次 | 抑制dB | R | H | C | EE90倍率 |
|---:|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(recipes)}

完整可执行位置表见[全部配方CSV](results/joint_adjustable/all_position_recipes.csv)，包括每根停放杆的位置。周期增大时谱峰更密，同样杆宽占用的目标频谱比例变大，制造宽度和同层净空同时变得更紧；**可移动不等于任意周期下都能保持10 dB抑制**。

## 8. 结果边界与下一步

1. 本轮回答的是固定355 nm、36 mm瞳、≤10根、全锯齿、独立可调条件下的工程取舍。候选范围、光学门槛、10～40 µm验证范围和制造/载荷假设均公开；不能称为任意结构的全局最优。本轮固定装机9根、齿距0.75 mm、层距4 mm，并未穷举10根装机结构或连续齿距/层距；齿距与较大尺寸的前轮扫描结果另见上一轮报告。
2. 低振铃和强去背景存在固有取舍：挡掉周期频谱也会损失目标在相同频率的分量。20 µm的对称候选中，同时R≥60%、H/C≤10%、EE90≤2倍的最高预筛抑制约{maxs:.3f} dB，远非10 dB；补查非对称后也没有得到同时满足强抑制与计量门槛的已验证解。
3. 0.75 mm齿距的一级像移为14.2 µm。256级TDI可能接收非零总级；本轮最终配方仍按主线扫条件优化，**没有把它宣称为256级TDI全级次最优**。TDI严格同步不拉长原像，接收光学副像是另一个问题。
4. 50 µm薄片的实际材料、边缘轮廓、355 nm吸收/温升、真实马达/导轨尺寸、位移反馈、FEA、机械耦合及实机模态都需要后续确认。没有这些，不能作机械零串扰或加工放行承诺。

建议先用本报告的同规格9杆独立可调路线制作**结构与工艺验证件**：先确认供应商能否保证±10 µm轮廓和≥300 MPa材料证明应力，再把实际滑台、电机与走线包络代入防撞模型。若任务确实以缺陷尺寸计量为优先，用低振铃配方；若背景残留仍不够低，不应直接切到10 dB配方并仍声称尺寸保真。

待确认问题：实际物体周期/线宽范围、可接受的R/H/C及EE90门槛、材料和加工供应商、允许的机构尺寸、滤杆实际振动/温升，以及最终25或256级TDI收集条件。

## 9. 文件、复现与验证

模型先行文档：[三维约束与模型](三维联合优化_约束与模型.md)。环境仍为本机Anaconda `E:\anconda\envs\fourier-filter\python.exe`。代码：[联合搜索](src/joint_adjustable.py)、[逐杆误差检查](src/joint_robustness.py)、[非对称和保守配方](src/joint_asymmetric.py)、[报告及机械包络检查](src/joint_report.py)。完整原始结果在`results/joint_adjustable`，失败的初选及标准工艺对照未删除。

```powershell
$py = 'E:\anconda\envs\fourier-filter\python.exe'
& $py src/joint_adjustable.py scan
& $py src/joint_adjustable.py qualify
& $py src/joint_adjustable.py refine
& $py src/joint_robustness.py
& $py src/joint_asymmetric.py
& $py src/joint_report.py
& $py -m pytest -q --junitxml=results/joint_adjustable/pytest.xml
& $py src/validate_joint_artifacts.py
```

验收记录见[数值与交付验证](results/joint_adjustable/artifact_validation.json)。单元测试、几何包络和误差抽查通过，不替代供应商制造确认、实机FEA/模态或整机计量验收。
'''
    (j.m.ROOT/'独立可调锯齿杆_三维联合最优方案.md').write_text(txt,encoding='utf-8')


def run():
    FIG.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    data=read('final_presets.json');search=read('search.json');ref=read('refined.json');robust=read('robustness.json')
    checks,rods=final_mechanics(data);mechanical_figure(data);tradeoff(data,search);profile_plot(data,ref)
    report(data,checks,rods,search,ref,robust)
    print('Three-domain report, final recipes, figures and motion checks written.',flush=True)


if __name__=='__main__':run()
