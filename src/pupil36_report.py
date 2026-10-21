"""Refresh result-bearing sections while retaining the full theory narrative.

The archived 50-mm-assumption report is a stable narrative template, NOT a
numerical source. Current numerical tables are read only from the 36-mm run.
"""
import csv
import json
import re
import numpy as np
import ch3_linescan as m
from ideal_na06 import save_json

ROOT=m.ROOT


def table(headers,rows):
    return "\n".join(["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+
                     ["|"+"|".join(map(str,row))+"|" for row in rows])


def build(cfg,opt,validation):
    out=m.OUT;sc=m.scales(cfg);d=opt["choices"]["selected"]
    read=lambda name:json.loads((out/name).read_text(encoding="utf-8"))
    link=lambda name:f'[{name}]({(out/name).as_posix()})'
    image=lambda name:f'![]({(out/name).as_posix()})'
    def rows(name):
        with (out/name).open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
    document=(ROOT/"results/ch3_linescan/综合报告_50mm假设历史版.md").read_text(encoding="utf-8")
    def section(start,end,body):
        nonlocal document
        i=document.index(start);j=document.index(end,i+len(start))
        document=document[:i]+start+"\n\n"+body.strip()+"\n\n"+document[j:]
    full=[];supp={};ret={}
    for case,tag,base in [("dots","selected_fine","none_fine"),("stripes","selected_fine","none_fine"),
                          ("dots","background_only","none"),("dots","best_hard","none")]:
        a=read(f"{case}_{tag}_full_metrics.json");b=read(f"{case}_{base}_full_metrics.json")
        s=10*np.log10(b["reference_energy"]/a["reference_energy"])
        rr=np.array(a["defect_energies"])/b["defect_energies"]
        full.append([case,tag,f'{a["dx_um"]:.3f}',f'{s:.3f}',*[f'{100*r:.2f}' for r in rr]])
        if tag=="selected_fine":supp[case]=s;ret[case]=rr
    choice=[]
    for key,label in [("none","无挡杆"),("ring_priority","振铃优先、弱去背景"),("selected","10 dB档综合折中"),
                      ("best_hard","10 dB档硬边折中"),("background_only","背景抑制优先")]:
        r=opt["choices"][key]
        choice.append([label,r["rod_count"],f'{r["profile"]}; {r["core_width_mm"]:.2f}/{r["b_mm"]:.2f}',
                       f'{r["worst_background_suppression_db"]:.3f}',f'{100*r["worst_retention"]:.2f}',
                       f'{100*r["maximum_halo_fraction"]:.2f}',f'{100*r["worst_pair_intensity_perturbation"]:.2f}'])
    strict=sum(r["feasible"]=="True" and float(r["worst_background_suppression_db"])>=10
               and float(r["maximum_halo_fraction"])<=.1 and float(r["worst_pair_intensity_perturbation"])<=.1
               for r in rows("candidate_scan.csv"))
    document=document.replace("50 mm有效傅里叶瞳","36 mm实际光学瞳、50 mm机械通光孔")
    document=document.replace("数值来自本项目保存的Python结果；本次整理重新执行了42项测试，全部通过，未重新运行整套大网格优化。",
        "本次已按实际36 mm光瞳、物理挡杆不超过10根重新优化并重跑二维传播。所有主成像图统一使用固定参考强度I/I_ref；保留此前通用理论，并替换参数、数值、图表和结论。不是本轮重新运行Zemax得到的结果。")
    section("## 摘要：目前得到的设计结论","## 1.",f"""
当前方案为**9根挡杆**，包含中央零级挡杆；实际光学瞳为**36 mm**，50 mm只作为机械开口，不增加瞳放大。
在{opt['candidate_count']:,}个尺寸与对称位置候选中，原先15 dB示例门槛未达到。改列已计算的10 dB档综合折中及最大去背景方案，不将它称为15 dB合格方案或全局最优。

{table(['项目','当前候选值'],[
['轮廓','二值齿形，经零级选择形成专利正弦振幅过渡'],['物理挡杆数',str(d['rod_count'])+'根，≤10根'],
['物体衍射级次n',', '.join(f'{n:+d}' for n in d['orders'])],['核心全宽 / 单侧齿深 / 包络','0.10 / 0.05 / 0.20 mm'],
['齿距','0.75 mm；齿尖及谷底20 µm平台'],['频谱格点间隔','0.5325 mm；并非每个格点都放杆'],
['一维离散公差最坏背景抑制',f"{d['worst_background_suppression_db']:.3f} dB"],
['完整二维圆点 / 长条背景抑制',f"{supp['dots']:.3f} / {supp['stripes']:.3f} dB，均8192²细网格"],
['完整二维孤立缺陷能量保留',f"{100*min(*ret['dots'],*ret['stripes']):.2f}%～{100*max(*ret['dots'],*ret['stripes']):.2f}%"],
['一维最坏保留率',f"{100*d['worst_retention']:.2f}%"],['最大沿线外围能量H / 双缺陷串扰C',f"{100*d['maximum_halo_fraction']:.2f}% / {100*d['worst_pair_intensity_perturbation']:.2f}%"]])}

**仍不能视为密集缺陷计量合格。** 同时要求S≥10 dB、R≥60%、H≤10%、C≤10%时，本候选集中合格数为{strict}。相干复制旁瓣沿相机长方向出现，不是TDI积分拉长。旧图右下角缺陷因显示范围裁切而不可见，新图统一±45 µm，三处真缺陷均标注。

强度统一为**I/I_ref（无量纲相对强度）**，固定同一均匀透明物体无挡杆参考；不做逐图峰值归一化。线性主图0～1.2，对数诊断图10⁻⁶～1.2，差图为有符号ΔI/I_ref。未给定功率和相机标定，不能将数值标作W/mm²或相机DN。
""")
    document=document.replace("|傅里叶有效口径|直径50 mm|解释为被瞳中继充满的有效滤波瞳|",
        "|实际光学瞳 / 机械开口|直径36 / 50 mm|瞳中继倍率1；机械孔不决定光学瞳径|")
    section("### 3.3 为什么30 mm物镜与50 mm傅里叶口径之间需要说明瞳中继","### 3.4",r"""
当前确认实际光学瞳仍为36 mm。因此不再从50 mm机械开口反推瞳放大：

$$
D_{\mathrm{pupil}}=2f_{\mathrm{obj}}\mathrm{NA}=2\times30\times0.6=36\ \mathrm{mm},
\qquad m_p=1,\quad f_{\mathrm{FF}}=30\ \mathrm{mm}.
$$

$$
\alpha=\lambda_0f_{\mathrm{FF}}=0.010650\ \mathrm{mm}^2,\qquad
\Delta Y=\frac{\alpha}{p_y}=\frac{0.010650}{0.020}=0.532500\ \mathrm{mm}.
$$

$$
\nu_x^2+\nu_y^2\le(\mathrm{NA}/\lambda_0)^2,\qquad
\nu_c=1690.141\ \mathrm{cycles/mm}.
$$

有效瞳半径18 mm；机械开口半径25 mm。18～25 mm环带没有新增物镜角谱。此前按50 mm被光学填满的计算保留为历史记录，不与本版尺寸混用。
""")
    grid=[]
    for n in [1,10,20,30,33]:
        sn=n*.000355/.020;r=30*sn;rt=r/np.sqrt(1-sn*sn)
        grid.append([n,f'{sn:.6f}',f'{r:.6f}',f'{rt:.6f}',f'{100*(rt/r-1):.5f}'])
    section("### 3.4 对同一个20 µm网格，频谱有什么区别","## 4.",
        "对象的离散空间频率不变，变的是这些频率对应的傅里叶面位置。当前无瞳放大：\n\n"+
        table(['n','sinθ=nλ/p','正弦坐标/mm','误用tan坐标/mm','相对误差/%'],grid)+
        "\n\n第30、33级按正弦映射仍在18 mm瞳半径内；误用tan却落在瞳外。二维角谱须按径向方向余弦共同换算，不能独立拉伸两轴。若真的构建另一种投影光学系统，还须处理频谱密度的雅可比，不能仅改坐标标签就声称能量守恒。")
    document=document.replace("19.722 µm，像方移位0.394444 mm","14.200 µm，像方移位0.284000 mm")
    document=document.replace("0.7396 mm的频谱峰距","0.5325 mm的频谱峰距")
    document=document.replace("当前使用基片上的二值遮光图形", "当前使用基片上的二值遮光图形")
    # Recalculate the original patent size table at the current focal scale.
    patent=[]
    for r in rows("patent_size_scan.csv"):
        if np.isclose(float(r['a_mm']),.05):
            patent.append([r['profile'],f"{float(r['b_mm']):.3f}",f"{float(r['outer_width_mm']):.3f}",f"{float(r['tail_reduction_db']):.3f}"])
    section("### 5.2 不同尺寸能把远尾降低多少","### 5.3",f"""
固定核心全宽0.10 mm，单根挡杆、相同入射振幅，积分物方两侧25～250 µm衍射强度，相对硬边计算降低量；改变b也改变遮挡面积，不是等吞吐量比较。传播使用355 nm、30 mm。

{table(['轮廓','单侧b/mm','包络/mm','远尾积分降低/dB'],patent)}

正值表示该固定区间尾能量降低，负值表示增加。单根远尾改善不保证整组复制峰或最坏串扰改善。

{image('patent_actual_size_effect.png')}

{image('patent_fig8_10_recalculated.png')}

第二图以专利a=0.5 mm、b=0～2 mm复算，纵轴log10振幅，不是强度dB，也不是可直接逐级安放的当前加工尺寸。
""")
    document=document.replace("固定0.10 mm核心及全部目标级次","固定0.10 mm核心及当前选中的9个级次")
    document=document.replace("若遮挡函数由间距ΔY的多根相同挡杆组成，在不重叠近似下", "若遮挡函数由频谱格点ΔY上的有限组挡杆组成，求和仅遍历当前9个选中级次，在不重叠近似下")
    document=document.replace("684候选、条件优选、Pareto候选",f"{opt['candidate_count']}候选、显式≤10根、条件优选、Pareto候选")
    document=document.replace("入射与输出在同一排使用相同标度。两排照明到的物体不同，不能跨排按色彩直接比较绝对能量。",
        "两排入射与输出都用同一个频谱参考I_FF,ref，允许相对比较；中列是强度透过率。频谱参考与像面I_ref不是同一个物理平面的量，不混作像面强度。青色线标注实际挡杆位置。")
    iref=m.camera_intensity_reference(cfg)
    extra=rf"""
### 6.4 像面强度的统一单位与显示规则

所有主成像数据先按相同物场、照明、FFT和相机算子求强度，再统一除以同一个参考：

$$
I_{{\mathrm{{ref}}}}=\frac{{1}}{{w_c}}\int_{{-w_c/2}}^{{w_c/2}}
\left|\mathcal{{F}}^{{-1}}\left[P\mathcal{{F}}\{{A(x)\}}\right](u)\right|^2\,\mathrm{{d}}u,
\qquad I_{{\mathrm{{display}}}}=I_{{\mathrm{{camera}}}}/I_{{\mathrm{{ref}}}}.
$$

参考物体为无限均匀单位振幅透明物体、无挡杆；相同1 µm线照明、NA、0.25 µm物方线高。解析角谱与12点高斯积分得到原入射振幅规范下I_ref={iref:.12f}。参考不依赖圆点/长条、缺陷大小、滤片方案、单图峰值或该图使用的3/5点积分设置。

原图显示几何参考强度|t|²，而非振幅t。主强度色标固定0～1.2；所有对数诊断图固定10⁻⁶～1.2；差图固定−1.2～1.2，单位为ΔI/I_ref。对数只改变色彩映射，不改变强度单位。以上范围覆盖本轮两类主图最大值。

一维缺陷强度剖面和双缺陷强度曲线也已恢复高斯物场的连续傅里叶幅度，并统一到同一I_ref，不再按各自中心峰值归一化。一维曲线取扫描中心u=0的场，再沿y积分像元；主图还包含有限线高积分，因此两者不是完全相同的采集算子。

FWHM、halo、串扰比例和单根挡杆变换曲线是单独定义的诊断指标，不等同于主图像面强度。不同照明的TDI点散射模型也单独声明参考，不能与本节有限缺陷成像按亮度直接混比。

本项目没有入射光功率、曝光时间、量子效率和DN增益标定，因此只能给出无量纲相对强度，不能虚构W/mm²、电子数或DN。
"""
    document=document.replace("## 7. 如何把振铃",extra+"\n\n## 7. 如何把振铃")
    section("### 7.2 搜索空间与约束","### 7.3",f"""
核心全宽0.10～0.40 mm、步长0.02 mm；单侧b=0～0.20 mm、步长0.025 mm。b=0为硬边，否则比较线性及正弦过渡。

**最多10根是物理根数硬约束**，中央零级挡杆也计数。此次位置搜索保持关于原点对称：中央1根，加上从n=1～8中选择的至多4对±n，因此候选为1、3、5、7、9根，9根并不超过用户的10根上限。共有{opt['layout_count']}种位置组合；未搜索非对称10根、任意连续位置或独立宽度，故不是全部≤10根设计的全局最优。

与加工净隙≥0.10 mm、齿尖/谷底20 µm平台、核心全宽误差余量结合后，共{opt['candidate_count']:,}个候选；{opt['feasible_count']:,}个满足R≥60%及核心FWHM倍率≤1.10。未被选中的级次不再隐含附加挡杆。

公差抽查位置/核心全宽误差：(0,0)、(+25,−10)、(−25,−10)、(0,+10) µm。背景初筛为7 µm及4.9 µm宽周期条带代理，S、R取离散组合不利值；未穷尽连续公差域，也不把代理S冒充完整圆点结果。
""")
    document=document.replace("先满足加工、R和FWHM约束，再限定S至少15 dB，最后取J最高者。15 dB门槛、5% halo及10%串扰的评分尺度是公开的示例工程偏好，不是已经得到用户确认的验收标准。",
        "先满足加工、根数、R和FWHM约束。原15 dB示例门槛在本候选集中不可达；现将预设的下一档10 dB作为展示折中门槛，再取J最高者，并另列最大S方案。10/15 dB门槛、5% halo和10%串扰评分尺度均为公开示例偏好，不是用户确认的验收标准。")
    defectrows=[[r['defect_fwhm_um'],f"{b['fwhm_um']:.3f}",f"{r['fwhm_um']:.3f}",f"{b['ee90_width_um']:.3f}",f"{r['ee90_width_um']:.3f}",f"{100*r['halo_fraction']:.2f}"] for r,b in zip(d['defect_metrics'],d['baseline_defect_metrics'])]
    pairrows=[[r['separation_um'],f"{100*r['worst_phase_intensity_perturbation']:.3f}"] for r in d['pair_metrics']]
    section("## 8. 最优候选、完整成像和未解决的问题","## 9.",f"""
### 8.1 候选之间的权衡

S、R为一维代理及离散公差不利值；H、C为名义轮廓指标，不能与二维表混用。

{table(['方案','根数','轮廓；核心/b mm','S/dB','R/%','H/%','C/%'],choice)}

最大去背景方案达到约10.58 dB，但缺陷保留更低、外围能量和串扰更高。主图选10 dB档综合评分最高的正弦方案，不把最大去背景方案等同于综合最优。无挡杆仍具有最低滤片诱导振铃，但不去除周期背景。

{image('joint_optimization.png')}

### 8.2 完整成像：统一强度、统一范围

{table(['对象','方案/网格','dx/µm','背景抑制/dB','D1保留/%','D2保留/%','D3保留/%'],full)}

{image('dots_full_scan.png')}

{image('stripes_full_scan.png')}

主图均为线性I/I_ref，横纵等比例，同一±45 µm范围。蓝圈D1=(9,9)、D2=(−9,29)、D3=(31,−31) µm仅为位置标注。滤后背景一般不为零；右下差图是|E_bg+E_d|²−|E_bg|²，有交叉项及负值，不是“最终物理成像”。第五图为三个缺陷场相干叠加、没有周期背景的独立计算。

无像差不代表无限NA；有限圆瞳仍可产生衍射、边缘振铃。原图应与几何强度|t|²比较，而不能把线性振幅图和低至10⁻⁶的对数强度图直接比较外观。无滤片输出总体恢复原物体结构，弱暗区振铃在对数图上被放大显示。

{image('stripes_log_diagnostic.png')}

对数图与主图单位和参考完全相同，不能据其可见亮斑直接判断缺陷变大多少。

### 8.3 为什么会复制、右下角为什么曾消失

旧图最后两格的y范围为−16～34 µm，D3的y=−31 µm被裁掉；另一个单缺陷面板只计算D1，原本就不应出现D2/D3。本次修正为共同视野，并分别展示三处孤立缺陷：

{image('stripes_defect_tracking.png')}

蓝圈是真缺陷，橙色菱形是y方向相距20 µm整数倍、且落在当前视野内的理论复制位置。单独传播一个缺陷、移除所有背景且不加入TDI后仍会出现旁瓣，来源是第5.3节有限挡杆阵列因子，不是新增物体或TDI拖尾。位置与对应强度可查{link('defect_tracking.csv')}；共同显示范围检查见{link('display_validation.json')}。

### 8.4 核心宽度、外围能量及双缺陷干扰

{table(['物体FWHM/µm','无滤片核心/µm','滤后核心/µm','无滤片EE90/µm','滤后EE90/µm','滤后H/%'],defectrows)}

{image('defect_core_tail_crosstalk.png')}

{image('defect_size_sweep.png')}

{table(['沿线缺陷间距/µm','最坏相位中心强度扰动/%'],pairrows)}

{image('coherent_pair_examples.png')}

20 µm附近复制响应可产生相位相关干扰；不能只看FWHM或只挑旁瓣零点。当前10 dB档也未同时达到H≤10%和C≤10%，不支持密集缺陷计量放行。
""")
    # TDI formula derivation remains intact; parameter tables/results are refreshed.
    section("### 9.2 为什么256行有时又可能接收更多光学鬼像","### 9.3", """
这是一项另外改变光学接收窗口的条件实验，不是仅切换TDI电荷积分级数：

|项目|25级基准|256级对照|
|---|---|---|
|像方行距假设|5 µm|相同|
|像方接收全高|125 µm|1280 µm|
|物方接收全高|6.25 µm|64 µm|
|固定照明强度FWHM|物方6.25 µm|保持6.25 µm，不扩大|
|挡杆|9根，核心/单侧b=0.10/0.05 mm，齿距0.75 mm|相同|
|实际光学瞳|36 mm，无瞳放大|相同|

当前齿形一级偏移物方14.2 µm、像方0.284 mm。25行窗半高3.125 µm，256行窗半高32 µm，因此后者不仅容纳±1级中心，也容纳±2级中心（±28.4 µm）。高斯照明和衍射没有有限硬边，“窗外”不等于严格零泄漏，必须计算收集权重。

若实机切换级数没有扩大实际光学接收路径，不能直接套用此窗口差异。点响应模型与主报告1 µm照明、有限缺陷优化不是同一计算，应分开引用。
""")
    tdir=ROOT/"results/tdi_pupil36_max10";regdir=ROOT/"results/tdi_charge_pupil36_max10"
    if (tdir/'summary.json').exists():
        t=json.loads((tdir/'summary.json').read_text(encoding='utf-8'))
        ts=[r for r in t['rows'] if r['illumination']=='25stage_FWHM' and r['optical'] in ['all','k0']]
        ref=next(r for r in ts if r['stages']==25 and r['optical']=='all')
        tab=[[r['stages'],r['optical'],f"{r['energy_window']/ref['energy_window']:.4f}",f"{r['scan_fwhm_um']:.3f}",f"{r['scan_ee99_um']:.3f}",f"{100*r['scan_halo_outside_2um_fraction']:.3f}"] for r in ts]
        tbody=table(['接收行数','齿形级次','响应能量/25行全级次','主峰FWHM/µm','含鬼像EE99/µm','扫描外围能量/%'],tab)
        tbody+='\n\n这些包围宽度包含离散光学旁瓣，不是TDI拉长后的缺陷直径。点散射强度图使用其自身声明的共同参考，不与主报告有限缺陷图直接比较绝对亮度。\n\n'
        tbody+=f"详见[25级配置下256级影响计算]({(ROOT/'TDI_25级配置下256级影响_计算结果.md').as_posix()})及[当前点响应原始值]({(tdir/'summary.json').as_posix()})。"
    else:
        tbody="36 mm、9根新参数的TDI点响应尚未生成；此处不复用旧50 mm数值。第9.1节同步电荷不拉长的原理不变。"
    section("### 9.4 现有点响应的条件对照","## 10.",tbody)
    if (regdir/'summary.json').exists():
        reg=json.loads((regdir/'summary.json').read_text(encoding='utf-8'))
        document=document.replace('3.66×10⁻¹⁵',f"{reg['maximum_normalized_error']:.3g}")
    rods=rows('fourier_rod_positions.csv')
    rodtable=table(['杆号','物体级次n','中心Y/mm','核心下/上界mm','包络下/上界mm'],[
        [r['rod_id'],r['object_order_n'],f"{float(r['centre_y_mm']):+.4f}",f"{float(r['core_y_min_mm']):+.4f} / {float(r['core_y_max_mm']):+.4f}",f"{float(r['envelope_y_min_mm']):+.4f} / {float(r['envelope_y_max_mm']):+.4f}"] for r in rods])
    sens=table(['OD/条件','单侧b/mm','最坏S/dB','最坏R/%'],[[r['od'],f"{float(r['b_mm']):.3f}",f"{float(r['worst_background_suppression_db']):.3f}",f"{100*float(r['worst_retention']):.2f}"] for r in rows('coating_depth_sensitivity.csv')])
    vtab=table(['复核','dx/µm','背景能量变化/%','D1能量变化/%'],[[r['tag'],f"{r['dx_um']:.3f}",f"{r['background_energy_change_percent']:+.5f}",f"{r['smallest_defect_energy_change_percent']:+.5f}"] for r in validation['full_grid_checks']])
    section("## 10. 加工、公差与数值可信度","## 11.",f"""
### 10.1 挡杆位置与加工检查尺寸

{image('fourier_rod_positions.png')}

{rodtable}

共9根，中央R5遮挡物体零级。每根沿X跨越光瞳；Y=n×0.5325 mm，选择n=0、±1、±2、±3、±5；**没有n=±4挡杆，也没有图外隐藏的高阶挡杆**。相邻选中杆的最小中心距0.5325 mm，最大1.0650 mm；名义0.20 mm包络对应最小净隙0.3325 mm。

核心全宽0.10 mm、单侧b=0.05 mm、齿距0.75 mm，尖端/谷底20 µm平台。建议按UV熔石英基片上的二值遮光图形询价，不把0.10 mm芯宽当作自由悬空金属棒的机械强度承诺。OD4及零相位仅为计算假设，355 nm透射、损伤阈值、膜层反射/相位、基片波前、固定结构和加工圆角仍须验证。

尺寸不照搬PPT的1.1～4.0 mm包络，因为本物体频谱间距仅0.5325 mm。保留参考齿节尺度，但必须重新进行DFM。输出{link('selected_rods_checkplot.svg')}、{link('selected_rods_checkplot.dxf')}、{link('selected_rod_dimensions.csv')}和{link('fourier_rod_positions.csv')}。蓝色36 mm圆和灰色50 mm圆是辅助线，不是镀膜轮廓；不是可直接下单的放行图纸。

### 10.2 膜层与齿深敏感性

{sens}

增大漏光会同时提高缺陷透过和背景残留，不能只按目标亮度判断优劣。过渡宽度改变也会改变整组复制响应。

### 10.3 数值验证及剩余误差

一维初筛32768点、2.048 mm，dx=0.0625 µm。S初筛={d['worst_background_suppression_db']:.6f} dB；频谱采样加密复核={validation['refined_frequency_sampling']['worst_background_suppression_db']:.6f} dB；物方采样加密复核={validation['refined_object_sampling']['worst_background_suppression_db']:.6f} dB。

{vtab}

表中完整二维能量变化相对4096²、1.024 mm基准：fine=8192²同窗口；wide=8192²、2.048 mm；q5=线高3点改5点。两类主成像图均采用fine，dx=0.125 µm，细于强度的保守Nyquist间隔λ/(4NA)=0.1479 µm；0.25 µm基准只满足复场带宽要求，不作为高频强度的严格收敛证明。

能量对照按共同I_ref计算，参考不随网格或积分点数重新归一。有限视野、周期边缘栅格化、像元积分近似仍可产生误差，不声称任意高精度收敛。背景抑制与保留率优先引用同细网格的比值。

另外校验专利解析/积分、齿形平均、实际瞳径与机械孔分离、显式根数上限、未选中频谱格点不遮挡、快速扫描与逐次FFT传播的一致性、统一强度参考和TDI配准。测试执行结果见本次交付说明。
""")
    section("## 11. 最终设计判断","## 12.",f"""
本轮将实际光瞳固定为36 mm，并在最多10根的约束下选择9根正弦过渡挡杆。完整圆点/长条细网格背景抑制为{supp['dots']:.3f}/{supp['stripes']:.3f} dB；主图强度和显示范围已统一，三处缺陷没有被显示裁掉。

它是本次有限候选族、10 dB示例门槛及公开评分下的折中；原15 dB要求未达到，低halo/低串扰也未同时合格。不能宣称“背景完全滤除、缺陷完全保留且没有相互干扰”，也不能宣称真实Zemax物镜或256级TDI已完成最终优化。

下一步仍需实际相机像元/行距、照明线宽、真实阵列周期分布和允许的外围能量/串扰阈值。若要求更强背景抑制同时更低串扰，需要扩大候选类型后再计算，例如各杆独立宽度或局部二维遮挡；这类改进尚未验证，不作为本版已完成结果。
""")
    document=document.replace('results/ch3_linescan/','results/ch3_pupil36_max10/')
    document=document.replace('results/tdi_charge_transfer/','results/tdi_charge_pupil36_max10/')
    document=document.replace('results/tdi_25_to_256/','results/tdi_pupil36_max10/')
    document=document.replace('|684个候选及条件优选|',f'|{opt["candidate_count"]}个候选及条件优选|')
    document=document.replace('stripes_selected_full_metrics.json','stripes_selected_fine_full_metrics.json')
    document=document.replace('本文是本轮结果的综合快照，若修改参数后重跑，应同步更新本文的表格和结论。',
        '综合报告由pupil36_report.py保留理论章节并自动更新当前结果；先主传播、再TDI，最后运行ch3_report.py同步TDI摘要。')
    document=document.replace('conda run -n fourier-filter python src/tdi_compare.py\nconda run',
        'conda run -n fourier-filter python src/tdi_compare.py\nconda run -n fourier-filter python src/ch3_report.py\nconda run')
    document+='\n\n历史50 mm填充瞳版本仅作追溯：[历史综合报告]('+str((ROOT/'results/ch3_linescan/综合报告_50mm假设历史版.md').as_posix())+')。本版结果目录为'+link('optimization.json')+'所在目录。\n'
    (ROOT/'傅里叶滤波_原理到Python仿真及最优结果_综合报告.md').write_text(document,encoding='utf-8')
    (ROOT/'Ch3中路线扫_仿真结果与设计建议.md').write_text(
        '# Ch3计算结果：36 mm实际光瞳、最多10根挡杆\n\n本轮完整推导、参数、统一强度图、挡杆位置和数值已合并到[综合报告]('
        +(ROOT/'傅里叶滤波_原理到Python仿真及最优结果_综合报告.md').as_posix()+')。\n\n'
        +table(['对象','方案/网格','dx/µm','S/dB','D1保留/%','D2保留/%','D3保留/%'],full)+'\n\n'
        +'使用9根，核心/单侧b=0.10/0.05 mm；实际36 mm瞳，不是50 mm填充瞳。15 dB初筛门槛未达到，现展示10 dB档综合折中。\n\n'
        +image('fourier_rod_positions.png')+'\n\n'+image('stripes_full_scan.png')+'\n',encoding='utf-8')
    save_json(out/'report_summary.json',dict(full_scan_table=full,selected=d,validation=validation,
        intensity_reference=iref,intensity_unit='I/I_ref (dimensionless)',strict_ring_feasible_at_10db=strict))


if __name__=='__main__':
    cfg=m.config()
    build(cfg,json.loads((m.OUT/'optimization.json').read_text(encoding='utf-8')),
          json.loads((m.OUT/'validation_summary.json').read_text(encoding='utf-8')))
