"""Keep the 25-stage optical/illumination setup; enlarge only TDI acceptance.

Exact ideal synchronised TDI response for point scatterers under fixed Gaussian
line illumination. Coherent tooth orders are retained BEFORE intensity; TDI
integrates photocharge, not optical amplitude. All lengths mm.
"""
from pathlib import Path
import json
import gc
import numpy as np
from scipy import special,ndimage
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import ch3_linescan as m
from ideal_na06 import save_json,save_csv

ROOT=m.ROOT;OUT=ROOT/"results"/"tdi_pupil36_max10"


def config():
    return yaml.safe_load((ROOT/"config/tdi_25_to_256.yml").read_text(encoding="utf-8"))


def gate(x,height,width):
    """Integral of Gaussian illumination intensity over TDI height.

    Divided by infinite-window illumination energy, so lies in [0,1].
    It is an optical collection fraction, NOT amplitude transmission or an
    N-row motion-blur kernel. It must not be used to stretch output coordinates.
    """
    q=2*np.sqrt(np.log(2))/width
    return .5*(special.erf(q*(height/2-x))+special.erf(q*(height/2+x)))


def tooth_coefficient(t,k,leak):
    c=(1-np.sqrt(leak))*t*np.sinc(k*t)
    return np.where(np.asarray(k)==0,c+np.sqrt(leak),c)


def optical_psf(cfg,tcfg,design,kmax,tag):
    n=tcfg["numerics"]["n"];window=tcfg["numerics"]["window_mm"];dx=window/n
    nu=np.fft.fftshift(np.fft.fftfreq(n,dx)).astype(np.float32)
    sc=m.scales(cfg);ff=nu*sc["alpha_mm2"]
    t=m.mean_transmission(ff,cfg,design,leak=0).astype(np.float32)
    leak=cfg["filter"]["blocked_intensity_transmission"]
    transmission=np.broadcast_to(tooth_coefficient(t,0,leak)[:,None],(n,n)).copy()
    for k in range(1,kmax+1):
        # ±k have the same real coefficient for the centred transparent strip.
        phase=np.cos(2*np.pi*k*ff/cfg["filter"]["tooth_pitch_mm"]).astype(np.float32)
        transmission+=2*tooth_coefficient(t,k,leak)[:,None]*phase[None,:]
    if tag=="none":transmission[:]=1
    # Form the pupil rowwise to avoid an additional n-by-n temporary.
    for j in range(n):transmission[j,nu**2+nu[j]**2>sc["cutoff_per_mm"]**2]=0
    h=m.ifftc(transmission)
    del transmission
    axis=(np.arange(n)-n//2)*dx
    ix=np.flatnonzero(abs(axis)<=tcfg["numerics"]["crop_half_x_mm"])
    iy=np.flatnonzero(abs(axis)<=tcfg["numerics"]["crop_half_y_mm"])
    result=h[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1].copy()
    del h;gc.collect()
    np.savez_compressed(OUT/f"optical_{tag}.npz",x=axis[ix],y=axis[iy],field=result,kmax=kmax)
    print(f"Saved optical PSF {tag}, orders -{kmax}..{kmax}",flush=True)
    return result,axis[ix],axis[iy]


def pixel_integrate(a):
    # dx=0.125 um, 0.25 um pixels: centred trapezoid in each coordinate.
    return ndimage.convolve1d(ndimage.convolve1d(a,[.25,.5,.25],axis=0,mode="constant"),[.25,.5,.25],axis=1,mode="constant")


def width_metrics(axis,intensity):
    center=len(axis)//2;i=intensity/max(intensity[center],1e-30)
    j=center+np.flatnonzero(i[center:]<.5)[0]
    half=np.interp(.5,[i[j],i[j-1]],[axis[j],axis[j-1]])
    order=np.argsort(abs(axis));cum=np.cumsum(intensity[order]);ee=[]
    for f in [.8,.9,.95,.99]:
        ee.append(2*abs(axis[order[np.searchsorted(cum,f*cum[-1])]])*1000)
    return dict(fwhm_um=float(2*half*1000),ee80_um=float(ee[0]),ee90_um=float(ee[1]),ee95_um=float(ee[2]),ee99_um=float(ee[3]))


def point_pair_bound(h,x,y,height,width,dx,dy):
    # Exact point-scatterer cross term after synchronised TDI integration.
    cy=len(y)//2;cx=len(x)//2
    from scipy.interpolate import RegularGridInterpolator
    re=RegularGridInterpolator((y,x),h.real,bounds_error=False,fill_value=0)
    im=RegularGridInterpolator((y,x),h.imag,bounds_error=False,fill_value=0)
    hd=complex(float(re([[dy,dx]])[0]),float(im([[dy,dx]])[0]));h0=h[cy,cx]
    own=abs(h0)**2*gate(0,height,width)
    incoherent=abs(hd)**2*gate(dx,height,width)/own
    cross=2*abs(hd*h0)*np.exp(-np.log(2)*dx*dx/(width*width))*gate(dx/2,height,width)/own
    return dict(separation_scan_um=1000*dx,separation_line_um=1000*dy,
                neighbour_intensity_fraction=float(incoherent),coherent_cross_bound=float(cross),
                worst_intensity_change_fraction=float(incoherent+cross))


def calculate():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=m.config();tcfg=config()
    if tcfg["tdi"].get("image_rows_per_clock",1)!=tcfg["tdi"].get("charge_rows_per_clock",1):
        raise ValueError("This model assumes perfectly matched image and charge motion.")
    design=json.loads((m.OUT/"optimization.json").read_text(encoding="utf-8"))["choices"]["selected"]
    for tag,k in [("none",0),("k0",0),("k4",tcfg["numerics"]["check_max_tooth_order"]),("all",tcfg["numerics"]["max_tooth_order"])]:
        optical_psf(cfg,tcfg,design,k,tag)
    z=np.load(OUT/"optical_all.npz");h=z["field"];x=z["x"];y=z["y"]
    h0=np.load(OUT/"optical_k0.npz")["field"]
    h4=np.load(OUT/"optical_k4.npz")["field"]
    hn=np.load(OUT/"optical_none.npz")["field"]
    wide=json.loads(json.dumps(tcfg));wide["numerics"]["window_mm"]*=2
    hw,xw,yw=optical_psf(cfg,wide,design,tcfg["numerics"]["max_tooth_order"],"all_wide")
    reference=h[::2,::2];hw=hw/4
    mask=(abs(xw[None,:])<.030)&(abs(yw[:,None])<.025)
    check=dict(frequency_grid_refinement_relative_complex_field_L2=float(np.linalg.norm((hw-reference)[mask])/np.linalg.norm(reference[mask])),
               physical_field_scale_correction=.25,comparison_x_half_um=30,comparison_y_half_um=25,
               note="Compare same physical coordinates, original dx=.125 um subsampled to .25 um. Frequency resolution only, not pixel integration convergence.")
    save_json(OUT/"frequency_grid_check.json",check)
    dx=x[1]-x[0];delta=m.scales(cfg)["tooth_first_order_shift_um"]/1000
    mag=cfg["optics"]["magnification"];tp=tcfg["tdi"];width=tp["illumination_intensity_fwhm_object_mm"]
    heights={n:n*tp["row_pitch_image_mm"]/mag for n in [tp["baseline_stages"],tp["comparison_stages"]]}
    rows=[];orders=[];pairs=[];maps={};convergence=[]
    for n,height in heights.items():
        for illumination_label,w in [("25stage_FWHM",width),("previous_1um",tp["narrow_line_sensitivity_fwhm_object_mm"])]:
            g=gate(x,height,w)
            for tag,field in [("none",hn),("k0",h0),("all",h)]:
                a=pixel_integrate(abs(field)**2*g[None,:])
                if illumination_label=="25stage_FWHM":maps[f"{n}_{tag}"]=a
                xprofile=a.sum(axis=0);yprofile=a.sum(axis=1)
                total=a.sum();core=abs(x)<.002
                r=dict(stages=n,illumination=illumination_label,optical=tag,height_image_mm=n*tp["row_pitch_image_mm"],
                       illumination_fwhm_object_um=1000*w,central_collection_fraction=float(gate(0,height,w)),
                       energy_window=float(total*dx*dx),scan_halo_outside_2um_fraction=float(a[:,~core].sum()/total),
                       central_intensity=float(a[len(y)//2,len(x)//2]),
                       **{"scan_"+k:v for k,v in width_metrics(x,xprofile).items()},
                       **{"line_"+k:v for k,v in width_metrics(y,yprofile).items()})
                midline=a[len(y)//2]
                for threshold in [.01,.001]:
                    included=abs(x[midline>=threshold*midline[len(x)//2]])
                    r[f"scan_threshold_{threshold:g}_full_span_um"]=float(2000*included.max())
                rows.append(r)
                if tag=="all":
                    for k in range(-2,3):
                        ids=abs(x-k*delta)<.002
                        orders.append(dict(stages=n,illumination=illumination_label,k=k,
                            shift_object_um=1000*k*delta,shift_image_mm=mag*k*delta,
                            illumination_collection_fraction=float(gate(k*delta,height,w)),
                            band_energy_fraction=float(a[:,ids].sum()/total)))
                    for sx,sy in [(delta,0),(0,.020),(delta,.020)]:
                        pairs.append(dict(stages=n,illumination=illumination_label,**point_pair_bound(h,x,y,height,w,sx,sy)))
        g=gate(x,height,width)
        a4=pixel_integrate(abs(h4)**2*g[None,:]);a8=maps[f"{n}_all"]
        convergence.append(dict(stages=n,k4_to_k8_total_energy_change_percent=float(100*(a8.sum()/a4.sum()-1)),
            normalized_l1_intensity_difference=float(np.sum(abs(a8-a4))/np.sum(a8))))
    np.savez_compressed(OUT/"tdi_point_responses.npz",x_um=x*1000,y_um=y*1000,**maps)
    save_csv(OUT/"point_response_metrics.csv",rows);save_csv(OUT/"order_collection.csv",orders);save_csv(OUT/"point_pair_metrics.csv",pairs)
    save_json(OUT/"summary.json",dict(config=tcfg,optics=cfg["optics"],filter=design,rows=rows,orders=orders,pairs=pairs,
               convergence=convergence,delta_object_um=delta*1000,delta_image_mm=delta*mag,
               note="Exact point-scatterer synchronised-TDI model; no claim of finite-defect/full-pattern validation. Crop +/-60 um scan, +/-50 um line."))
    print(json.dumps(dict(rows=rows,convergence=convergence),indent=2),flush=True)


def figures_and_report():
    s=json.loads((OUT/"summary.json").read_text(encoding="utf-8"));z=np.load(OUT/"tdi_point_responses.npz")
    x=z["x_um"];y=z["y_um"];cfg=m.config();tp=s["config"]["tdi"]
    from tdi_charge_transfer import run as verify_charge_transfer, OUT as registration_out
    verify_charge_transfer()
    registration=json.loads((registration_out/"summary.json").read_text(encoding="utf-8"))
    s["interpretation_revision"]={
        "motion_model":"Perfectly matched charge transfer; no N-dependent motion blur.",
        "existing_scan_halo_and_ee_metrics":"Optical PSF plus changed collection window, NOT TDI scan deformation.",
        "registration_validation":"../tdi_charge_pupil36_max10/summary.json",
        "spatial_plot_aspect":"equal"}
    save_json(OUT/"summary.json",s)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for n,color in [(25,"#087e8b"),(256,"#bc4a32")]:
        height=n*tp["row_pitch_image_mm"]/cfg["optics"]["magnification"]
        axs[0].plot(x,gate(x/1000,height,tp["illumination_intensity_fwhm_object_mm"]),color=color,label=f"{n}-row optical window")
        a=z[f"{n}_all"];norm=z["25_all"].sum(axis=0).max()
        axs[1].semilogy(x,np.maximum(a.sum(axis=0)/norm,1e-10),color=color,label=f"{n}-row window, all tooth orders")
    axs[1].semilogy(x,np.maximum(z["256_k0"].sum(axis=0)/norm,1e-10),"--",color="#777777",label="256-row window, k=0 only")
    for ax in axs:
        for k in [-2,-1,0,1,2]:ax.axvline(k*s["delta_object_um"],color="#888888",lw=.7,alpha=.4)
        ax.set(xlim=(-50,50),xlabel="Scan-direction object-equivalent offset (um)");ax.legend(fontsize=8);ax.grid(alpha=.2)
    axs[0].set(ylabel="Fraction of fixed illumination collected",title="Optical window weighting\n(not a motion-blur kernel)")
    axs[1].set(ylabel="Line-integrated optical response / 25-row peak",ylim=(1e-9,2),title="Collected optical diffraction response\n(not TDI motion smear)")
    fig.savefig(OUT/"tdi_acceptance_and_ghosts.png",dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    peak=z["25_all"].max()
    for ax,key,title in zip(axs,["25_all","256_all","256_k0"],["25-row optical window\nall calculated orders","256-row optical window\nall calculated orders","256-row optical window\nk=0-only reference"]):
        im=ax.imshow(np.maximum(z[key]/peak,1e-10),origin="lower",extent=[x[0],x[-1],y[0],y[-1]],
                     cmap="magma",norm=LogNorm(1e-6,2),aspect="equal")
        ax.set(xlim=(-45,45),ylim=(-45,45),xlabel="x scan (um)",ylabel="y camera-line (um)",title=title)
    fig.colorbar(im,ax=axs,label="Intensity / 25-stage central peak",shrink=.8)
    fig.savefig(OUT/"tdi_point_maps.png",dpi=180);plt.close(fig)
    def table(headers,rows):return "\n".join(["|"+"|".join(headers)+"|","|"+"|".join(["---"]*len(headers))+"|"]+["|"+"|".join(map(str,r))+"|" for r in rows])
    main=[r for r in s["rows"] if r["illumination"]=="25stage_FWHM"]
    ref=next(r for r in main if r["stages"]==25 and r["optical"]=="all")
    metrics=[[r["stages"],r["optical"],f'{r["energy_window"]/ref["energy_window"]:.4f}',
        f'{100*r["scan_halo_outside_2um_fraction"]:.3f}',f'{r["scan_fwhm_um"]:.3f}',f'{r["scan_ee90_um"]:.3f}',f'{r["scan_ee99_um"]:.3f}',f'{r["scan_threshold_0.001_full_span_um"]:.3f}'] for r in main]
    orderrows=[[r["stages"],r["k"],f'{r["shift_object_um"]:.3f}',f'{100*r["illumination_collection_fraction"]:.6f}',f'{100*r["band_energy_fraction"]:.4f}'] for r in s["orders"] if r["illumination"]=="25stage_FWHM"]
    pairrows=[[r["stages"],f'{r["separation_scan_um"]:.3f}',f'{r["separation_line_um"]:.3f}',f'{100*r["neighbour_intensity_fraction"]:.4f}',f'{100*r["worst_intensity_change_fraction"]:.4f}'] for r in s["pairs"] if r["illumination"]=="25stage_FWHM"]
    icon=lambda name:f'![]({(OUT/name).as_posix()})'
    r256=next(r for r in main if r["stages"]==256 and r["optical"]=="all")
    txt=rf'''# 25级/256级TDI：同步积分无扫描拖尾与光学接收对照

## 结论

**修正结论：像面运动与电荷转移严格匹配时，同一物点的光生电荷逐级配准累加。TDI积分本身不把图像拉长，不按25行或256行的高度对图像做空间平均。光学输入、像元和有效曝光条件相同时，归一化成像应与单行线扫一致。** [Hamamatsu官方同步积分说明](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/tdi-imaging-an-efficient-aoi-and-axi-tool.html)

本版已改为实际36 mm光瞳、9根挡杆并重新传播。历史50 mm填充瞳版曾将“256级39 µm的99%包围宽度”作为扫描方向变宽展示；**不能把含光学鬼像的包围宽度解释为TDI拖尾，且旧数值不适用于当前参数**。主像、光学外围响应和电荷运动失配分开评价。

二维图原先的部分`aspect="auto"`也会把相同物理尺寸按不同横纵比例画出。现改为**横纵等比例**，主像尺寸、光学旁瓣/鬼像、速度失配拖尾三者分开评价。

## 0. 逐时钟电荷转移验证：同步时不拉长

令r为TDI物理行、j为相机长方向像元、n为时钟，电荷更新为

$$
Q_n(r,j)=Q_{{n-1}}(r-1,j)+w_r I_n(r,j).
$$

图像每时钟正好移动一行，$I_n(r,j)=I_0(r-n,j)$。电荷也移一行，因此在充满积分流水线后

$$
Q_n(N-1,j)=\left(\sum_{{r=0}}^{{N-1}}w_r\right)I_0(N-1-n,j).
$$

输出坐标是配准后的物点坐标，不是把N个不同位置的像素拼在一起。除以增益$\sum w_r$后得到原图；等曝光时增益为N。固定窄照明的各行权重可以不同，但在本项**光学输入形状保持相同、各行仅曝光权重变化**的验证中，不改变归一化几何形状。若照明和接收窗口另外改变了光学传递，必须另项分析，不能归咎于电荷转移。

新增脚本逐步执行“图像移一行→电荷移一行→累加→末行输出”，覆盖流水线预填充、边缘和非均匀曝光。对规定强度图和先前光学点像分别检查1、25、256级，共12组；最大归一化像素误差 **{registration["maximum_normalized_error"]:.3g}**（浮点舍入量级），新增运动拖尾为0像素。

![]({(registration_out/"matched_tdi_images.png").as_posix()})

![]({(registration_out/"matched_tdi_profiles.png").as_posix()})

代码：[tdi_charge_transfer.py]({(ROOT/"src/tdi_charge_transfer.py").as_posix()})；[逐配置验证数值]({(registration_out/"registration_validation.csv").as_posix()})。

## 后续各节的适用范围：光学接收窗口另外变化

以下保留“以25级的照明和滤片为基准，把光学有效接收区全高由125 µm扩大到1280 µm”的独立条件实验。当前尺度可以额外接收挡杆±1和±2级附近的光学响应，但**不是只改变TDI积分级数、保持同一光学输入的成像对比**。如果实际硬件只是切换积分级数，而光学窗口/有效光信号路径没有如此改变，不能直接套用下面的收集差异。

这是含光学接收窗口的**点散射体响应验证**，不是已经重新完成全部有限尺寸缺陷、二维周期对象和相机噪声的优化。先前完整Ch3报告仍是满足零级空间选择时的条件基线，不能作为具体256级配置的最终性能。

## 1. 参数与保持不变的条件

|项目|25级基准|256级对照|
|---|---|---|
|波长/NA/物镜焦距|355 nm / 0.6 / 30 mm|相同|
|实际光学瞳 / 机械开口|36 mm / 50 mm，不做瞳放大|相同|
|总倍率|20×|相同|
|TDI行距假设|5 µm|5 µm|
|像方有效接收全高|125 µm|1280 µm|
|物方有效接收全高|6.25 µm|64 µm|
|固定高斯照明强度FWHM|物方6.25 µm，像方125 µm|仍为相同宽度，不扩大|
|挡杆|{s['filter']['rod_count']}根；核心全宽{s['filter']['core_width_mm']:.2f} mm、单侧正弦过渡{s['filter']['b_mm']:.2f} mm、齿距0.75 mm|完全相同|
|遮光/齿尖|OD4零相位假设、20 µm平台|相同|

5 µm行距和照明FWHM尚非用户提供的设备实测值。这里以“照明FWHM匹配25级投影高度”定义25级光学基准；另保留此前物方1 µm窄线作敏感性对照。若实际25级照明不同，只需改配置重算，不能把当前数值不加换算地用于真实芯片。

由于照明没有扩大，**不应预期256/25=10.24倍有效曝光增益**。TDI核心信号收集比例还取决于25级窗口原先截去了多少照明尾部，而额外收集的鬼响应不属于有用信号增益。

## 2. 正确的同步积分模型

使用固定在实验室坐标中的照明 $A(x)=\exp[-2\ln2\,x^2/w^2]$，物体相对它平移。令 u 为某TDI积分行的像面位置（以下换算到物方），ξ为经过运动配准后的输出物方坐标，h为**包含挡杆全部所算级次**的相干脉冲响应。相同输出物点对应的场为

$$
E_u(\xi,y)=\iint t(\xi-z,y')A(u-z)h(z,y-y')\,\mathrm dz\mathrm dy'.
$$

理想同步积分输出

$$
Q_N(\xi,y)\propto\int_{{-H_N/2}}^{{H_N/2}}|E_u(\xi,y)|^2\,\mathrm du.
$$

这是**运动配准后累加强度/电荷**，不是未配准地对多行求平均，也不是先把场在各行相加再平方。假设连续理想同步、行间无间隙；另用0.25 µm物方像元积分核近似采样，未包含实际电荷转移效率、饱和、行间死区、读噪和同步误差。

点散射体 $t=\delta$ 时，有精确的简化式：

$$
\boxed{{Q_N(\xi,y)=|h(\xi,y)|^2G_N(\xi)}},\qquad
G_N(\xi)=\int_{{-H_N/2}}^{{H_N/2}}|A(u-\xi)|^2\,\mathrm du.
$$

把G除以无限接收窗口时的总照明能量，得到

$$
G_N(\xi)=\frac12\left[\operatorname{{erf}}\!\left(\frac{{2\sqrt{{\ln2}}}}{{w}}(H_N/2-\xi)\right)+
\operatorname{{erf}}\!\left(\frac{{2\sqrt{{\ln2}}}}{{w}}(H_N/2+\xi)\right)\right].
$$

这个G是运动同步积分的**空间收集比例**，不是额外乘到光场上的振幅滤片。它直接说明为什么扩大的接收区会容纳偏移鬼响应。

## 3. 保留挡杆非零级

透明占空比 t(Y)的锯齿展开为

$$
T(X,Y)=c_0(Y)+2\sum_{{k=1}}^{{K}}c_k(Y)\cos(2\pi kX/p_s),
$$

$$
c_0=\sqrt\eta+(1-\sqrt\eta)t,\qquad c_k=(1-\sqrt\eta)t\operatorname{{sinc}}(kt).
$$

程序先计算 $h=\mathcal F^{{-1}}[PT]$，**在场层面保留所有级次的干涉**，之后才计算强度和TDI积分。采用K=8，并与K=4对照；有限K是收敛近似，不将其称作无限精确二值边缘。

第k级在像面偏移 $k\times{s['delta_image_mm']:.6f}$ mm，物方为 $k\times{s['delta_object_um']:.4f}$ µm。25级接收半高只有3.125 µm；256级接收半高32 µm，因此±1级和±2级的中心都进入256行接收窗口。高斯照明及衍射存在尾部，实际泄漏仍按积分计算，不按几何窗作严格零截断。

{icon("tdi_acceptance_and_ghosts.png")}

## 4. 级次收集与点响应数值

下表“收集比例”仅描述位于该级次中心的照明包络被TDI窗口收集的比例；“能量占比”则来自完整相干PSF的数值积分。两者不是同一个量。能量带取各级中心扫描方向±2 µm、沿线±50 µm；有重叠尾部时不把此带积分称作严格正交级次功率。

{table(["TDI级数","k","物方移位/µm","中心照明收集/%","该±2µm带占总响应/%"],orderrows)}

{icon("tdi_point_maps.png")}

统一归一化到25行光学窗口、保留全部级次的点响应总能量。光学外围能量定义为扫描方向目标中心±2 µm之外的能量；EE90/99来自沿相机长方向积分后的边缘分布，全宽而非半径。**下表不是TDI运动拖尾表，包围宽度包含离散鬼像，不等于主像尺寸。**

{table(["接收窗行数","光学级次","总响应/25行窗","光学外围能量/%","主峰FWHM/µm","含鬼像EE90/µm","含鬼像EE99/µm","0.1%阈值全跨度/µm"],metrics)}

0.1%阈值全跨度取穿过目标中心的扫描方向剖面，统计超过本图中心强度0.1%的最外两点之间距离；它包括不连通的鬼像斑点，不是连通目标的直径，也不是已经考虑噪声后能被相机检测到的实测尺寸。其作用是避免FWHM/EE90对弱鬼像不敏感时漏报风险。

256级全部级次模型的扫描方向halo为 **{100*r256["scan_halo_outside_2um_fraction"]:.3f}%**，25级为 **{100*ref["scan_halo_outside_2um_fraction"]:.3f}%**。这反映了额外鬼响应，并不等价于随机噪声。将256级错误地继续按k=0计算，会漏掉图中扫描方向的偏移能量。

## 5. 小缺陷之间的串扰：TDI不是简单相干相加

对两个点散射体，扫描方向间距dₓ、沿线间距dᵧ，TDI会在各时刻先产生相干强度，再累积。高斯照明下，两点场的跨时间积分权重带有

$$
\exp[-\ln2\,d_x^2/w^2]\,G_N(\xi-d_x/2).
$$

因此，沿扫描方向相距约{s['delta_object_um']:.2f} µm的两个点，在6.25 µm线照明下同时受照的重叠较小，相干交叉项被压低；**但邻点鬼像的强度仍可能在不同时间被积入相同输出位置，不会因为不同时照明就自动消失**。沿线方向两个点则可同时受照，保留较强相干串扰。

下表给出邻点中心位置、相对本点单独成像中心信号的干扰；相位上界在TDI积分后计算，不是套用单次全场相干的2r+r²。

{table(["级数","扫描间距/µm","沿线间距/µm","邻点强度串入/%","最坏相位总变化/%"],pairrows)}

以上点响应可用于诊断亚分辨缺陷的串扰机制，不代表已计算0.5～3 µm所有有限形貌的定量尺寸误差。后者须在上述同步积分算子中放入有限物体，不能把点PSF强度直接与相干物体强度卷积。

## 6. 设计后果与验证

若照明仍按25级配置保持不变，有三个待选择的方向：保持25级有效接收区；为256级增加实际光学接收窗口并核验读出；或缩小齿距把±1级推到256级窗口外。第三种可先用 $\Delta x_i>(H+W)/2$ 估算，再按允许泄漏量计算。

对于256×5 µm的1.280 mm接收高，若按固定高斯照明约99%能量全宽273.5 µm计，几何上要求一级像移大于约0.7768 mm。当前 $M\alpha=0.213000\ \mathrm{{mm}}^2$，对应齿距小于约 **0.274 mm**（还没有加系统衍射与装调余量）。它只是一条初筛界限，**并未在本轮改动0.75 mm齿距重新优化**。

K=4到K=8的截断核验结果：

{table(["级数","总响应能量变化/%","归一化强度L1差"],[[r["stages"],f'{r["k4_to_k8_total_energy_change_percent"]:.5f}',f'{r["normalized_l1_intensity_difference"]:.6g}'] for r in s["convergence"]])}

8192²网格、1.024 mm窗口、0.125 µm物方采样；结果截取扫描±60 µm、沿线±50 µm。报告的能量分数以此窗口为分母，不是无限域总能量。本轮对比改变TDI区域的效应，不声称所有网格/有限齿长误差已经消除。

另将FFT窗口扩大到2.048 mm，在相同物理采样点比较扫描±30 µm、沿线±25 µm区域的复场，细化傅里叶面采样后的相对L2差约 **{100*json.loads((OUT/"frequency_grid_check.json").read_text(encoding="utf-8"))["frequency_grid_refinement_relative_complex_field_L2"]:.3f}%**。这是频谱采样检查，不是实际相机像元积分精度的保证。

文件：[配置]({(ROOT/"config/tdi_25_to_256.yml").as_posix()})；[计算与报告脚本]({(ROOT/"src/tdi_compare.py").as_posix()})；[全部数值]({(OUT/"summary.json").as_posix()})。

```powershell
& E:/anconda/envs/fourier-filter/python.exe src/tdi_compare.py
```

原理来源：挡杆高阶落在感光区外才可只选零级，见[US7397557B2 Fig.4](https://patents.google.com/patent/US7397557B2/en)；TDI多行运动同步累积机制，见[Hamamatsu官方说明](https://hub.hamamatsu.com/us/en/technical-notes/image-sensors/tdi-imaging-an-efficient-aoi-and-axi-tool.html)。所有25/256数值为本项目独立计算，不是厂家实测规格。
'''
    (ROOT/"TDI_25级配置下256级影响_计算结果.md").write_text(txt,encoding="utf-8")


if __name__=="__main__":
    calculate();figures_and_report()
