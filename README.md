# 显微系统傅里叶滤波

## 最终报告

[双层交错傅里叶档杆：最终工程方案与验证结果](双层交错傅里叶档杆_工程实现与仿真验证.md)

最终方案参数：355 nm、30 mm傅里叶物镜、NA=0.6、36 mm真实光学瞳、20×成像；每通道安装9根双层锯齿杆，默认启用7根。档杆核心宽1.0 mm、齿距0.75 mm。A/B层单侧锯齿深度在0.5–2.0 mm的49组联合扫描中，最优值均为0.5 mm。各杆沿傅里叶面X独立移动并与实测频谱峰对准；档杆长轴和锯齿周期沿傅里叶面Y。

25级TDI沿Y只接收锯齿零级；它不能消除沿相机长边X展开的多杆振铃。滤波前后结果全部采用同一原始强度标尺，不进行周期背景相减。

## 最终结果入口

- [最终工程方案与验证](双层交错傅里叶档杆_工程实现与仿真验证.md)
- [三通道正弦物镜与倾斜照明频谱推导](三通道傅里叶滤波_原理思路.md)
- [频谱对准和TDI积分宽度判据](TDI与面阵_挡杆衍射级次选择.md)
- [最终相机坐标下的Ch3仿真](更新相机坐标_Ch3倾斜照明仿真.md)
- [真实物镜Zemax建模与验证](真实物镜傅里叶滤波_建模与验证.md)
- [当前355 nm目标参数的Zemax理想透镜原生POP](Zemax_理想透镜_2um周期particle原生POP.md)
- [355 nm Corning 7980真实物镜与阿贝正弦验证](Zemax_355nm_Corning7980真实物镜_阿贝验证.md)
- [355 nm真实物镜：阿贝、频谱线性与傅里叶滤波闭环验证](Zemax_355nm真实物镜_阿贝频谱线性与滤波验证.md)

## Python环境

项目使用独立Anaconda环境 `fourier-filter`。环境定义见 [environment.yml](environment.yml)。

```powershell
conda env create --name fourier-filter --file .\environment.yml
conda run --name fourier-filter python .\src\ch3_confirmed_geometry_rod_tradeoff.py
```

主要结果位于：

- `results/ch3_confirmed_geometry_rod_tradeoff/`
- `results/ch3_oblique75_confirmed_geometry/`
- `results/small_period_point_defects/`
- `results/dual_deck_engineering/`
- `results/real_objective/`
- `results/zemax_current_ideal_particle/`
- `results/uv_real_objective/`

早期参数试算、旧坐标图和诊断过程不作为最终方案依据。
