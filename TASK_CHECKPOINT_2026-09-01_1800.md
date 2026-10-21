# 傅里叶滤波任务检查点

保存时间：2026-09-01 17:25（Asia/Shanghai）

## 已完成并保存

- Python 二维相干傅里叶光学滤波前后全视场图：
  `results/dual_deck_engineering/python_2d_before_after.png`
- Python 三个 0.5 µm 目标的滤波前后局部放大图：
  `results/dual_deck_engineering/python_2d_defect_zoom_before_after.png`
- 本机 OpticStudio 2022 R2.02 原生 POP 全窗绝对辐照度图：
  `results/dual_deck_zemax_equivalent_fullwindow/zemax_before_after_absolute.png`
- 同一 Zemax 网格的共同 dB 参考图：
  `results/dual_deck_zemax_equivalent_fullwindow/zemax_before_after_common_db.png`
- Zemax 原始 1024×1024 POP 网格、模型、UDA、指标和参数：
  `results/dual_deck_zemax_equivalent_fullwindow/`
- Python/Zemax 汇总：
  `results/before_after_comparison_summary.json`

## 当前已核实结果

- Python 二维模型：周期背景抑制 10.373 dB；独立 0.5 µm 缺陷能量保留 66.63%。
- Zemax 原生 POP：355 nm、理想 f=30 mm、NA=0.6、20×、10 mm 全窗、1024×1024；7 根档杆背景功率抑制 28.206 dB。
- Zemax 使用的是当前双层硬件映射到物镜 BFP 的等效模型，不是已经完成的真实各向异性中继镜头处方。
- Zemax 缺陷对象是局部条纹段移位，不等同于 Python 的独立 0.5 µm 高斯目标；两种“目标保留率”不能直接比较。

## 18:00 后继续事项

1. 把新生成的 Python 和 Zemax 图片、指标、模型边界写入
   `双层交错傅里叶档杆_工程实现与仿真验证.md`。
2. 增加结果文件与参数的一致性校验并运行现有验证脚本。
3. 最终向用户直接展示滤波前后图，并给出可点击的结果路径。

## 复算入口

```powershell
$env:MPLBACKEND='Agg'
E:\anconda\envs\fourier-filter\python.exe src\plot_before_after_comparison.py
```

Zemax 全窗结果已经完成，不应无理由重复运行。
