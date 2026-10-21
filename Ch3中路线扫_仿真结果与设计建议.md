# Ch3计算结果：36 mm实际光瞳、最多10根挡杆

本轮完整推导、参数、统一强度图、挡杆位置和数值已合并到[综合报告](D:/work/git_share/Fourier/傅里叶滤波_原理到Python仿真及最优结果_综合报告.md)。

|对象|方案/网格|dx/µm|S/dB|D1保留/%|D2保留/%|D3保留/%|
|---|---|---|---|---|---|---|
|dots|selected_fine|0.125|10.437|87.51|79.28|73.25|
|stripes|selected_fine|0.125|10.338|87.51|79.28|73.25|
|dots|background_only|0.250|10.611|83.09|71.94|63.92|
|dots|best_hard|0.250|10.210|88.31|80.59|75.09|

使用9根，核心/单侧b=0.10/0.05 mm；实际36 mm瞳，不是50 mm填充瞳。15 dB初筛门槛未达到，现展示10 dB档综合折中。

![](D:/work/git_share/Fourier/results/ch3_pupil36_max10/fourier_rod_positions.png)

![](D:/work/git_share/Fourier/results/ch3_pupil36_max10/stripes_full_scan.png)
