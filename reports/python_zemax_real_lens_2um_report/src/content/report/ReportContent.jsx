import React from "react";

import {
  DataComponent, DataTable, MetricCard, ReportSection, RichNarrative, useDataApp,
} from "../../data-app-public.jsx";
import exactImageFigure from "../assets/exact_image_no_transpose_zemax_pop.png";
import frequencyFigure from "../assets/bfp_sampling_comparison.png";

const orderColumns = [
  { field: "order", label: "级次 m" },
  { field: "expectedBfpMm", label: "理论 BFP Y (mm)" },
  { field: "pythonDb", label: "Python (dB)" },
  { field: "zemaxOnDb", label: "真实镜头 POP (dB)" },
  { field: "lossDb", label: "相对损失 (dB)" },
];

const samplingColumns = [
  { field: "surface", label: "传播面" },
  { field: "windowMm", label: "窗口 (mm)" },
  { field: "samplePitchUm", label: "1024²采样间隔 (µm)" },
  { field: "nyquistFor2Um", label: "2 µm Nyquist", presentation: "status" },
];

const metricColumns = [
  { field: "metric", label: "比较量" },
  { field: "pythonDisplay", label: "Python" },
  { field: "zemaxDisplay", label: "真实镜头 Zemax" },
  { field: "validity", label: "解释边界", presentation: "status" },
];

const methodColumns = [
  { field: "method", label: "方法" },
  { field: "lens", label: "镜头模型" },
  { field: "object", label: "物体" },
  { field: "frequencyStatus", label: "频谱状态", presentation: "status" },
  { field: "use", label: "可用范围" },
];

function db(value) {
  return 10 * Math.log10(Math.max(Number(value), 1e-14));
}

function fixed(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function metricDisplay(row, key) {
  const value = row[key];
  if (row.metric.includes("correlation") || row.metric.includes("cosine")) return fixed(value, 6);
  if (row.metric.includes("suppression")) return `${fixed(value, 2)} dB`;
  if (row.metric.includes("retention")) return `${fixed(value, 3)}%`;
  return fixed(value, 6);
}

export function ReportContent() {
  const { reviewedRows, visible, canEdit, mode, appTitle, setAppTitle } = useDataApp();
  const configuration = reviewedRows("case_configuration");
  const orders = reviewedRows("frequency_orders");
  const sampling = reviewedRows("surface_sampling");
  const metrics = reviewedRows("comparison_metrics");
  const methods = reviewedRows("method_scope");
  const firstOrder = orders.find((row) => row.order === 1);
  const startCheck = metrics.find((row) => row.metric.includes("start-surface"));
  const orderRows = orders.map((row) => ({
    ...row,
    expectedBfpMm: fixed(row.expectedBfpMm, 3),
    pythonDb: fixed(db(row.pythonRelativeIntensity), 2),
    zemaxOnDb: fixed(db(row.zemaxResamplingOnRelativeIntensity), 2),
    lossDb: fixed(db(row.zemaxResamplingOnRelativeIntensity) - db(row.pythonRelativeIntensity), 2),
  }));
  const samplingRows = sampling.map((row) => ({
    ...row,
    windowMm: fixed(row.windowMm, 3),
    samplePitchUm: fixed(row.samplePitchUm, 3),
    nyquistFor2Um: row.nyquistFor2Um === "pass" ? "通过" : "失败",
  }));
  const metricRows = metrics.map((row) => ({
    ...row,
    pythonDisplay: metricDisplay(row, "python"),
    zemaxDisplay: metricDisplay(row, "zemax"),
    validity: row.validity
      .replace("valid input check", "有效：输入一致性")
      .replace("diagnostic only; Zemax carrier lost", "仅诊断：载频已丢失")
      .replace("Zemax value invalid", "Zemax数值无效"),
  }));

  return <article className="report-content" aria-label="Python and Zemax comparison report">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"}
        suppressContentEditableWarning
        onBlur={canEdit && mode === "edit" ? (event) =>
          setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>
        {appTitle}
      </h1>
      <RichNarrative id="report:description" className="report-deck" label="编辑报告说明"
        value="同一张未经转置的 2 µm 横条纹三粒子图已经逐像素送入真实六片 Zemax 模型。输入完全一致，但原生逐面 POP 在第一片真实镜面就发生欠采样，因此当前 Zemax 像面只能作为失败诊断，不能用于评价档杆性能。" />
    </header>

    <div className="report-facts" aria-label="关键结论">
      {visible("metric-input") && <MetricCard id="metric-input" title="ZBF输入相关系数"
        queryId="comparison_metrics" sourceRows={metrics}
        value={fixed(startCheck?.zemax, 14)} comparison="与Python原图逐像素一致"
        description="S1强度相关系数；此项有效。" />}
      {visible("metric-first-order") && <MetricCard id="metric-first-order" title="Python一级频谱"
        queryId="frequency_orders" sourceRows={orders}
        value={`${fixed(100 * firstOrder?.pythonRelativeIntensity, 2)}%`}
        comparison="位于理论 ±5.166 mm"
        description="相对零级的X积分频谱强度。" />}
      {visible("metric-zemax-order") && <MetricCard id="metric-zemax-order" title="Zemax一级频谱"
        queryId="frequency_orders" sourceRows={orders}
        value={`${fixed(db(firstOrder?.zemaxResamplingOnRelativeIntensity), 1)} dB`}
        comparison="相对零级；已被数值抹除" negative
        description="这是欠采样结果，不是镜头真实传递。" />}
      {visible("metric-first-failure") && <MetricCard id="metric-first-failure" title="首次失败面"
        queryId="surface_sampling" sourceRows={sampling}
        value="S2 · 8.41 µm" comparison="Nyquist要求 ≤ 1 µm" negative
        description="第一片真实折射面，1024²网格。" />}
    </div>

    {visible("report-conclusion") && <ReportSection id="report-conclusion" title="结论"
      queryId="frequency_orders" queryIds={["frequency_orders", "surface_sampling", "comparison_metrics"]}
      sourceRowsByQuery={{ frequency_orders: orders, surface_sampling: sampling, comparison_metrics: metrics }}
      showHeading={false}>
      <RichNarrative id="report-conclusion:body" className="report-summary-lead" label="编辑结论"
        value={`## 结论

Python结果是本轮可用的数值参考；真实六片镜头的当前原生POP结果不具备性能比较资格。原因不是输入图片、粒子位置或档杆提前遮挡：无杆BFP探针中级次已经消失。1024²网格从S1的0.117 µm间隔跳到S2的8.41 µm，超过2 µm载频所需的1 µm上限约8.4倍。

因此，Zemax给出的35.18 dB“抑制”和0.071%“粒子峰保留”都必须标为无效。它们描述的是数值低通和错误遮挡，不是实际真实镜头性能。`} />
    </ReportSection>}

    {visible("report-case") && <ReportSection id="report-case" title="同一输入与像面结果"
      queryId="case_configuration" queryIds={["case_configuration", "comparison_metrics"]}
      sourceRowsByQuery={{ case_configuration: configuration, comparison_metrics: metrics }}
      showHeading={false}>
      <RichNarrative id="report-case:body" className="report-analysis" label="编辑输入说明"
        value={`## 同一输入与像面结果

- 周期：2 µm；波长：355 nm；NA：0.60；名义焦距：30 mm；放大率：20×。
- 图像不转置，三粒子仍位于物方X = −12、0、+12 µm，Y = 0。
- 档杆按最终工程定义保持长轴Y_F、沿X_F独立定位。
- 横条纹周期沿物方Y，因此该原图主要检查中央杆对Y_F级次线的覆盖，不代表七个X_F杆位全部被利用。

下图B–D采用同一未滤波参考尺度、无背景扣除。C中的平滑场和D中的近黑场均是采样失败现象。`} />
      <DataComponent id="image-comparison" title="原图、Python与真实镜头POP像面"
        queryId="comparison_metrics" kind="custom" sourceRows={metrics}
        description="原始网格显示；Zemax像面已换算到物方坐标。">
        <figure className="report-figure">
          <img src={exactImageFigure} alt="Exact untransposed object, Python reference and real-lens Zemax POP images" />
        </figure>
      </DataComponent>
    </ReportSection>}

    {visible("report-frequency") && <ReportSection id="report-frequency" title="频谱对照"
      queryId="frequency_orders" sourceRows={orders} showHeading={false}>
      <RichNarrative id="report-frequency:body" className="report-analysis" label="编辑频谱分析"
        value={`## 频谱对照

Python输入在理论级次位置保留清楚的离散峰，一级约为零级的19.42%。真实镜头POP开启折射后重采样时，理论一级只剩约8.4×10⁻¹⁰，即约−90.8 dB。关闭重采样会出现约±7.59 mm的强假峰，而理论一级应为±5.166 mm，证明它们是混叠而不是恢复后的真实级次。`} />
      <DataComponent id="frequency-figure" title="Python正确级次与Zemax抹除/混叠结果"
        queryId="frequency_orders" kind="custom" sourceRows={orders}
        description="红线为真实光线给出的理论BFP级次位置。">
        <figure className="report-figure">
          <img src={frequencyFigure} alt="Python spectrum and real-lens POP refraction-resampling comparison" />
        </figure>
      </DataComponent>
      <DataComponent id="frequency-table" title="理论级次位置的相对强度"
        queryId="frequency_orders" kind="table" sourceRows={orders} displayRows={orderRows}
        description="dB均相对各自零级；相对损失为Zemax ON减Python。">
        <DataTable rows={orderRows} columns={orderColumns} searchable={false} />
      </DataComponent>
    </ReportSection>}

    {visible("report-sampling") && <ReportSection id="report-sampling" title="错误从第一片真实镜面开始"
      queryId="surface_sampling" sourceRows={sampling} showHeading={false}>
      <RichNarrative id="report-sampling:body" className="report-analysis" label="编辑采样说明"
        value={`## 错误从第一片真实镜面开始

S1物面满足采样要求；S2窗口从0.12 mm扩到8.61 mm，而点数仍为1024，采样间隔立刻增至8.41 µm。此后所有真实镜面均不满足2 µm载频的Nyquist条件。到40 mm BFP窗口时，若仍要按不大于1 µm采样，单轴至少需要约40000点，简单提高到2048²不能解决。`} />
      <DataComponent id="sampling-table" title="真实六片镜头逐面POP采样"
        queryId="surface_sampling" kind="table" sourceRows={sampling} displayRows={samplingRows}
        description="采样间隔 = 物理窗口 / 1024；通过阈值约为1 µm。">
        <DataTable rows={samplingRows} columns={samplingColumns} searchable={false} />
      </DataComponent>
    </ReportSection>}

    {visible("report-metrics") && <ReportSection id="report-metrics" title="数值汇总与有效性"
      queryId="comparison_metrics" sourceRows={metrics} showHeading={false}>
      <RichNarrative id="report-metrics:body" className="report-analysis" label="编辑指标说明"
        value={`## 数值汇总与有效性

只有输入相关系数能够作为有效结论。未滤波形状相似度可用于观察采样损失，但不能代表真实镜头像质；所有Zemax过滤性能指标均被频谱丢失污染。Python过滤值仍属于原Python模型自身，并且其旧档杆方向与最终工程档杆方向不同，不能与当前Zemax过滤值直接做硬件优劣比较。`} />
      <DataComponent id="metrics-table" title="Python与真实镜头Zemax指标"
        queryId="comparison_metrics" kind="table" sourceRows={metrics} displayRows={metricRows}
        description="无效值保留用于审计，但不进入性能结论。">
        <DataTable rows={metricRows} columns={metricColumns} searchable={false} />
      </DataComponent>
    </ReportSection>}

    {visible("report-methods") && <ReportSection id="report-methods" title="与此前Zemax结果的关系"
      queryId="method_scope" sourceRows={methods} showHeading={false}>
      <RichNarrative id="report-methods:body" className="report-caveat" label="编辑方法边界"
        value={`## 与此前Zemax结果的关系

此前清楚显示七个级次的结果不能与本轮真实六片原生POP混为一类：理想4f结果使用单个近轴物镜，属于理想上限；真实镜头清晰频谱图使用Zemax真实光线/BFP映射/OPD加Python FFT，属于联合模型。旧的真实镜头native POP图本身也是宽零级包络，图中级次线是理论位置标记而非解析出的尖峰。`} />
      <DataComponent id="method-table" title="四类仿真结果的适用范围"
        queryId="method_scope" kind="table" sourceRows={methods} displayRows={methods}
        description="对象或传播引擎不同的结果不作为同条件直接对照。">
        <DataTable rows={methods} columns={methodColumns} searchable={false} />
      </DataComponent>
    </ReportSection>}
  </article>;
}
