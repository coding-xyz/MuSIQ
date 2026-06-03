# Report Notebook Visualization Inventory

## Goal

整理 `report/` 下 notebook 里现有的绘图代码，归并成可复用的 `src/musiq/visualization` 画图接口。目标接口以 `plot_*(ax, ...)` 为基础，不直接改 notebook 文件。

## Current Notebook Categories

### 1. Pulse / waveform plots

- `report/task1_single_qubit_rabi/task1_single_qubit_rabi.ipynb`
  - 单通道 `I/Q` 脉冲包络折线图
- `report/compile_test/Bell_circuti.ipynb`
  - 与 `task1` 基本相同的单通道脉冲包络图
- `report/task3_gaussian_drag_comparison/task3_gaussian_drag_comparison.ipynb`
  - Gaussian / DRAG 波形对比
  - `I/Q` 双线叠加，带零线、标题、图例
- `report/task8_dynamical_decoupling/task8_dynamical_decoupling.ipynb`
  - 多通道脉冲序列堆叠图
  - 每条通道按行错开，`I/Q` 分别画实线/虚线
- `report/task9_small_circuit_examples/task9_small_circuit_examples.ipynb`
  - 多通道 pulse sequence 堆叠图
  - 与 `task8` 风格接近
- `report/task6_single_qubit_readout/task6_single_qubit_readout.ipynb`
  - 2x2 readout 过程图，属于 readout waveform / response 视图

### 2. Population / trajectory time series

- `report/task1_single_qubit_rabi/task1_single_qubit_rabi.ipynb`
  - `P0/P1/P2` 随时间演化
- `report/compile_test/Bell_circuti.ipynb`
  - 同类 population trace
- `report/task2_single_qubit_decoherence/task2_single_qubit_decoherence.ipynb`
  - `T1/T2/Ramsey/Echo` 的 population 或 coherence trace
- `report/task8_dynamical_decoupling/task8_dynamical_decoupling.ipynb`
  - 多序列 `P1` 演化对比
- `report/task9_small_circuit_examples/task9_small_circuit_examples.ipynb`
  - ideal / noise population 对比

### 3. Sweep curves

- `report/task1_single_qubit_rabi/task1_single_qubit_rabi.ipynb`
  - gate time sweep
  - leakage vs gate time
- `report/task2_single_qubit_decoherence/task2_single_qubit_decoherence.ipynb`
  - delay sweep
- `report/task3_gaussian_drag_comparison/task3_gaussian_drag_comparison.ipynb`
  - beta scan
  - gate-time scan

### 4. Bar-chart summaries

- `report/task1_single_qubit_rabi/task1_single_qubit_rabi.ipynb`
  - 最终态 population 柱状图
- `report/task8_dynamical_decoupling/task8_dynamical_decoupling.ipynb`
  - final `P1` / leakage 分组柱状图
- `report/task9_small_circuit_examples/task9_small_circuit_examples.ipynb`
  - final population grouped bars
- `src/musiq/pulse/visualize.py`
  - error budget bar chart

### 5. IQ cloud / readout separation

- `report/task6_single_qubit_readout/task6_single_qubit_readout.ipynb`
  - `I/Q` 散点云
  - 判别边界 / 中线
- `src/musiq/ui/notebook.py`
  - integrated heterodyne `I/Q` cloud helpers

## Common Plot Primitives

从 notebook 代码里抽出来后，复用价值最高的原子能力有：

- `plot_pulse_envelope(ax, ...)`
- `plot_pulse_channels(ax, ...)`
- `plot_population_series(ax, ...)`
- `plot_metric_series(ax, ...)`
- `plot_grouped_bars(ax, ...)`
- `plot_error_budget(ax, ...)`
- `plot_iq_cloud(ax, ...)`
- `plot_iq_clouds(ax, ...)`
- `plot_trajectory(ax, ...)`

## Refactor Direction

- 在 `src/musiq/visualization/` 新建通用画图包。
- 以 `plot_*(ax, ...)` 作为基础 API。
- 允许保留少量 `make_*_figure(...)` 包装器，供现有 workflow / notebook helper 继续返回 `matplotlib.figure.Figure`。
- `src/musiq/pulse/visualize.py` 保留，但不再作为新调用入口。

## Recommended Model-Aware API

- `plot_pulse(ax, model, run_id="solver_0", ...)`
  - 从 `model.runs[*].artifacts.pulse_ir` 取 pulse 并画到已有 `ax`
- `plot_case_metrics(ax, model, "case_0", "population", ...)`
  - 从 `model.analyses["case_0"].output.metrics["population"]` 取时间演化
- `plot_sweep_metrics(ax, model, "sweep_0", "final_P0", ...)`
  - 从 `ParametricAnalysis` 里取参数轴和 sweep metric
- `plot_case_final_population(ax, model, "case_0", "population", ...)`
  - 把 case metric 的末态值画成柱状图
- `plot_case_iq_cloud(ax, model, "case_0", ...)`
  - 从 case analysis 对应的 trajectory / measurement records 取 integrated IQ
