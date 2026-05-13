---
paths:
  - "src/**"
  - "packages/**"
  - "tests/**"
  - "examples/**"
  - "scripts/**"
  - "pyproject.toml"
  - "*.py"
  - "*.yaml"
  - "*.yml"
---

# MuSIQ 架构规则

本规则是 `MuSIQ` 项目的局部架构规则，只负责架构边界、数据归属、类型结构和命名约定。写代码时仍必须同时遵守全局代码质量规则中的 import、API 兼容、版本检查和测试验证要求。

## 核心原则

`MuSIQ` 是分层量子仿真工作流系统。一个概念只能有一个权威归属。配置、运行时契约、编译产物、参数点结果、trajectory / shot 采样记录和派生分析不能混在一起。稳定结构优先使用类型对象，不要用裸 `dict[str, Any]` 充当核心架构。

数据流方向必须保持为：

```text
config -> runtime contract -> compiled artifacts -> run results -> trajectories / shots -> analysis
```

## 目标模型结构

长期目标结构如下。新增代码必须朝这个结构收敛；legacy code 只能作为兼容层存在。

```text
Model (顶级工作流容器)
├── config: ModelConfig (静态输入定义)
│   ├── tasks: dict[str, TaskConfig]              # 任务定义
│   ├── devices: dict[str, DeviceConfig]          # 设备定义
│   ├── pulses: dict[str, PulseConfig]            # 脉冲定义
│   ├── solvers: dict[str, SolverConfig]          # 求解器配置
│   ├── analysers: dict[str, AnalyserConfig]      # 分析器配置
│   └── parameter_list: ParameterSweepConfig | None
│       └── parameters: dict[str, ParameterList]  # 参数扫描定义
│
├── state: ModelState (轻量级会话状态)
│   ├── default_solver_id: str | None
│   ├── default_analyser_id: str | None
│   └── last_run_id: str | None
│
├── runs: dict[str, ModelRun] (编译单元 / 运行组)
│   └── <run_id>: ModelRun
│       ├── identity: RunIdentity
│       ├── runtime_task: Task                    # 运行时契约
│       ├── status: RunStatus                     # RUNNING / COMPLETED / FAILED
│       ├── artifacts: RunArtifacts               # 共享编译产物，编译一次
│       │   ├── circuit: Circuit IR
│       │   ├── normalized_circuit: Normalized Circuit IR
│       │   ├── model_spec: ModelSpec             # 核心引擎中立模型
│       │   ├── pulse_ir: Pulse IR
│       │   ├── executable_model: Lowered Model
│       │   └── timings: dict[str, float]
│       └── results: dict[str, RunResult]         # 参数扫描点结果
│           └── <parameter_id>: RunResult
│               ├── result_id: str
│               ├── parameters: ParameterValues   # 参数点取值快照
│               ├── trajectories: dict[str, Trajectory]
│               │   └── <trajectory_id>: Trajectory
│               └── provenance: RunProvenance
│
├── analyses: dict[str, ModelAnalysis] (派生分析产物)
│   └── <analysis_id>: ModelAnalysis
│       ├── analysis_id: str
│       ├── analyser_id: str
│       ├── input_results: list[ResultRef]
│       ├── scope: AnalysisScope                  # CASE / PARAMETRIC / COMPREHENSIVE 等
│       └── output: CaseAnalysis | ParametricAnalysis | ComprehensiveAnalysis
│           ├── CaseAnalysis
│           │   ├── quantum_state: QuantumStateAnalysis | None
│           │   ├── observables: ObservablesAnalysis | None
│           │   └── readout: ReadoutAnalysis | None
│           │       ├── signal: ReadoutSignalAnalysis | None
│           │       ├── iq: IQAnalysis | None
│           │       └── inferred_state: InferredStateAnalysis | None
│           ├── ParametricAnalysis
│           │   ├── parameter_axes
│           │   ├── metric_series
│           │   └── input_results: list[ResultRef]
│           └── ComprehensiveAnalysis
│               ├── parametric_analyses
│               └── cross_analysis
│
└── registry: ModelRegistry (指标注册表)
    └── metric_registry: MetricRegistry
```

## 命名约定

### 配置类命名

不要使用 `Workflow*Config` 前缀。配置对象直接使用短名称：

```text
TaskConfig
DeviceConfig
PulseConfig
SolverConfig
AnalyserConfig
```

运行时任务契约使用：

```text
Task
```

如果旧代码中存在 `WorkflowTaskConfig`、`WorkflowDeviceConfig`、`WorkflowPulseConfig`、`WorkflowSolverConfig`、`WorkflowAnalyserConfig` 等名称，新增代码不要继续扩散这些旧名称。迁移时应逐步改向短名称，必要时保留明确标注的兼容别名。

### 参数扫描命名

参数扫描配置使用：

```text
parameter_list: ParameterSweepConfig | None
```

其中：

```text
ParameterSweepConfig
├── parameters: dict[str, ParameterList]
├── mode: SweepMode | None
└── metadata: dict[str, Any]

ParameterList
├── name: str
├── target: str
├── values: list[Any]
├── unit: str | None
└── description: str | None

ParameterValues
├── parameter_id: str
├── values: dict[str, Any]
└── metadata: dict[str, Any]
```

语义必须保持清楚：

- `ParameterSweepConfig` 是配置侧“整个参数扫描配置”。
- `parameter_list` 是 `ModelConfig` 中保存参数扫描配置的字段名。
- `ParameterList` 是配置侧“某一个被扫描参数及其候选取值列表”。
- `ParameterValues` 是结果侧“某一个参数点实际绑定的值”。
- `parameter_id` 是某个参数点结果的稳定标识。

不要使用以下命名作为主结构：

- `sweep_space`
- `parameter_space`
- `dimensions`
- `SweepAxis`
- `SweepDimension`
- `sample_params`

这些名字要么过于抽象，要么容易把参数扫描点和随机采样 / shot 混淆。

## ModelRun、RunResult、Trajectory 的层级

必须使用下面这个语义层级：

```text
ModelRun = 编译单元 / 运行组
RunResult = 一个参数点的结果容器
Trajectory = 该参数点下的一条数值轨迹、一次重复、一次 shot 或一次 stochastic realization
```

`ModelRun.results` 必须保持为：

```text
results: dict[str, RunResult]
```

其中 key 是 `parameter_id`，不是 `sample_id`、`shot_id` 或 `trajectory_id`。

标准结构是：

```text
Model
└── runs
    └── <run_id>: ModelRun
        ├── artifacts
        └── results: dict[str, RunResult]
            └── <parameter_id>: RunResult
                ├── parameters: ParameterValues
                └── trajectories: dict[str, Trajectory]
                    └── <trajectory_id>: Trajectory
```

这表示：

- 一个 `Model` 里，不同配置组合可以生成多个 `ModelRun`。
- 一个 `ModelRun` 共享一次编译得到的 `RunArtifacts`。
- 一个 `ModelRun.results[parameter_id]` 对应参数扫描中的一个参数点。
- 一个 `RunResult` 内部可以有多条 `Trajectory`，表示该参数点下的多次重复、shot、Monte Carlo trajectory 或 stochastic realization。
- shot / sample / repetition 是更下层概念，不能作为 `ModelRun.results` 的 key。

如果某个 `Trajectory` 内部还包含更细的测量采样记录，应继续放在 `Trajectory.measurements` 下面，例如：

```text
Trajectory
└── measurements: MeasurementRecords
    ├── shots: list[ShotRecord]
    ├── iq_records
    └── raw_samples
```

## 核心对象归属

### Model

`Model` 是顶级工作流容器，只保存：

- 静态输入配置集合
- registry
- 轻量 session state
- run collection
- model-level analyses

不要把 per-run compiled artifacts、trajectory、pulse IR、decoder outputs 放到 `Model` 顶层。

### ModelConfig

`ModelConfig` 保存静态输入定义，不保存执行产物。

必须使用复数映射：

```text
tasks: dict[str, TaskConfig]
devices: dict[str, DeviceConfig]
pulses: dict[str, PulseConfig]
solvers: dict[str, SolverConfig]
analysers: dict[str, AnalyserConfig]
parameter_list: ParameterSweepConfig | None
```

不同配置文件、不同任务组合、不同 device / pulse / solver / analyser 组合可以生成不同 `ModelRun`。不要把“当前任务”“当前设备”“当前脉冲”硬编码成单例字段。

### ModelState

`ModelState` 只保存轻量 session 状态，例如：

- `default_solver_id`
- `default_analyser_id`
- `last_run_id`

不要保存 compiled model、trajectory、analysis payload、pulse IR 或完整配置副本。

### ModelRun

`ModelRun` 是编译单元或运行组，不是单个参数点，也不是单个 shot。

一个 `ModelRun` 表示某组配置和运行时契约被编译后形成的共享执行上下文。它包含：

```text
identity: RunIdentity
runtime_task: Task
status: RunStatus
artifacts: RunArtifacts
results: dict[str, RunResult]
```

如果两个执行共享同一个编译产物，只是参数值不同，它们应该位于同一个 `ModelRun.results` 下。

如果配置文件、线路结构、设备结构、脉冲 IR 结构、Hilbert space 维度、求解器后端或编译结果发生实质变化，应创建新的 `ModelRun`。

### RunArtifacts

编译、lowering、解析和执行准备阶段的中间产物放在 `RunArtifacts`，并且在一个 `ModelRun` 内共享。

允许字段：

- `circuit`
- `normalized_circuit`
- `compile_report`
- `model_spec: ModelSpec`
- `pulse_ir`
- `executable_model`
- `decoder_outputs`
- `timings: dict[str, float]`

`RunArtifacts` 不是用户配置，也不是数值结果，更不是分析输出。

如果参数扫描值只改变数值绑定，不改变结构，应让 `RunArtifacts.model_spec` 表示可绑定参数的模板或编译后模型，并把具体值放入 `RunResult.parameters`。

如果参数扫描值会改变线路拓扑、pulse IR 结构、Hilbert space 维度、求解器类型或 executable model 结构，应拆成多个 `ModelRun`，不要强行塞进同一个 run 的 `results`。

### RunResult

`RunResult` 是某一个参数点的事实执行结果容器，不是单条 trajectory，也不是单个 shot。

它必须包含：

```text
result_id: str
parameters: ParameterValues
trajectories: dict[str, Trajectory]
provenance: RunProvenance
```

不要把 compile artifact、pulse IR、analysis summary 或任意 runtime bag 塞进 `RunResult`。

不要用 `RunResult` 表示 shot。shot / trial / repetition 是 `RunResult.trajectories` 或 `Trajectory.measurements` 更下层 typed record 的内容。

### Trajectory

`Trajectory` 表示某个参数点下的一条数值轨迹或一次重复采样结果。

推荐结构：

```text
Trajectory
├── trajectory_id: str
├── schema_version: str
├── engine: str
├── times: list[float]
├── wave_function
├── density_matrix
├── classical
├── measurements
└── metadata
```

其中：

- `wave_function` 和 `density_matrix` 是可选的量子态负载。
- `classical` 存储经典或读出侧时间序列。
- `measurements` 存储测量侧输出、shot 记录、IQ 记录或结构化采样记录。
- `metadata` 只存储非权威注释和扩展信息。

### ModelAnalysis

派生分析结果属于 model-level：

```text
model.analyses[analysis_id]
```

每个分析必须显式声明依赖哪些参数点结果：

```text
input_results: list[ResultRef]
```

其中：

```text
ResultRef
├── run_id: str
└── parameter_id: str
```

如果分析需要精确引用某一条 trajectory，应使用更细粒度的可选引用：

```text
TrajectoryRef
├── run_id: str
├── parameter_id: str
└── trajectory_id: str
```

分析 scope 必须显式，并且只保留在 `ModelAnalysis.scope` 这一处，不要在具体 analysis payload 内重复保存一份。

推荐的 analysis 分级是：

- `CASE`
- `PARAMETRIC`
- `COMPREHENSIVE`

即使是单参数点分析，也不要直接挂在 `RunResult` 下作为权威存储。可以保留便捷引用，但权威分析产物必须在 `Model.analyses`。

分析输出必须按粒度分层，不要把单点分析字段和参数扫描序列字段混放在同一个通用 output 容器中。

推荐结构如下：

```text
CaseAnalysis
├── quantum_state: QuantumStateAnalysis | None
├── observables: ObservablesAnalysis | None
└── readout: ReadoutAnalysis | None
    ├── signal: ReadoutSignalAnalysis | None
    ├── iq: IQAnalysis | None
    └── inferred_state: InferredStateAnalysis | None

ParametricAnalysis
├── parameter_axes
├── metric_series
└── input_results: list[ResultRef]

ComprehensiveAnalysis
├── parametric_analyses
└── cross_analysis
```

其中：

- `CaseAnalysis` 只表示单个 `RunResult` 级别的分析产物。
- `ParametricAnalysis` 只表示参数扫描或参数序列上的聚合分析结果。
- `ComprehensiveAnalysis` 只表示跨多个参数分析结果、多个 study step 或多个 sweep 的更高层汇总与比较。

`CaseAnalysis` 内部必须继续区分“物理量子态”与“读出链路推断态”：

- `quantum_state` 描述仿真或演化得到的系统本身状态，例如 population、density matrix、purity、Bloch vector、state fidelity。
- `observables` 描述从状态、轨迹或测量记录中导出的可观测量与指标，是 case 级别指标容器，不等同于读出判决结果。
- `readout.signal` 描述原始或预处理的读出信号。
- `readout.iq` 描述 IQ 平面上的云团、聚类、质心和噪声分析。
- `readout.inferred_state` 描述根据读出链路推断出的离散状态、分类概率、assignment fidelity、confusion matrix 等判读结果。

不得使用同一个字段同时承载“量子系统物理态”和“读出后推断态”。

`ParametricAnalysis.metric_series` 用于保存参数扫描序列数据，例如某个指标随 `parameter_id` 或某个 `ParameterList` 变化的曲线。它是派生结果，不是原始 trajectory，也不是 `RunResult` 的替代。

## ModelSpec 规则

`ModelSpec` 是 engine-neutral executable model description。它的权威位置是：

```text
model.runs[run_id].artifacts.model_spec
```

不得同时把 `model.spec`、`runtime_metadata["model_spec"]`、`run.artifacts.model_spec` 当作并列真源。

如果 legacy 兼容期需要镜像字段，必须明确标注 deprecated / read-only / compatibility mirror。

`ModelSpec` 内部应保持结构化：

```text
ModelSpec
├── circuit
├── solver
├── time
├── frame
├── system
├── hamiltonian
├── noise
├── readout
├── analysis_request
├── study
└── metadata
```

不要把它掏空成无结构的大字典。

## Pulse 三层分离

Pulse 必须拆成三层：

1. 用户配置：`model.config.pulses[pulse_id]: PulseConfig`
2. lowering 后 IR：`model.runs[run_id].artifacts.pulse_ir`
3. 运行产物：run-scoped artifacts 或 run output directory 中的文件

不要用同一个裸 dict 同时表示 pulse config、pulse IR 和 pulse-generated outputs。

## 类型规则

默认使用类型对象，不默认使用 `dict[str, Any]`。

必须 typed 的对象包括：

- `Model`
- `ModelConfig`
- `ModelState`
- `ModelRun`
- `RunIdentity`
- `RunStatus`
- `RunArtifacts`
- `TaskConfig`
- `DeviceConfig`
- `PulseConfig`
- `SolverConfig`
- `AnalyserConfig`
- `ParameterSweepConfig`
- `ParameterList`
- `ParameterValues`
- `ResultRef`
- `TrajectoryRef`
- `Task`
- `ModelSpec`
- `Trajectory`
- `RunResult`
- `RunProvenance`
- `ModelAnalysis`
- `CaseAnalysis`
- `ParametricAnalysis`
- `ComprehensiveAnalysis`
- `MetricSeries`
- manifest / provenance classes

允许小型 leaf dict，但必须位于 typed container 内，例如：

- `metadata`
- `extras`
- backend-specific options
- engine-specific options
- decoder option bags
- `ParameterValues.values`
- `RunArtifacts.timings`
- `RunResult.trajectories`
- `ParametricAnalysis.metric_series`

如果一个 dict 被多个模块读取、反复验证、反复合并默认值、拥有稳定字段名，必须提升为 dataclass / schema class。

## runtime_metadata 使用限制

`runtime_metadata` 只能作为兼容层或轻量追踪信息。它不能成为核心结构的永久归属。

允许：

- debug note
- tracing info
- 小型 provenance helper
- 指向已有结构化对象的 reference

禁止：

- 藏权威 `ModelSpec`
- 藏 pulse IR / executable model / circuit
- 混放 config、result、analysis
- 作为新功能的默认扩展垃圾桶

## 修改前自检

引入或修改架构对象前，先回答：

1. 这个对象是 config、runtime contract、domain model、IR/artifact、parameter result、trajectory / shot record、derived analysis，还是 metadata？
2. 它的唯一权威归属在哪里？
3. 它是否必须 typed？
4. 是否已经存在竞争归属？
5. 如果存在 legacy 重复字段，应删除、迁移，还是标注为兼容镜像？
6. 这个对象属于 model、run、parameter result、trajectory measurement，还是 model-level analysis？

回答不清楚时，不要新增平行结构。

## 禁止事项

不得：

- 把 `model.spec` 和 run-scoped `model_spec` 当作两个真源。
- 把 per-run truth 放到 `Model` 顶层。
- 把 `ModelRun.results` 改名为 `samples` 或 `parameter_results`。
- 把参数点 key 命名为 `sample_id`、`shot_id` 或 `trajectory_id`。
- 把 shot / trial / measurement sample 放到 `ModelRun.results` 的 key 层。
- 把单条 trajectory 直接当作 `RunResult`。
- 把 multi-run analysis 放进只代表 last run 的字段。
- 混合 config 和 execution output。
- 混合 artifact 和 analysis output。
- 用裸 dict 定义 pulse config、run artifacts、核心 analysis outputs、parameter sweep。
- 用 `runtime_metadata` 承担永久架构职责。
- 使用 `Workflow*Config` 前缀扩展新配置类型。
- 为了方便一个调用方而破坏层级方向。

## 完成前自检

完成架构相关修改前，必须确认：

1. `ModelSpec` 是否只有一个权威 runtime 位置？
2. per-run 对象是否仍在 run 下？
3. 参数点结果是否存放在 `ModelRun.results[parameter_id]`？
4. 每个 `RunResult` 是否包含 `parameters` 和 `trajectories`？
5. shot / sample / repetition 是否放在 `RunResult.trajectories` 或 `Trajectory.measurements` 更下层？
6. analysis 是否通过 `ResultRef(run_id, parameter_id)` 指向输入？
7. config、artifact、result、trajectory、analysis 是否分层清楚？
8. core object 是否没有被藏进裸 dict 或 `runtime_metadata`？
9. 新增对象是否有明确类型和单一归属？
10. 是否运行了全局代码质量规则要求的 compile / collect / test / smoke test？
