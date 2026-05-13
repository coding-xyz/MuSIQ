---
paths:
  - "docs/**"
  - "README*"
  - "mkdocs.yml"
  - "mkdocs.yaml"
  - "**/*.md"
  - "**/*.mdx"
---

# 中文文档规则

本规则用于创建、更新、审查或生成项目文档。目标不是只写 README，而是形成一个可以在浏览器中打开的中文 Wiki / 操作手册式文档站点。

本规则是项目局部规则。它只应在文档任务中主导行为，不应干扰普通代码开发。普通代码开发优先遵守全局代码质量门禁和项目架构规则。

## 语言要求

文档必须使用中文，包括：

- 页面标题
- 导航栏
- 项目介绍
- 操作流程
- 模块说明
- API Reference 中的说明文字
- 示例代码前后的解释
- 错误排查说明
- 配置字段说明
- changelog / migration guide 中的说明文字

代码中的 public docstring 也必须使用中文。若发现公开 API 的 docstring 是英文，应翻译并改写为准确、自然、工程化的中文。

保留英文的情况仅限：

- 函数名、类名、模块名、参数名、字段名
- 第三方库名称
- CLI 命令
- 协议名、文件名、路径名
- 必须与外部 API 保持一致的英文枚举值或字符串字面量

## 文档目标

文档应写成“可查、可用、可维护”的工程手册：

1. 像 Wiki 一样组织，可以从导航栏进入不同主题。
2. 覆盖项目主要功能、操作流程、配置文件、模块、命令行入口和示例。
3. 包含完整 API Reference。
4. 每个公开函数、类、类型、枚举、dataclass、Pydantic model、TypedDict、Protocol、异常、常量和模块入口都要能查到。
5. API Reference 必须与源码 docstring 同步。
6. 如果公开 API 没有 docstring，必须先补中文 docstring，再生成或更新 API 文档。
7. 不得凭空编造行为；必须阅读实现、测试、示例和已有文档后再写。

## 文档系统

优先沿用项目已有文档系统：MkDocs、Sphinx、Docusaurus、VitePress、TypeDoc 等。

如果项目没有文档系统，且是 Python 项目，优先使用：

```text
MkDocs + Material + mkdocstrings
```

文档必须能构建成静态 HTML，并能在本地浏览器中打开。

常用命令：

```bash
mkdocs build
mkdocs serve
```

如果项目使用其他系统，使用对应命令，例如：

```bash
sphinx-build docs docs/_build/html
npm run docs:build
npm run docs:dev
```

## 导航结构

文档导航至少包含：

```yaml
nav:
  - 项目介绍:
      - 概览: index.md
      - 设计目标: overview/design-goals.md
      - 架构总览: overview/architecture.md
  - 快速开始:
      - 安装: getting-started/installation.md
      - 第一个示例: getting-started/quickstart.md
      - 基本操作流程: getting-started/workflow.md
  - 使用指南:
      - 配置文件说明: guide/configuration.md
      - 常见任务: guide/tasks.md
      - 输入输出: guide/io.md
      - 运行与调试: guide/running-and-debugging.md
  - 模块说明:
      - 模块总览: modules/index.md
      - <模块名>: modules/<module-name>.md
  - API 参考:
      - API 总览: api/index.md
      - <package.module>: api/<module-name>.md
  - 开发者文档:
      - 代码结构: developer/code-structure.md
      - 扩展新功能: developer/extending.md
      - 测试与验证: developer/testing.md
```

小项目可以压缩层级，但必须保留：项目介绍、基本操作流程、模块说明、API 参考。

## 写作风格

文档要写成操作指南，不要写成宣传材料。

每个功能或模块应说明：

- 解决什么问题；
- 用户什么时候需要用；
- 最小可运行示例；
- 需要哪些输入；
- 产生哪些输出；
- 重要参数、默认值和单位；
- 常见错误和排查方法；
- 相关模块和 API Reference 链接。

避免空泛描述。不要写“该模块提供强大的功能”。应写清楚具体职责，例如：“该模块负责把任务配置、设备配置和脉冲配置合并为统一的 ModelSpec，供后端求解器使用。”

伪代码必须明确标注为伪代码。不要把伪代码伪装成可运行示例。

## API Reference 要求

API Reference 必须完整，不能只记录常用接口。

每个公开函数至少写清：

- 用途；
- 完整签名；
- 参数名、类型、默认值、单位和含义；
- 返回类型和返回值含义；
- 可能抛出的异常；
- 副作用，例如文件 I/O、网络 I/O、状态修改；
- 最小使用示例；
- 相关函数或类。

每个公开类或类型至少写清：

- 用途和生命周期；
- 构造参数；
- 属性和 property；
- 方法；
- 不变量和校验规则；
- 典型用法；
- 扩展点。

每个配置对象至少写清：

- 字段名；
- 字段类型；
- 是否必填；
- 默认值；
- 单位；
- 合法范围或枚举值；
- 示例值；
- 对运行行为的影响。

## 中文 docstring 要求

docstring 是生成 API 文档的源头。更新 API Reference 前必须检查公开 API 的 docstring。

要求：

1. 公开函数、类、类型、配置对象必须有中文 docstring。
2. 如果 docstring 缺失，先补中文 docstring。
3. 如果 docstring 是英文，翻译成中文，并根据实现改写准确。
4. 如果 docstring 与实现、测试或示例冲突，以实现和测试为准，修正文档和 docstring。
5. 保留项目已有 docstring 风格；若没有统一风格，Python 使用 Google-style docstring。
6. docstring 解释行为、参数、返回值、异常和重要副作用，不要复制大段实现细节。

推荐格式：

```python
def run_workflow(config_path: str) -> WorkflowResult:
    """从配置文件运行一次工作流。

    该函数读取工作流配置，完成校验和模型构建，并把任务交给选定的
    求解后端执行。

    Args:
        config_path: 工作流配置文件路径。

    Returns:
        工作流运行结果，包含输出数据、元信息和生成的 artifact 路径。

    Raises:
        ConfigValidationError: 配置文件格式或字段不合法。
        RuntimeError: 配置校验通过后，后端执行过程失败。
    """
```

## 文档任务流程

处理文档任务时，按顺序执行：

1. 先检查项目结构和已有文档系统。
2. 识别源码目录、公开模块、CLI 入口、示例、测试、配置文件。
3. 建立模块职责图。
4. 检查文档涉及的公开 API 是否有中文 docstring。
5. 缺失或英文 docstring 要先补齐或翻译。
6. 更新用户指南和操作流程页面。
7. 更新模块说明页面。
8. 更新 API Reference 页面。
9. 更新导航栏。
10. 运行文档构建命令。
11. 修复断链、Markdown 错误、API 渲染错误、导入错误。
12. 在最终回复中说明构建是否通过。

## MkDocs + mkdocstrings 约定

Python 项目使用 MkDocs 时，API 页面优先用 mkdocstrings，不要手工复制签名：

```markdown
# `package.module`

::: package.module
    options:
      show_source: true
      show_root_heading: true
      show_signature_annotations: true
      members_order: source
```

每个重要模块应有独立 API 页面，`api/index.md` 作为 API 参考入口。

## 质量检查清单

完成前检查：

- [ ] 文档可以构建为静态站点。
- [ ] 导航栏包含项目介绍、快速开始、基本流程、模块说明、API 参考。
- [ ] 主要功能都有操作指南。
- [ ] 公开函数、类、类型、配置结构都有 API 文档。
- [ ] 公开 API 的 docstring 已补齐，且为中文。
- [ ] 英文 docstring 已翻译为中文。
- [ ] API Reference 与 docstring、实现、测试一致。
- [ ] 示例使用真实项目路径、参数和命令。
- [ ] 没有断链。
- [ ] 没有未解释的 TODO。
- [ ] 最终回复写明构建命令和构建结果。

## 禁止事项

不要：

- 只改 README 而不维护完整文档站点。
- 只写概念介绍，不写 API Reference。
- 只记录几个常用函数，遗漏其他公开接口。
- 根据记忆写 API 文档。
- 给不存在的函数、参数或配置字段写文档。
- 保留公开 API 的英文 docstring。
- 声称文档构建通过，但没有实际运行构建命令。
- 为了写文档而改坏代码 import 或测试。

## 最终回复格式

完成文档任务后，最终回复使用：

```text
已完成文档更新：

- 更新文件：
- 新增文件：
- 补齐或翻译的中文 docstring：
- API Reference 覆盖模块：
- 本地预览命令：
- 构建命令：
- 构建结果：
- 未确认事项：
```
