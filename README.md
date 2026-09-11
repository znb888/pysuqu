<div align="center">
  <img
    src="https://raw.githubusercontent.com/znb888/pysuqu/main/docs/assets/pysuqu-logo.svg"
    alt="pysuqu logo"
    width="760"
  />
</div>

# pysuqu

[![PyPI version](https://img.shields.io/pypi/v/pysuqu?style=flat-square&label=PyPI&color=0f766e&logo=pypi&logoColor=white)](https://pypi.org/project/pysuqu/)
[![Python versions](https://img.shields.io/pypi/pyversions/pysuqu?style=flat-square&label=Python&color=2563eb&logo=python&logoColor=white)](https://pypi.org/project/pysuqu/)
[![License](https://img.shields.io/badge/License-AGPLv3%2B-f59e0b?style=flat-square&logo=gnu&logoColor=white)](https://github.com/znb888/pysuqu/blob/main/LICENSE)

Python toolkit for superconducting qubit simulation.

`pysuqu` provides qubit modeling, multi-qubit and coupler workflows, and
decoherence/noise analysis in one package.

[English](#english) | [简体中文](#zh-cn)

<a id="english"></a>

## English

`pysuqu` is a public Python package for superconducting qubit simulation. This
repository contains the library, documentation, tests, and demo notebooks with
synthetic or public-safe data.

### Install

For most users, install from PyPI:

```bash
pip install pysuqu
```

`pysuqu` supports Python `3.9+` and QuTiP `5+`. Native wheels include the optional
C++ backend; a platform-independent wheel also supports compiler-free QuTiP use.

If you want the latest repository version for development:

```bash
git clone https://github.com/znb888/pysuqu.git
cd pysuqu
pip install -r requirements.txt
pip install -e .
```

### What It Covers

- `pysuqu.qubit` for single-qubit and multi-qubit superconducting circuit
  modeling
- `pysuqu.decoherence` for decoherence and noise-analysis workflows
- transmission-chain simulation for AWG/qubit reference planes, Touchstone
  S-parameters, digital precorrection, and multi-line MIMO propagation
- `demo/` for end-to-end notebook examples

### What's New In 2.1.2

Version `2.1.2` extends the prepared propagation and optional C++ simulation acceleration:

- reusable QuTiP and native contexts, state batches, and gate-fidelity integration
- dense and structured sparse kernels, exact interaction frames, and Krylov actions
- static-collapse Lindblad evolution without an explicit Kronecker Liouvillian
- explicit backend selection, capability-aware fallback, and opt-in RWA models
- a runnable synthetic tutorial, comparative benchmarks, and tested native wheels
- opt-in gate, waveform, Touchstone, and filtered-PSD preparation caches
- dynamic Lindblad rate descriptors, predistortion design helpers, and portable build tuning

See the [2.1.2 release notes](https://github.com/znb888/pysuqu/blob/main/docs/releases/2.1.2.md)
for the complete public summary.

### Quick Start

This example builds a simple transmon-like qubit model and prints the first
three energy levels:

```python
from pysuqu.qubit import AbstractQubit

qubit = AbstractQubit(
    frequency=5e9,
    anharmonicity=-250e6,
    frequency_max=6e9,
    qubit_type="Transmon",
    energy_trunc_level=12,
    is_print=False,
)

print(qubit.get_energylevel()[:3])
```

For decoherence and coupler workflows, the notebook examples in `demo/` are the
best next stop.

### Where To Start

- [Getting started guide](https://github.com/znb888/pysuqu/blob/main/docs/guides/getting-started.md)
- [Documentation index](https://github.com/znb888/pysuqu/blob/main/docs/README.md)
- [Demo notebooks](https://github.com/znb888/pysuqu/blob/main/demo/README.md)
- [Transmission-chain guide](https://github.com/znb888/pysuqu/blob/main/docs/guides/transmission-chains.md)
- [Propagation backend guide](https://github.com/znb888/pysuqu/blob/main/docs/guides/propagation-backends.md)
- [Module map](https://github.com/znb888/pysuqu/blob/main/docs/architecture/module-map.md)

### Public API Notes

- For normal usage, start from `pysuqu.qubit` and `pysuqu.decoherence`.
- `pysuqu.qubit.experimental` contains exploratory placeholders and is not part
  of the usual workflow.
- Demo notebooks in this public repository use synthetic or public-safe data.

<a id="zh-cn"></a>

## 简体中文

`pysuqu` 是一个面向超导量子比特仿真的 Python 包。这个公开仓库包含库代码、
文档、测试，以及使用合成数据或公开安全数据整理出的 demo notebook。

### 安装

普通用户直接从 PyPI 安装：

```bash
pip install pysuqu
```

`pysuqu` 支持 Python `3.9+` 和 QuTiP `5+`。原生 wheel 包含可选 C++ 后端；
平台无关 wheel 也支持无需编译器的 QuTiP 仿真。

如果你想基于仓库开发，安装方式如下：

```bash
git clone https://github.com/znb888/pysuqu.git
cd pysuqu
pip install -r requirements.txt
pip install -e .
```

### 主要内容

- `pysuqu.qubit`：单比特、多比特和耦合器相关的建模能力
- `pysuqu.decoherence`：退相干与噪声分析工作流
- 传输链仿真：AWG/量子比特参考面、Touchstone S 参数、数字预校正与多线 MIMO 传播
- `demo/`：端到端 notebook 示例

### 2.1.2 新增内容

`2.1.2` 扩展了可复用传播上下文和可选 C++ 仿真加速：

- 可复用的 QuTiP 与原生传播、批量初态演化和门保真度接口
- 稠密与结构化稀疏内核、精确相互作用绘景和 Krylov 指数作用
- 无需显式构造 Kronecker Liouvillian 的静态耗散 Lindblad 演化
- 明确的后端选择、按能力回退，以及需要显式启用的 RWA 近似
- 可运行的合成参数教程、对照基准和经过测试的原生 wheel
- 门、波形、Touchstone 和滤波 PSD 的可选准备缓存
- 动态 Lindblad 速率描述符、预校正设计工具和可移植构建调优

完整说明见 [2.1.2 发布说明](https://github.com/znb888/pysuqu/blob/main/docs/releases/2.1.2.md)。

### 快速开始

下面这个例子会构建一个简化的 transmon 模型，并输出前三个能级：

```python
from pysuqu.qubit import AbstractQubit

qubit = AbstractQubit(
    frequency=5e9,
    anharmonicity=-250e6,
    frequency_max=6e9,
    qubit_type="Transmon",
    energy_trunc_level=12,
    is_print=False,
)

print(qubit.get_energylevel()[:3])
```

如果你要看退相干分析或 coupler 工作流，最直接的入口是 `demo/` 里的 notebook。

### 从哪里开始

- [入门指南](https://github.com/znb888/pysuqu/blob/main/docs/guides/getting-started.md)
- [文档索引](https://github.com/znb888/pysuqu/blob/main/docs/README.md)
- [Demo 说明](https://github.com/znb888/pysuqu/blob/main/demo/README.md)
- [传输链指南](https://github.com/znb888/pysuqu/blob/main/docs/guides/transmission-chains.md)
- [传播后端指南](https://github.com/znb888/pysuqu/blob/main/docs/guides/propagation-backends.md)
- [模块地图](https://github.com/znb888/pysuqu/blob/main/docs/architecture/module-map.md)

### 公共 API 说明

- 正常使用时，优先从 `pysuqu.qubit` 和 `pysuqu.decoherence` 开始。
- `pysuqu.qubit.experimental` 里是探索性占位入口，不属于常规工作流。
- 本公开仓库中的 demo notebook 只使用合成数据或公开安全数据。

## License / 许可证

`pysuqu` is licensed under the
[GNU Affero General Public License v3.0 or later](https://github.com/znb888/pysuqu/blob/main/LICENSE).

欢迎通过 issue 和 pull request 提出改进。
