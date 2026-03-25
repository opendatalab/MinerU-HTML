# MinerU-HTML

[English](README.md)/中文

**MinerU-HTML** 是一款基于小语言模型（SLM）的高级 HTML 正文提取工具。它能够从复杂的网页 HTML 中准确识别并提取主要内容，自动过滤导航栏、广告和元数据等辅助元素。

[**在线体验 -\> Mineru-Extractor**](https://mineru.net/OpenSourceTools/Extractor)

欢迎体验我们的在线文档提取工具！支持 HTML 正文提取以及多种文档格式的 OCR 识别。

或

[**下载模型进行本地部署 -\> MinerU-HTML-v1.1**](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) —— 现已更新至 v1.1 版本！

## 最新动态

  - 2026.03.19 🎉 [MinerU-HTML-v1.1](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) 发布！相比 v1.0 版本更高效、更强大，现已集成 [MinerU-Webkit](https://github.com/opendatalab/MinerU-Webkit)，支持 HTML 到 Markdown/JSON/Txt 的转换。欢迎使用！
  - 2025.12.1 🎉 [AICC](https://huggingface.co/datasets/opendatalab/AICC) 数据集发布，欢迎使用！AICC 数据集包含 7.3T 由 MinerU-HTML 提取并转换成 Markdown 格式的网页，具有更干净的正文内容以及高质量的代码、公式和表格。
  - 2025.12.1 🎉 [MinerU-HTML](https://huggingface.co/opendatalab/MinerU-HTML) 模型发布！该模型基于 Qwen3 进行微调，在 HTML 正文提取任务上表现卓越。
  - 2025.12.1 🎉 体验页面上线，欢迎访问 [Opendatalab-AICC](https://opendatalab.com/ai-ready/AICC#playground) 试用我们的提取工具！

## ✨ 功能特性

  - 🎯 **LLM 驱动提取**：利用先进的语言模型智能识别正文内容。
  - 📝 **可扩展输出**：集成 [MinerU-Webkit](https://github.com/opendatalab/MinerU-Webkit)，支持将提取的 HTML 高效转换为 Markdown、JSON 和 Txt 格式。
  - ⚡ **紧凑格式 (Compact Format)**：模型现采用紧凑格式而非原始 JSON 格式，推理速度更快。
  - ⚡ **正则结构化输出**：模型现采用正则结构化输出替代自定义 logits 处理器，从而支持使用 vLLM v1 后端进行推理。
  - 🔄 **完整的处理流水线**：包括 HTML 简化、提示词构建、LLM 推理、结果解析、正文提取等完整流程。
  - 🚀 **多推理后端支持**：支持 VLLM、Transformers 和 OpenAI API 三种推理后端。
  - 🛡️ **容错机制**：支持回退机制（trafilatura、bypass 或 empty），确保处理失败时仍能提取内容。
  - 🔌 **模块化设计**：架构清晰，易于扩展和自定义。
  - 🧪 **完善的测试**：包含全面的单元测试和集成测试。

## 评估结果

我们在 [WebMainBench v1.1](https://github.com/opendatalab/WebMainBench/) 基准测试上对 MinerU-HTML 进行了评估。该基准包含 7,809 个（v1.0 为 7,887 个，我们剔除了一些低质量页面）精心标注的网页及其对应的由 `html2text` 转换的 Markdown 格式正文。评估通过计算提取结果与标准答案（Ground-truth）之间的 ROUGE-N 分数来衡量准确度。主要评估结果如下表所示：

| Extractor        | ROUGE-N.f1 |
| ---------------- | ---------- |
| DeepSeek-V3\*    | 0.9098     |
| GPT-5\*          | 0.9024     |
| MinerU-HTML-v1.1 | 0.9001     |
| Magic-HTML       | 0.7138     |
| Readability      | 0.6542     |
| Trafilatura      | 0.6402     |
| Resiliparse      | 0.6290     |
| html2text        | 0.6042     |
| BoilerPy3        | 0.5434     |
| GNE              | 0.5171     |
| news-please      | 0.5032     |
| justText         | 0.4782     |
| Goose3           | 0.4371     |
| ReaderLM-v2      | 0.2279     |

注：标有 \* 的项目表示在 MinerU-HTML 框架中使用 GPT-5/Deepseek-V3 提取 HTML 正文，而非使用我们微调的模型。

## 🚀 快速上手

### 下载模型

访问 Hugging Face 页面 [MinerU-HTML-v1.1-compact](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) 下载模型，或使用以下命令：

```bash
huggingface-cli download opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact
```

### 安装

#### 默认安装（VLLM 后端）

我们推荐使用 VLLM 后端进行安装，以获得最佳性能并适配主要使用场景：

```bash
pip install mineru_html[vllm]
```

从源码安装：

```bash
git clone https://github.com/opendatalab/MinerU-HTML
cd MinerU-HTML
pip install .[vllm]
```

#### 备选后端

如果 VLLM 不适用于您的环境（例如您希望使用不同的推理引擎、缺少 GPU 资源，或需要使用 OpenAI API），可以选择以下后端之一：

- **Transformers 后端**（使用 Hugging Face Transformers 进行本地推理，无额外依赖）：

  ```bash
  pip install mineru_html
  ```

- **OpenAI API 后端**（调用远程 OpenAI 兼容 API）：

  ```bash
  pip install mineru_html[openai]
  ```

### 基本用法

#### 默认用法（GPU 上的 VLLM 后端）

```python
from mineru_html import MinerUHTML, MinerUHTMLConfig


config = MinerUHTMLConfig(
    use_fall_back='trafilatura',    # 可选 'trafilatura','bypass' 或 'empty'
    prompt_version='short_compact', # v1.1 仅支持 'short_compact'
    response_format='compact',      # v1.1 仅支持 'compact'
    early_load=True
)

# 初始化 MinerUHTML
extractor = MinerUHTML(
    model_path='path/to/your/model',
    config=config
)

# 处理单个 HTML
html_content = '<html>...</html>'
result = extractor.process(html_content)
print(result[0].output_data.main_content)

# 批量处理 HTML
html_list = ['<html>...</html>', '<html>...</html>']
results = extractor.process(html_list)
for result in results:
    print(result.output_data.main_content)
    print(result.case_id)
extractor.llm.cleanup()
```

#### 使用 Transformers 后端

```python
from mineru_html import MinerUHTML_Transformers, MinerUHTMLConfig

config = MinerUHTMLConfig(
    use_fall_back='trafilatura',
    prompt_version='short_compact',  # v1.1 仅支持 'short_compact'
    response_format='compact',       # v1.1 仅支持 'compact'
    early_load=True
)

# 初始化 MinerUHTML_Transformers
extractor = MinerUHTML_Transformers(
    model_path='path/to/your/model',
    config=config,
    model_init_kwargs={
        'device_map': 'auto',
        'dtype': 'auto',
    },
    model_gen_kwargs={
        'max_new_tokens': 16 * 1024,
    }
)

# 处理 HTML
html_content = '<html>...</html>'
result = extractor.process(html_content)
print(result[0].output_data.main_content)
```

## 📖 核心概念

### 处理流水线

MinerU-HTML 的处理流水线包含以下步骤：

1.  **HTML 简化** (`simplify_html`)：将原始 HTML 简化为结构化格式，并为每个元素分配唯一的 `_item_id` 属性。
2.  **提示词构建** (`build_prompt`)：基于简化后的 HTML 构建 LLM 提示词，引导模型进行内容分类。
3.  **LLM 推理** (`inference`)：利用 LLM 对每个元素进行分类，标记为“main”（正文）或“other”（辅助内容）。
4.  **结果解析** (`parse_result`)：解析 LLM 返回的分类结果。
5.  **正文提取** (`extract_main_html`)：根据分类结果从原始 HTML 中提取正文内容。
6.  **格式转换** (`convert2content`)：将提取的 HTML 转换为 Markdown、JSON 或 Txt 格式。
7.  **回退处理**：若上述流程失败，则使用回退机制（trafilatura、bypass 或 empty）进行提取。

### 配置选项

`MinerUHTMLConfig` 支持以下配置：

  - `use_fall_back`: 回退类型，可选 `'trafilatura'`、`'bypass'` 或 `'empty'`。
  - `early_load`: 是否提前加载模型（默认 `True`）。
  - `prompt_version`: 提示词版本，可选 `'v0'`、`'v1'`、`'v2'`、`'compact'`、`'short_compact'`。MinerUHTML 和 MinerUHTML\_Transformers 接口默认使用 `'short_compact'`；MinerUHTML\_OpenAI 接口默认使用 `'v2'`。
  - `response_format`: 模型的输出格式。仅允许 `'json'` 或 `'compact'`。VLLM/Transformers 默认使用 `'compact'`；OpenAI 默认使用 `'json'`。
  - `output_format`: 目标转换格式，支持 `'mm_md'`（标准 Markdown）、`'md'`（带图片的 Markdown）、`'json'` 和 `'txt'`。

#### 不同 prompt\_version 的使用说明

  - `'compact'`: 用于本地模型推理，返回更简洁的结果（仅保留 JSON 字典中的键和值）。推荐使用 `'compact'` 模型以获得更快的推理速度。
  - `'v2'`: 用于 OpenAI API 推理，是经过提示词优化后的结果。

## 🔧 进阶用法

### 使用工厂函数创建后端

您可以直接使用工厂函数创建后端，然后将其传递给 `MinerUHTMLGeneric`：

```python
from mineru_html import MinerUHTMLGeneric, MinerUHTMLConfig
from mineru_html.inference.factory import create_vllm_backend, create_transformers_backend

# 使用工厂函数创建 VLLM 后端
llm = create_vllm_backend(
    model_path='path/to/model',
    response_format='compact',
    max_context_window=32 * 1024,
    model_init_kwargs={'tensor_parallel_size': 1}
)

# 使用工厂函数创建 Transformers 后端
llm = create_transformers_backend(
    model_path='path/to/model',
    max_context_window=32 * 1024,
    response_format='compact',
    model_init_kwargs={
        'device_map': 'auto',
        'dtype': 'auto',
    },
    model_gen_kwargs={
        'max_new_tokens': 8192,
    }
)

# 使用创建的后端
config = MinerUHTMLConfig()
extractor = MinerUHTMLGeneric(llm=llm, config=config)
```

### 错误处理

```python
from mineru_html.exceptions import MinerUHTMLError

try:
    result = extractor.process(html_content)
except MinerUHTMLError as e:
    print(f"处理失败: {e}")
    print(f"Case ID: {e.case_id}")
```

## 基准测试评估 (Baselines)

运行评估前，需先安装 `baselines.txt` 中的依赖：

```bash
pip install -r baselines.txt
```

然后执行以下命令：

```bash

BENCHMARK_DATA=benchmark/WebMainBench_100.jsonl
RESULT_DIR=benchmark_results
mkdir $RESULT_DIR

# 评估 MinerU-HTML
EXTRACTORS=(
"mineru_html_fallback-html-md"
)
MODEL_PATH=YOUR_MINERUHTML_MODEL_PATH

for extractor in ${EXTRACTORS[@]}; do
    python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
done

# 评估 CPU 提取器
EXTRACTORS=(
"magichtml-html-md"
"readability-html-md"
"trafilatura-html-md"
"resiliparse-text"
"trafilatura-md"
"trafilatura-text"
"fullpage-html-md"
"boilerpy3-text"
"gne-html-md"
"newsplease-text"
"justtext-text"
"boilerpy3-html-md"
"goose3-text"
)

for extractor in ${EXTRACTORS[@]}; do
    python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name $extractor
done

# 评估 ReaderLM
extractor=readerlm-text
MODEL_PATH=YOUR_READERLM_MODEL_PATH

python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
```

MinerU-HTML 支持与多种主流提取器进行对比评估：
  - [**MinerU-HTML**](https://opendatalab.com/ai-ready/AICC#playground) (`mineru_html-html-md`, `mineru_html-html-text`): 主打的基于 LLM 的提取器。

  - [**Magic-HTML**](https://github.com/opendatalab/magic-html): 同样由 **OpenDatalab** 开发的仅限 CPU 的 HTML 提取工具。

  - [**Trafilatura**](https://github.com/adbar/trafilatura): 快速且准确的内容提取工具。

  - [**Readability**](https://github.com/mozilla/readability): Mozilla 的 Readability 算法。

  - [**BoilerPy3**](https://github.com/jmriebold/BoilerPy3): Boilerpipe 的 Python 移植版。

  - [**NewsPlease**](https://github.com/fhamborg/news-please): 新闻文章提取器。

  - [**Goose3**](https://github.com/goose3/goose3): 文章提取器。

  - [**GNE**](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor): 通用新闻提取器 (General News Extractor)。

  - [**ReaderLM**](https://huggingface.co/jinaai/ReaderLM-v2): 基于 LLM 的文本提取器。


## 开源协议

本项目采用 Apache License, Version 2.0 开源协议。详情请参阅 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{liu2025drippertokenefficientmainhtml,
      title={Dripper: Token-Efficient Main HTML Extraction with a Lightweight LM},
      author={Mengjie Liu and Jiahui Peng and Pei Chu and Jiantao Qiu and Ren Ma and He Zhu and Rui Min and Lindong Lu and Wenchang Ning and Linfeng Hou and Kaiwen Liu and Yuan Qu and Zhenxiang Li and Chao Xu and Zhongying Tu and Wentao Zhang and Conghui He},
      year={2025},
      eprint={2511.23119},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.23119},
}
```

如果您使用了提取的 [AICC](https://huggingface.co/datasets/opendatalab/AICC) 数据集，请引用：

```bibtex
@misc{ma2025aiccparsehtmlfiner,
      title={AICC: Parse HTML Finer, Make Models Better -- A 7.3T AI-Ready Corpus Built by a Model-Based HTML Parser},
      author={Ren Ma and Jiantao Qiu and Chao Xu and Pei Chu and Kaiwen Liu and Pengli Ren and Yuan Qu and Jiahui Peng and Linfeng Hou and Mengjie Liu and Lindong Lu and Wenchang Ning and Jia Yu and Rui Min and Jin Shi and Haojiong Chen and Peng Zhang and Wenjian Zhang and Qian Jiang and Zengjie Hu and Guoqiang Yang and Zhenxiang Li and Fukai Shang and Runyuan Ma and Chenlin Su and Zhongying Tu and Wentao Zhang and Dahua Lin and Conghui He},
      year={2025},
      eprint={2511.16397},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.16397},
}
```

## 致谢

  - 基于 [vLLM](https://github.com/vllm-project/vllm) 构建，实现高效的 LLM 推理。
  - 使用 [Trafilatura](https://github.com/adbar/trafilatura) 作为回退提取机制。
  - 基于 [Hunyuan](https://github.com/Tencent-Hunyuan/Hunyuan-0.5B) 进行微调。
  - 受到多项 HTML 内容提取研究的启发。
  - 使用 [dingo](https://github.com/MigoXLab/dingo) 进行 LLM-as-a-judge 成对胜率评估。
