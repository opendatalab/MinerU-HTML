# MinerU-HTML

English/[中文](README_zh.md)

**MinerU-HTML** is an advanced HTML main content extraction tool based on Small Language Models (SLM). It can accurately identify and extract main content from complex web page HTML, automatically filtering out auxiliary elements such as navigation bars, advertisements, and metadata.

[**Try on our website -> Mineru-Extractor**](https://mineru.net/OpenSourceTools/Extractor)

Welcome to try our online document extraction tool! Supports HTML main content extraction and OCR for various document formats.

OR

[**Download the model for local usage -> MinerU-HTML-v1.1**](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) —— Now updated to v1.1 !

## News

- 2026.03.19 🎉 The [MinerU-HTML-v1.1](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) is released! A more efficient and powerful version than v1.0, now integrated with [MinerU-Webkit](https://github.com/opendatalab/MinerU-Webkit) for HTML-to-Markdown/JSON/Txt conversion. Welcome to use!
- 2025.12.1 🎉 The [AICC](https://huggingface.co/datasets/opendatalab/AICC) dataset is released, welcome to use! AICC dataset contains 7.3T web pages extracted and converted to Markdown format by MinerU-HTML, with cleaner main content and high-quality code, formulas, and tables.
- 2025.12.1 🎉 The [MinerU-HTML](https://huggingface.co/opendatalab/MinerU-HTML) model is released, welcome to use! MinerU-HTML model is a fine-tuned model on Qwen3, with better performance on HTML main content extraction.
- 2025.12.1 🎉 The trial page is online, welcome to visit [Opendatalab-AICC](https://opendatalab.com/ai-ready/AICC#playground) to try our extraction tool!

## ✨ Features

- 🎯 **LLM-Powered Extraction**: Uses state-of-the-art language models to intelligently identify main content
- 📝 **Extensible Output**: Integrated with [MinerU-Webkit](https://github.com/opendatalab/MinerU-Webkit) to support efficient conversion of extracted HTML into Markdown, JSON and Txt.
- ⚡ **Compact Format**: Now use compact format for the model instead of the original JSON format for faster inference speed
- ⚡ **Regex structured output**: Now use regex structured output for the model instead of the custom logits processor, so we can use vLLM v1 backend for inference.
- 🔄 **Complete Processing Pipeline**: Includes HTML simplification, prompt construction, LLM inference, result parsing, main content extraction, and other complete processes
- 🚀 **Multiple Inference Backend Support**: Supports three inference backends: VLLM, Transformers, and OpenAI API
- 🛡️ **Fault Tolerance**: Supports fallback mechanism (trafilatura, bypass or empty) to ensure content extraction even when processing fails
- 🔌 **Modular Design**: Clear architecture design, easy to extend and customize
- 🧪 **Complete Testing**: Includes comprehensive unit tests and integration tests

## Evaluation Results

We evaluated MinerU-HTML on the [WebMainBench v1.1](https://github.com/opendatalab/WebMainBench/) benchmark, which contains 7,809 (7,887 for v1.0, we removed some low quality web pages) meticulously annotated web pages along with their corresponding Markdown-formatted main content converted using `html2text`. This benchmark measures the extraction accuracy of content extractors by computing ROUGE-N scores between the extracted results and ground-truth content. The primary evaluation results are presented in the table below:

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

where * denotes that use GPT-5/Deepseek-V3 to extract the main html in MinerU-HTML framework instead of our finetuned model.

## 🚀 Quick Start

### Download the model

visit our model at [MinerU-HTML-v1.1-compact](https://huggingface.co/opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact) and download the model, you can use the following command to download the model:

```bash
huggingface-cli download opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact
```

### Installation

#### Default Installation (with VLLM backend)

We recommend installing with the VLLM backend for the best performance and the primary use case:

```bash
pip install mineru_html[vllm]
```

To install from source:

```bash
git clone https://github.com/opendatalab/MinerU-HTML
cd MinerU-HTML
pip install .[vllm]
```

#### Alternative Backends
If VLLM is not suitable for your environment (e.g., you prefer a different inference engine, lack GPU resources, or need to use OpenAI API), you can choose one of the following backends:

- Transformers Backend (local inference with Hugging Face Transformers, no extra dependencies):

```bash
pip install mineru_html
```

- OpenAI API Backend (call remote OpenAI‑compatible APIs):

```bash
pip install mineru_html[openai]
```

### Basic Usage

#### Default Usage (VLLM Backend on GPU)

```python
from mineru_html import MinerUHTML, MinerUHTMLConfig


config = MinerUHTMLConfig(
    use_fall_back='trafilatura',    # optional 'trafilatura','bypass' or 'empty'
    prompt_version='short_compact', # only support 'short_compact' for v1.1
    response_format='compact',      # only support 'compact' for v1.1
    early_load=True
)

# initialize MinerUHTML
extractor = MinerUHTML(
    model_path='path/to/your/model',
    config=config
)

# process single HTML
html_content = '<html>...</html>'
result = extractor.process(html_content)
print(result[0].output_data.main_content)

# process multi HTML
html_list = ['<html>...</html>', '<html>...</html>']
results = extractor.process(html_list)
for result in results:
    print(result.output_data.main_content)
    print(result.case_id)
extractor.llm.cleanup()
```

#### Using Transformers Backend

```python
from mineru_html import MinerUHTML_Transformers, MinerUHTMLConfig

config = MinerUHTMLConfig(
    use_fall_back='trafilatura',
    prompt_version='short_compact',  # only support 'short_compact' for v1.1
    response_format='compact',       # only support 'compact' for v1.1
    early_load=True
)

# initialize MinerUHTML_Transformers
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

# process HTML
html_content = '<html>...</html>'
result = extractor.process(html_content)
print(result[0].output_data.main_content)
```

## 📖 Core Concepts

### Processing Pipeline

The MinerU-HTML processing pipeline includes the following steps:

1. **HTML Simplification** (`simplify_html`): Simplifies raw HTML into a structured format, assigning a unique `_item_id` attribute to each element
2. **Prompt Construction** (`build_prompt`): Constructs LLM prompts based on simplified HTML to guide the model in content classification
3. **LLM Inference** (`inference`): Uses LLM to classify each element, marking them as "main" (main content) or "other" (auxiliary content)
4. **Result Parsing** (`parse_result`): Parses the classification results returned by LLM
5. **Main Content Extraction** (`extract_main_html`): Extracts main content from original HTML based on classification results
6. **Format Conversion** (`convert2content`): Transform extracted HTML into Markdown, JSON and Txt.
7. **Fallback Processing**: If the above process fails, uses fallback mechanism (trafilatura, bypass or empty) for content extraction

### Configuration Options

`MinerUHTMLConfig` supports the following configurations:

- `use_fall_back`: Fallback type, optional `'trafilatura'`, `'bypass'`, or `'empty'`
- `early_load`: Whether to load the model early (default `True`)
- `prompt_version`: Prompt version, optional `'v0'`, `'v1'`, `'v2'`, `'compact'`, `'short_compact'`. The MinerUHTML and MinerUHTML_Transformers interfaces default to `'short_compact'`; the MinerUHTML_OpenAI interface defaults to `'v2'`
- `response_format`: The output format of the model. Only `'json'` or `'compact'` are permitted. VLLM/Transformers defaults to `'compact'`; OpenAI defaults to `'json'`
- `output_format`: The target format for Format Conversion, supporting `'mm_md'` (standard Markdown), `'md'` (Markdown with images), `'json'` and `'txt'`

#### Usage instructions for different prompt_version

- `'compact'`: Used for local model inference, it returns more concise results (only keeping the key and value in the JSON dictionary). It is recommended to use the `'compact'` model for faster inference speed.
- `'v2'`: Used for OpenAI API inference, it is the result after prompt optimization.

## 🔧 Advanced Usage

### Using Factory Functions to Create Backends

You can also directly use factory functions to create backends and then pass them to `MinerUHTMLGeneric`:

```python
from mineru_html import MinerUHTMLGeneric, MinerUHTMLConfig
from mineru_html.inference.factory import create_vllm_backend, create_transformers_backend

# Create VLLM backend using factory function
llm = create_vllm_backend(
    model_path='path/to/model',
    response_format='compact',
    max_context_window=32 * 1024,
    model_init_kwargs={'tensor_parallel_size': 1}
)

# Create Transformers backend using factory function
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

# Use the created backend
config = MinerUHTMLConfig()
extractor = MinerUHTMLGeneric(llm=llm, config=config)
```

### Error Handling

```python
from mineru_html.exceptions import MinerUHTMLError

try:
    result = extractor.process(html_content)
except MinerUHTMLError as e:
    print(f"Processing failed: {e}")
    print(f"Case ID: {e.case_id}")
```

## Baselines evaluation

To run the evaluation, you need to install the dependencies in `baselines.txt` first.

```bash
pip install -r baselines.txt
```

Then you can use the following command:

```bash

BENCHMARK_DATA=benchmark/WebMainBench_100.jsonl
RESULT_DIR=benchmark_results
mkdir $RESULT_DIR

# For MinerU-HTML
EXTRACTORS=(
"mineru_html_fallback-html-md"
)
MODEL_PATH=YOUR_MINERUHTML_MODEL_PATH

for extractor in ${EXTRACTORS[@]}; do
    python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
done

# For CPU Extractors
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

# For ReaderLM
extractor=readerlm-text
MODEL_PATH=YOUR_READERLM_MODEL_PATH

python eval_baselines.py --bench $BENCHMARK_DATA --task_dir $RESULT_DIR/$extractor --extractor_name  $extractor --model_path $MODEL_PATH --default_config gpu
```

MinerU-HTML supports various baseline extractors for comparison:
  - [**MinerU-HTML**](https://opendatalab.com/ai-ready/AICC#playground) (`mineru_html-html-md`, `mineru_html-html-text`): The main LLM-based extractor

  - [**Magic-HTML**](https://github.com/opendatalab/magic-html): CPU only HTML extraction tool, also from **OpenDatalab**

  - [**Trafilatura**](https://github.com/adbar/trafilatura): Fast and accurate content extraction

  - [**Readability**](https://github.com/mozilla/readability): Mozilla's readability algorithm

  - [**BoilerPy3**](https://github.com/jmriebold/BoilerPy3): Python port of Boilerpipe

  - [**NewsPlease**](https://github.com/fhamborg/news-please): News article extractor

  - [**Goose3**](https://github.com/goose3/goose3): Article extractor

  - [**GNE**](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor): General News Extractor

  - [**ReaderLM**](https://huggingface.co/jinaai/ReaderLM-v2): LLM-based text extractor


## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

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

If you use the extracted [AICC](https://huggingface.co/datasets/opendatalab/AICC) dataset, please cite:

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

## Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- Uses [Trafilatura](https://github.com/adbar/trafilatura) for fallback extraction
- Finetuned on [Hunyuan](https://github.com/Tencent-Hunyuan/Hunyuan-0.5B)
- Inspired by various HTML content extraction research
- Pairwise win rates LLM-as-a-judge by [dingo](https://github.com/MigoXLab/dingo)
