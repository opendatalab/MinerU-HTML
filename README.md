# MinerU-HTML(Dripper)

**MinerU-HTML(Dripper)** is an advanced HTML main content extraction tool based on Large Language Models (LLMs). It provides a complete pipeline for extracting primary content from HTML pages using LLM-based classification and state machine-guided generation.

## News

- 2025.12.1 🎉 The [AICC](https://huggingface.co/datasets/opendatalab/AICC) dataset is released, welcome to use! AICC dataset contains 7.3T web pages extracted and converted to Markdown format by MinerU-HTML(Dripper), with cleaner main content and high-quality code, formulas, and tables.
- 2025.12.1 🎉 The [MinerU-HTML](https://huggingface.co/opendatalab/MinerU-HTML) model is released, welcome to use! MinerU-HTML model is a fine-tuned model on Qwen3, with better performance on HTML main content extraction.
- 2025.12.1 🎉 The trial page is online, welcome to visit [Opendatalab-AICC](https://opendatalab.com/ai-ready/AICC#playground) to try our extraction tool!

## Features

- 🚀 **LLM-Powered Extraction**: Uses state-of-the-art language models to intelligently identify main content
- 🎯 **State Machine Guidance**: Implements logits processing with state machines for structured JSON output
- 🔄 **Fallback Mechanism**: Automatically falls back to alternative extraction methods on errors
- 📊 **Comprehensive Evaluation**: Built-in evaluation framework with ROUGE
- 🌐 **REST API Server**: FastAPI-based server for easy integration
- ⚡ **Distributed Processing**: Ray-based parallel processing for large-scale evaluation
- 🔧 **Multiple Extractors**: Supports various baseline extractors for comparison

## Evaluation Results

We evaluated MinerU-HTML on the [WebMainBench](https://github.com/opendatalab/WebMainBench/) benchmark, which contains 7,887 meticulously annotated web pages along with their corresponding Markdown-formatted main content converted using `html2text`. This benchmark measures the extraction accuracy of content extractors by computing ROUGE-N scores between the extracted results and ground-truth content. The primary evaluation results are presented in the table below:

| Extractor                | ROUGE-N.f1 |
| ------------------------ | ---------- |
| MinerU-HTML              | 0.8399     |
| GPT-5\*                  | 0.8302     |
| DeepSeek-V3\*            | 0.8252     |
| MinerU-HTML(no fallback) | 0.8182     |
| Magic-HTML               | 0.7091     |
| Readability              | 0.6491     |
| Trafilatura              | 0.6358     |
| Resiliparse              | 0.6233     |
| html2text                | 0.5977     |
| BoilerPy3                | 0.5413     |
| GNE                      | 0.5148     |
| news-please              | 0.5012     |
| justText                 | 0.4770     |
| BoilerPy3                | 0.4766     |
| Goose3                   | 0.4354     |
| ReaderLM-v2              | 0.2264     |

where * denotes that use GPT-5/Deepseek-V3 to extract the main html in MinerU-HTML framework instead of our finetuned model.

## Installation

### Prerequisites

- Python >= 3.10
- CUDA-capable GPU (recommended for LLM inference)
- Sufficient memory for model loading

### Install from Source

The installation process automatically handles dependencies. The `setup.py` reads dependencies from `requirements.txt` and optionally from `baselines.txt`.

#### Basic Installation (Core Functionality)

For basic usage of Dripper, install with core dependencies only:

```bash
# Clone the repository
git clone https://github.com/opendatalab/MinerU-HTML
cd MinerU-HTML

# Install the package with core dependencies only
# Dependencies from requirements.txt are automatically installed
pip install .
```

#### Installation with Baseline Extractors (for Evaluation)

If you need to run baseline evaluations and comparisons, install with the `baselines` extra:

```bash
# Install with baseline extractor dependencies
pip install -e .[baselines]
```

This will install additional libraries required for baseline extractors:

- `magic-html` - CPU only HTML extraction tool, also from **OpenDatalab**
- `readabilipy`, `readability_lxml` - Readability-based extractors
- `resiliparse` - Resilient HTML parsing
- `justext` - JustText extractor
- `gne` - General News Extractor
- `goose3` - Goose3 article extractor
- `boilerpy3` - Boilerplate removal
- `crawl4ai` - AI-powered web content extraction

**Note**: The baseline extractors are only needed for running comparative evaluations. For basic usage of Dripper, the core installation is sufficient.

## Quick Start

### 1. Download the model

visit our model at [MinerU-HTML](https://huggingface.co/opendatalab/MinerU-HTML) and download the model, you can use the following command to download the model:

```bash
huggingface-cli download opendatalab/MinerU-HTML
```

### 2. Using the Python API

```python
from dripper.api import Dripper

# Initialize Dripper with model configuration
dripper = Dripper(
    config={
        'model_path': '/path/to/your/model',
        'use_fall_back': True,
        'raise_errors': False,
        'inference_backend': "vllm",
        "model_init_kwargs": {
          'tensor_parallel_size': 1,  # Tensor parallel size
        },
        "model_gen_kwargs": {
          'temperature': 0.0,
        },
    }
)

# Extract main content from HTML
html_content = """
<html>
  <body>
    <div>
    <h1>This is a title</h1>
    <p>This is a paragraph</p>
    <p>This is another paragraph</p>
    </div>
    <div>
    <p>Related content</p>
    <p>Advertising content</p>
    </div>
  </body>
</html>
"""
result = dripper.process(html_content)

# Access results
main_html = result[0].main_html
```

### 3. Using the REST API Server

```bash
# Start the server
python -m dripper.server \
    --model_path /path/to/your/model \
    --port 7986

# Or use environment variables
export DRIPPER_MODEL_PATH=/path/to/your/model
export DRIPPER_PORT=7986
python -m dripper.server
```

Then make requests to the API:

```bash
# Extract main content
curl -X POST "http://localhost:7986/extract" \
  -H "Content-Type: application/json" \
  -d '{"html": "<html><body>Hello World</body></html>", "url": "https://example.com"}'

# Health check
curl http://localhost:7986/health
```

## Configuration

### Dripper Configuration Options

| Parameter           | Type | Default      | Description                                    |
| ------------------- | ---- | ------------ | ---------------------------------------------- |
| `model_path`        | str  | **Required** | Path to the LLM model directory                |
| `state_machine`     | str  | None         | State machine version                          |
| `use_fall_back`     | bool | True         | Enable fallback to trafilatura on errors       |
| `raise_errors`      | bool | False        | Raise exceptions on errors (vs returning None) |
| `debug`             | bool | False        | Enable debug logging                           |
| `early_load`        | bool | False        | Load model during initialization               |
| `inference_backend` | str  | `vllm`       | The inference backend you want to use          |
| `model_init_kwargs` | dict | `{}`         | Parameters used during model initialization    |
| `model_gen_kwargs`  | dict | `{}`         | Parameters used during model inference         |

### Environment Variables

- `DRIPPER_MODEL_PATH`: Path to the LLM model
- `DRIPPER_STATE_MACHINE`: State machine version (default:None)
- `DRIPPER_PORT`: Server port number (default: 7986)
- `VLLM_USE_V1`: Must be set to `'0'` when using state machine
- `INFERENCE_BACKEND`: Inference backend to use (default `vllm`)
- `MODEL_INIT_KWARGS`: JSON string of model initialization kwargs (default `{}`)
- `MODEL_GEN_KWARGS`: JSON string of model inference kwargs (default `{}`)

## Usage Examples

### Batch Processing

```python
from dripper.api import Dripper

dripper = Dripper(config={'model_path': '/path/to/model'})

# Process multiple HTML strings
html_list = ["<html><body>Hello,</body></html>", "<html><body>World!</body></html>"]
results = dripper.process(html_list)

for result in results:
    print(result.main_html)
```

## Supported Extractors

Dripper supports various baseline extractors for comparison:

- [**Dripper**](https://opendatalab.com/ai-ready/AICC#playground) (`dripper-md`, `dripper-html`): The main LLM-based extractor

- [**Magic-HTML**](https://github.com/opendatalab/magic-html): CPU only HTML extraction tool, also from **OpenDatalab**

- [**Trafilatura**](https://github.com/adbar/trafilatura): Fast and accurate content extraction

- [**Readability**](https://github.com/mozilla/readability): Mozilla's readability algorithm

- [**BoilerPy3**](https://github.com/jmriebold/BoilerPy3): Python port of Boilerpipe

- [**NewsPlease**](https://github.com/fhamborg/news-please): News article extractor

- [**Goose3**](https://github.com/goose3/goose3): Article extractor

- [**GNE**](https://github.com/GeneralNewsExtractor/GeneralNewsExtractor): General News Extractor

- [**ReaderLM**](https://huggingface.co/jinaai/ReaderLM-v2): LLM-based text extractor

- [**Crawl4ai**](https://github.com/unclecode/crawl4ai): AI-powered web content extraction

- And more...

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENCE](LICENCE) file for details.

### Copyright Notice

This project contains code and model weights derived from Qwen3. Original Qwen3 Copyright 2024 Alibaba Cloud, licensed under Apache License 2.0. Modifications and additional training Copyright 2025 OpenDatalab Shanghai AILab, licensed under Apache License 2.0.

For more information, please see the [NOTICE](NOTICE) file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
- Finetuned on [Qwen3](https://github.com/QwenLM/Qwen3)
- Inspired by various HTML content extraction research
- Pairwise win rates LLM-as-a-judge by [dingo](https://github.com/MigoXLab/dingo)
