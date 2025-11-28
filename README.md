# MinerU-HTML(Dripper)

**MinerU-HTML(Dripper)** is an advanced HTML main content extraction tool based on Large Language Models (LLMs). It provides a complete pipeline for extracting primary content from HTML pages using LLM-based classification and state machine-guided generation.

## Features

- 🚀 **LLM-Powered Extraction**: Uses state-of-the-art language models to intelligently identify main content
- 🎯 **State Machine Guidance**: Implements logits processing with state machines for structured JSON output
- 🔄 **Fallback Mechanism**: Automatically falls back to alternative extraction methods on errors
- 📊 **Comprehensive Evaluation**: Built-in evaluation framework with ROUGE
- 🌐 **REST API Server**: FastAPI-based server for easy integration
- ⚡ **Distributed Processing**: Ray-based parallel processing for large-scale evaluation
- 🔧 **Multiple Extractors**: Supports various baseline extractors for comparison

## Evaluation Results
We evaluated MinerU-HTML on the [WebMainBench](https://github.com/ccprocessor/WebMainBench/) benchmark, which contains 7,887 meticulously annotated web pages along with their corresponding Markdown-formatted main content converted using `html2text`. This benchmark measures the extraction accuracy of content extractors by computing ROUGE-N scores between the extracted results and ground-truth content. The primary evaluation results are presented in the table below:

| Extractor                | ROUGE-N.f1 |
|--------------------------|------------|
| MinerU-HTML              | 0.8399     |
| GPT-5*                   | 0.8302     |
| DeepSeek-V3*             | 0.8252     |
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
        'tp': 1,  # Tensor parallel size
        'use_fall_back': True,
        'raise_errors': False,
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
    --state_machine None \
    --port 7986

# Or use environment variables
export DRIPPER_MODEL_PATH=/path/to/your/model
export DRIPPER_STATE_MACHINE=None
export DRIPPER_PORT=7986
python -m dripper.server
```

Then make requests to the API:

```bash
# Extract main content
curl -X POST "http://localhost:7986/extract" \
  -H "Content-Type: application/json" \
  -d '{"html": "<html>...</html>", "url": "https://example.com"}'

# Health check
curl http://localhost:7986/health
```

## Configuration

### Dripper Configuration Options

| Parameter       | Type | Default      | Description                                    |
| --------------- | ---- | ------------ | ---------------------------------------------- |
| `model_path`    | str  | **Required** | Path to the LLM model directory                |
| `tp`            | int  | 1            | Tensor parallel size for model inference       |
| `state_machine` | str  | None         | State machine version                          |
| `use_fall_back` | bool | True         | Enable fallback to trafilatura on errors       |
| `raise_errors`  | bool | False        | Raise exceptions on errors (vs returning None) |
| `debug`         | bool | False        | Enable debug logging                           |
| `early_load`    | bool | False        | Load model during initialization               |

### Environment Variables

- `DRIPPER_MODEL_PATH`: Path to the LLM model
- `DRIPPER_STATE_MACHINE`: State machine version (default:None)
- `DRIPPER_PORT`: Server port number (default: 7986)
- `VLLM_USE_V1`: Must be set to `'0'` when using state machine

## Usage Examples

### Batch Processing

```python
from dripper.api import Dripper

dripper = Dripper(config={'model_path': '/path/to/model'})

# Process multiple HTML strings
html_list = ["<html>...</html>", "<html>...</html>"]
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

## Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- Uses [Trafilatura](https://github.com/adbar/trafilatura) for fallback extraction
- Finetuned on [Qwen3](https://github.com/QwenLM/Qwen3)
- Inspired by various HTML content extraction research
- Pairwise win rates LLM-as-a-judge by [dingo](https://github.com/MigoXLab/dingo)
