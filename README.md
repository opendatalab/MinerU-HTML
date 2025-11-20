# Dripper

**Dripper** is an advanced HTML main content extraction tool based on Large Language Models (LLMs). It provides a complete pipeline for extracting primary content from HTML pages using LLM-based classification and state machine-guided generation.

## Features

- 🚀 **LLM-Powered Extraction**: Uses state-of-the-art language models to intelligently identify main content
- 🎯 **State Machine Guidance**: Implements logits processing with state machines for structured JSON output
- 🔄 **Fallback Mechanism**: Automatically falls back to alternative extraction methods on errors
- 📊 **Comprehensive Evaluation**: Built-in evaluation framework with ROUGE and item-level metrics
- 🌐 **REST API Server**: FastAPI-based server for easy integration
- ⚡ **Distributed Processing**: Ray-based parallel processing for large-scale evaluation
- 🔧 **Multiple Extractors**: Supports various baseline extractors for comparison

## Installation

### Prerequisites

- Python >= 3.12
- CUDA-capable GPU (recommended for LLM inference)
- Sufficient memory for model loading

### Install from Source

The installation process automatically handles dependencies. The `setup.py` reads dependencies from `requirements.txt` and optionally from `baselines.txt`.

#### Basic Installation (Core Functionality)

For basic usage of Dripper, install with core dependencies only:

```bash
# Clone the repository
git clone <repository-url>
cd MinerU-HTML

# Install the package with core dependencies only
# Dependencies from requirements.txt are automatically installed
pip install -e .
```

#### Installation with Baseline Extractors (for Evaluation)

If you need to run baseline evaluations and comparisons, install with the `baselines` extra:

```bash
# Install with baseline extractor dependencies
pip install -e .[baselines]
```

This will install additional libraries required for baseline extractors:

- `readabilipy`, `readability_lxml` - Readability-based extractors
- `resiliparse` - Resilient HTML parsing
- `justext` - JustText extractor
- `gne` - General News Extractor
- `goose3` - Goose3 article extractor
- `boilerpy3` - Boilerplate removal
- `crawl4ai` - AI-powered web content extraction

**Note**: The baseline extractors are only needed for running comparative evaluations. For basic usage of Dripper, the core installation is sufficient.

## Quick Start

### 1. Using the Python API

```python
from dripper.api import Dripper

# Initialize Dripper with model configuration
dripper = Dripper(
    config={
        'model_path': '/path/to/your/model',
        'tp': 1,  # Tensor parallel size
        'state_machine': 'v2',  # or 'v1', or None
        'use_fall_back': True,
        'raise_errors': False,
    }
)

# Extract main content from HTML
html_content = "<html>...</html>"
result = dripper.process(html_content)

# Access results
main_html = result[0].main_html
main_content = result[0].main_content
```

### 2. Using the REST API Server

```bash
# Start the server
python -m dripper.server \
    --model_path /path/to/your/model \
    --state_machine v2 \
    --port 7986

# Or use environment variables
export DRIPPER_MODEL_PATH=/path/to/your/model
export DRIPPER_STATE_MACHINE=v2
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

| Parameter       | Type | Default      | Description                                      |
| --------------- | ---- | ------------ | ------------------------------------------------ |
| `model_path`    | str  | **Required** | Path to the LLM model directory                  |
| `tp`            | int  | 1            | Tensor parallel size for model inference         |
| `state_machine` | str  | None         | State machine version: `'v1'`, `'v2'`, or `None` |
| `use_fall_back` | bool | True         | Enable fallback to trafilatura on errors         |
| `raise_errors`  | bool | False        | Raise exceptions on errors (vs returning None)   |
| `debug`         | bool | False        | Enable debug logging                             |
| `early_load`    | bool | False        | Load model during initialization                 |

### Environment Variables

- `DRIPPER_MODEL_PATH`: Path to the LLM model
- `DRIPPER_STATE_MACHINE`: State machine version (`v1`, `v2`, or empty)
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

### Evaluation

#### Baseline Evaluation

```bash
python app/eval_baseline.py \
    --bench /path/to/benchmark.jsonl \
    --task_dir /path/to/output \
    --extractor_name dripper-md \
    --default_config gpu \
    --model_path /path/to/model
```

#### Two-Step Evaluation

```bash
# Step 1: Generate predictions
python app/run_inference.py \
    --bench /path/to/benchmark.jsonl \
    --task_dir /path/to/output \
    --model_path /path/to/model \
    --state_machine v2

# Step 2: Evaluate with answers
python app/eval_with_answer.py \
    --bench /path/to/benchmark.jsonl \
    --task_dir /path/to/output \
    --answer /path/to/answers.jsonl
```

## Project Structure

```
MinerU-HTML/
├── dripper/                 # Main package
│   ├── api.py              # Dripper API class
│   ├── server.py           # FastAPI server
│   ├── base.py             # Core data structures
│   ├── exceptions.py        # Custom exceptions
│   ├── inference/          # LLM inference modules
│   │   ├── inference.py    # Generation functions
│   │   ├── prompt.py       # Prompt generation
│   │   ├── logits.py       # Response parsing
│   │   └── logtis_processor/  # State machine logits processors
│   ├── process/            # HTML processing
│   │   ├── simplify_html.py
│   │   ├── map_to_main.py
│   │   └── html_utils.py
│   ├── eval/               # Evaluation modules
│   │   ├── metric.py       # ROUGE and item-level metrics
│   │   ├── eval.py         # Evaluation functions
│   │   ├── process.py      # Processing utilities
│   │   └── benckmark.py    # Benchmark data structures
│   └── eval_baselines/     # Baseline extractors
│       ├── base.py         # Evaluation framework
│       └── baselines/       # Extractor implementations
├── app/                    # Application scripts
│   ├── eval_baseline.py    # Baseline evaluation script
│   ├── eval_with_answer.py # Two-step evaluation
│   ├── run_inference.py    # Inference script
│   └── process_res.py     # Result processing
├── requirements.txt        # Core Python dependencies (auto-installed)
├── baselines.txt          # Optional dependencies for baseline extractors
├── LICENCE                # Apache License 2.0
├── NOTICE                 # Copyright and attribution notices
└── setup.py               # Package setup (handles dependency installation)
```

## Supported Extractors

Dripper supports various baseline extractors for comparison:

- **Dripper** (`dripper-md`, `dripper-html`): The main LLM-based extractor
- **Trafilatura**: Fast and accurate content extraction
- **Readability**: Mozilla's readability algorithm
- **BoilerPy3**: Python port of Boilerpipe
- **NewsPlease**: News article extractor
- **Goose3**: Article extractor
- **GNE**: General News Extractor
- **Crawl4ai**: AI-powered web content extraction
- And more...

## Evaluation Metrics

- **ROUGE Scores**: ROUGE-N precision, recall, and F1 scores
- **Item-Level Metrics**: Per-tag-type (main/other) precision, recall, F1, and accuracy
- **HTML Output**: Extracted main HTML for visual inspection

## Development

### Running Tests

```bash
# Add test commands here when available
```

### Code Style

The project uses pre-commit hooks for code quality. Install them:

```bash
pre-commit install
```

## Troubleshooting

### Common Issues

1. **VLLM_USE_V1 Error**: When using state machine, ensure `VLLM_USE_V1=0` is set:

   ```bash
   export VLLM_USE_V1=0
   ```

2. **Model Loading Errors**: Verify model path and ensure sufficient GPU memory

3. **Import Errors**: Ensure the package is properly installed:

   ```bash
   # Reinstall the package (this will automatically install dependencies from requirements.txt)
   pip install -e .

   # If you need baseline extractors for evaluation:
   pip install -e .[baselines]
   ```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENCE](LICENCE) file for details.

### Copyright Notice

This project contains code and model weights derived from Qwen3. Original Qwen3 Copyright 2024 Alibaba Cloud, licensed under Apache License 2.0. Modifications and additional training Copyright 2025 OpenDatalab Shanghai AILab, licensed under Apache License 2.0.

For more information, please see the [NOTICE](NOTICE) file.

## Citation

If you use Dripper in your research, please cite:

```bibtex
[Add citation information]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on top of [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- Uses [Ray](https://www.ray.io/) for distributed processing
- Inspired by various HTML content extraction research

## Contact

\[Add contact information or links to issues/discussions\]
