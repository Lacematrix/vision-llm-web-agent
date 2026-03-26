# Vision-LLM Web Agent

An autonomous web agent powered by Vision Language Models, LangChain, and Playwright that executes natural language instructions through multi-round interaction. Supports using vision language models through OpenAI-compatible APIs (can be local or remote).

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- If you use a local LLM
  - VRAM: Minimum 20GB (40GB+ recommended for better performance)

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/DreamWalkerXZ/vision-llm-web-agent.git
cd vision-llm-web-agent

# Initialize and install dependencies
uv sync

# Install Playwright browsers
uv run playwright install chromium

# Terrecat OCR Tool
You need to download the offline OCR tool of Terrecat. Please refer to https://blog.csdn.net/showgea/article/details/82656515 for more information.
```

### Usage

```bash
# Copy example environment variables
cp example.env .env

# Edit the .env file with your own API keys and model names.

# Run the agent
uv run python main.py
```

### Testing

```bash
# Test web agent completion rate
$env:RUN_AGENT_EVAL=1; uv run pytest tests/test_agent_completion_rate.py -k completion_rate -s
```

## 📦 Project Structure

```text
.
├── report
├── tests
│   ├── conftest.py
│   ├── test_agent_completion_rate.py
│   ├── test_browser_control.py
│   ├── test_file_operations.py
│   ├── test_information.py
│   └── test_waiting.py
├── vision_llm_web_agent
│   ├── config
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── browser_control.py
│   │   ├── dom_analyzer.py
│   │   ├── file_operations.py
│   │   ├── information.py
│   │   ├── registry.py
│   │   └── waiting.py
│   ├── __init__.py
│   ├── agent_controller.py
│   └── vllm_client.py
├── example.env
├── main.py
├── pyproject.toml
└── README.md
```

## 🛠️ Tools Provided

| Program File Name    | Tool Name            | Description                  | Implemented | Tested |
| -------------------- | -------------------- | ---------------------------- | ----------- | ------ |
| `browser_control.py` | `goto`               | Navigate to URL              | ✅           | ✅      |
| `browser_control.py` | `click`              | Click element                | ✅           | ✅      |
| `browser_control.py` | `type_text`          | Type into input              | ✅           | ✅      |
| `browser_control.py` | `press_key`          | Press keyboard keys          | ✅           | ✅      |
| `waiting.py`         | `wait_seconds`       | Wait for duration            | ✅           | ✅      |
| `information.py`     | `screenshot`         | Capture page screenshot      | ✅           | ✅      |
| `information.py`     | `dom_summary`        | Get simplified DOM structure | ✅           | ✅      |
| `information.py`     | `ocr`                | Extract text from images     | ✅           | ✅      |
| `file_operations.py` | `download_pdf`       | Download PDF files           | ✅           | ✅      |
| `file_operations.py` | `pdf_extract_text`   | Extract text from PDFs       | ✅           | ✅      |
| `file_operations.py` | `pdf_extract_images` | Extract images from PDFs     | ✅           | ✅      |
| `file_operations.py` | `save_image`         | Save/crop images             | ✅           | ✅      |
| `file_operations.py` | `write_text`         | Write text to files          | ✅           | ✅      |

- ✅ Implemented
- ❌ Not implemented
- ❓ Problematic

## 📋 Example Task

```text
Find the most recent technical report (PDF) about Qwen, 
then interpret Figure 1 by describing its purpose and key findings.
```

## 📊 Output Artifacts

After execution, check the `artifacts/` directory for:

- Step-by-step screenshots (`step_0.png`, `step_1.png`, ...)
- Execution log (`execution_log.json`)
- Downloaded PDFs
- Extracted images

## 🔧 Roadmap

- [x] Implement and test pdf related tools.
- [x] Implement and test the ocr tool.
- [x] Multiple rounds of interaction.
- [x] Save the final answer to a txt file.
- [x] Implement locale alignment and check whether it is useful.
- [x] Make the agent more robust and reliable (maybe tuning the dom_summary tool and prompt).
- [x] Add more tests.
- [x] Add supported vision LLMs list.
- [x] Add hardware requirements.
- [ ] Add license.
- [ ] Add documentation.
