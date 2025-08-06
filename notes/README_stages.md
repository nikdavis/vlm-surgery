# Multi-Stage Dataset Generation Pipeline

## Overview
This pipeline supports multiple document types and includes an analysis stage to optimize prompts for each document type.

## Stages

### Stage 1: Flexible Data Staging
Use `stage1_flexible.py` to stage documents from various sources.

```bash
# Stage encyclopedia pages (original behavior)
uv run python stage1_flexible.py --source encyclopedia --seed 42 --pages 30

# Stage committee documents with random sampling
uv run python stage1_flexible.py --source committee --mode random --pages 30

# Stage specific committee volume
uv run python stage1_flexible.py --source committee --volume "pt01" --mode sequential --pages 20

# Custom source with pattern
uv run python stage1_flexible.py --source custom --pattern "my-docs-vol-\d+" --seed 42
```

Options:
- `--source`: encyclopedia, committee, or custom
- `--mode`: random, sequential, or contiguous
- `--pages`: number of pages to stage
- `--volume`: specific volume pattern to match
- `--seed`: for reproducible selection

Output naming:
- Encyclopedia: `enc_brit_v15_p0470.jpg`, `enc_brit_v15_p0470_ocr.txt`
- Committee: `mcc_comm_pt01_p0005.jpg`, `mcc_comm_pt01_p0005_ocr.txt`

### Stage 2.5: Analyze and Optimize Prompts
Use `stage2_5_analyze.py` to analyze samples and generate optimized prompts.

```bash
# Analyze committee documents
uv run python stage2_5_analyze.py --pattern mcc_comm --samples 5 --output committee_prompt.txt

# Analyze all staged documents
uv run python stage2_5_analyze.py --samples 8

# Analyze encyclopedia samples
uv run python stage2_5_analyze.py --pattern enc_brit --output encyclopedia_prompt_v2.txt
```

This will:
1. Select representative samples
2. Send them to Gemini with core ground rules
3. Analyze document structure and suggest appropriate XML schema
4. Generate an optimized prompt for that document type
5. Save the analysis and suggested prompt

### Stage 3: Generate Dataset
Use the optimized prompt from stage 2.5 or the default prompt.

```bash
# Use default prompt
uv run python generate_dataset.py

# Use custom prompt for committee documents
mv committee_prompt.txt gemini_prompt.txt
uv run python generate_dataset.py
```

## Full Workflow Example

```bash
# 1. Stage committee documents
uv run python stage1_flexible.py --source committee --mode random --pages 30

# 2.5. Analyze and get optimized prompt
uv run python stage2_5_analyze.py --pattern mcc_comm --output committee_prompt.txt

# Review the suggested prompt
cat committee_prompt.txt

# 3. Use the optimized prompt for generation
mv committee_prompt.txt gemini_prompt.txt
uv run python generate_dataset.py
```

## Benefits
- Handles different document types automatically
- Optimizes prompts based on actual document characteristics
- Maintains consistent output format across document types
- Supports incremental processing and restart