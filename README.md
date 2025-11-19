
An AI agent (built with **LangGraph**) that reads a natural-language description of a **cryptographic protocol** and produces an **Alice–Bob** style flow. It runs a generator → optimizer loop, validates with simple heuristics, and renders ** ASCII **.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI Usage](#cli-usage)
  - [Single file](#single-file)
  - [Batch over a dataset](#batch-over-a-dataset)
  - [Common arguments](#common-arguments)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [Benchmark](#benchmark)
  - [Sample Results](#sample-results)
- [Dataset Layout](#dataset-layout)
- [Troubleshooting](#troubleshooting)

---

## Features
- Convert natural-language protocol descriptions into **Alice–Bob** message flows
- **Generator → Optimizer** loop with simple heuristic validation
- renderer: **ASCII**
- Command-line interface for **single** files and **batch** datasets

---

## Requirements
- Python **3.10+** (3.11 recommended)
- An LLM provider key (e.g., OpenAI, Google, etc.)
- A POSIX-like shell (macOS/Linux); Windows via WSL or PowerShell

---

## Installation

1) Create a virtual environment and install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

1) Place your API keys in a `.env` file at the project root:
```dotenv
# Example — include the keys your setup requires
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
K2_THINK_API_KEY=...

### models: gpt-4.1,gpt-5.1,gpt-5-mini,gemini-2.5-pro,k2-think
LLM_OPTIMIZER="k2-think"
LLM_EVALUATOR="gpt-5.1"
```

2) Run on a single example:
```bash
cd src/

python -m anbProtocol.cli single --in ../examples/tls_like.txt --out out.md
```
---

## CLI Usage

### Single file
Parse one natural-language description and write the result:
```bash
python -m AnBProtocol.cli single --in <path-to-input>.txt --out <path-to-output>.md
# or (depending on package name)
python -m anbprotocol.cli single --in <path-to-input>.txt --out <path-to-output>.md
```

### Batch over a dataset
Reproduce AnB files for all items in a dataset:
```bash
python -m anbprotocol.cli batch \
  --dataset ../dataset/natural_language \
  --output ../outputResults_gemini-2.5-pro_gpt5.
```

### Common arguments
- `--in` *(single)*: input `.txt` file with the natural-language description  
- `--out` *(single)*: output file (e.g., `.md`, `.json`)  
- `--dataset` *(batch)*: directory containing input `.txt` files  
- `--output` *(batch)*: directory to write generated results  

---

## Configuration

This project reads configuration from the environment. A simple `.env` file is supported:

```dotenv
# Core
OPENAI_API_KEY=...
GOOGLE_API_KEY=...

```



---

## Output Formats

The agent can render outputs in:
- **ASCII** Alice–Bob flows


---

## Benchmark

Reproduce the benchmark:
```bash
python benchmark/benchmark.py
```

### Sample Results

Output of debate between **Gemini-2.5-pro** and **GPT-5.1**:

<details>
<summary>Per-file metrics (click to expand)</summary>

```
                                        file  gt_clauses  pred_clauses  TP  FN  FP  ExactCoverage_EC  BoundedErrorRate_BER  JaccardIndex
0                                     1.anb          15            24   7   8  17          0.466667              0.641026      0.218750
1                                    10.anb          23            28  15   8  13          0.652174              0.411765      0.416667
2                                    11.anb          29            32  18  11  14          0.620690              0.409836      0.418605
3                                    12.anb          31            35  12  19  23          0.387097              0.636364      0.222222
4                                  1tls.anb          29            34  12  17  22          0.413793              0.619048      0.235294
5                                     2.anb          16            27   7   9  20          0.437500              0.674419      0.194444
6   3isosymkeyonepassunilateralauthprot.anb           8            22   3   5  19          0.375000              0.800000      0.111111
7                                     4.anb          20            29   9  11  20          0.450000              0.632653      0.225000
8                                     5.anb          18            25  16   2   9          0.888889              0.255814      0.592593
9                                     8.anb          24            26  16   8  10          0.666667              0.360000      0.470588
10                                    9.anb          24            28  12  12  16          0.500000              0.538462      0.300000
11                         amended-nsck.anb          34            38  23  11  15          0.676471              0.361111      0.469388
12           andrewsecurerpcsecrecy_fix.anb          24            28  17   7  11          0.708333              0.346154      0.485714
13                                  asw.anb          22            29  14   8  15          0.636364              0.450980      0.378378
14                                 asym.anb          16            22  11   5  11          0.687500              0.421053      0.407407
15                   basic-kerberos_fix.anb          21            32  12   9  20          0.571429              0.547170      0.292683
16               bilateral-key_exchange.anb          21            27  16   5  11          0.761905              0.333333      0.500000
17                                   cr.anb          19            26  10   9  16          0.526316              0.555556      0.285714
18                        denning-sacco.anb          18            29  10   8  19          0.555556              0.574468      0.270270
19                 denning-sacco_public.anb          20            27  12   8  15          0.600000              0.489362      0.342857
20                                   dh.anb          19            27  12   7  15          0.631579              0.478261      0.352941
21      isoccfonepassunilateralauthprot.anb          15            20   7   8  13          0.466667              0.600000      0.250000
22                isoccfthreepassmutual.anb          18            22   5  13  17          0.277778              0.750000      0.142857
23     isoccftwopassmutualauthprot-corr.anb          20            24  12   8  12          0.600000              0.454545      0.375000
24   isopubkeyonepassunilateralauthprot.anb          18            21   8  10  13          0.444444              0.589744      0.258065
25  isopubkeytwopassmutualauthprot-corr.anb          20            23  11   9  12          0.550000              0.488372      0.343750
26   isopubkeytwopassunilateralauthprot.anb          19            25  12   7  13          0.631579              0.454545      0.375000
27             isosymkeythreepassmutual.anb          17            26   8   9  18          0.470588              0.627907      0.228571
28       isosymkeytwopassmutualauthprot.anb          16            24   8   8  16          0.500000              0.600000      0.250000
29   isosymkeytwopassunilateralauthprot.anb          15            20   8   7  12          0.533333              0.542857      0.296296
30                        nonreversible.anb          19            30  10   9  20          0.526316              0.591837      0.256410
31                                 nsck.anb          22            34  12  10  22          0.545455              0.571429      0.272727
32                               nsl-ks.anb          22            30   9  13  21          0.409091              0.653846      0.209302
33                                 nspk.anb          21            30  12   9  18          0.571429              0.529412      0.307692
34                                 nssk.anb          25            32   7  18  25          0.280000              0.754386      0.140000
35                                   or.anb          21            27   8  13  19          0.380952              0.666667      0.200000
36                               pkinit.anb          29            30  10  19  20          0.344828              0.661017      0.204082
37                            signed_dh.anb          19            30  11   8  19          0.578947              0.551020      0.289474
38                                  toy.anb          19            22   8  11  14          0.421053              0.609756      0.242424
39                              yahalom.anb          20            29  10  10  19          0.500000              0.591837      0.256410
```

</details>

**Summary:**
```json
{
  "num_files": 40,
  "EC_weighted_by_GT": 0.5326876513317191,
  "EC_macro_avg": 0.5311596509808736,
  "BER_weighted": 0.5416666666666666,
  "BER_macro_avg": 0.5456502230252858,
  "Jaccard_weighted": 0.2972972972972973,
  "Jaccard_macro": 0.3022172257961973
}
```

---

## Dataset Layout

If you use the batch command, the dataset directory might look like:
```
dataset/
└── natural_language/
    ├── 1.txt
    ├── 2.txt
    ├── ...
    └── n.txt
```

and outputs will be written to:
```
outputResults_gemini-2.5-pro_gpt5./
├── 1.anb
├── 2.anb
└── ...
```



---

## Troubleshooting

- **Module not found (`AnBProtocol.cli` vs `anbprotocol.cli`)**  
  Depending on how the package is installed or named, the import path can be either `AnBProtocol` or `anbprotocol`. Try the alternative capitalization if one fails.

- **No API key / auth error**  
  Ensure your `.env` is loaded in the current shell and includes the provider key you need (e.g., `OPENAI_API_KEY` or `GOOGLE_API_KEY`).



