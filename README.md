
An AI agent workflow (built with **LangGraph**) that takes a natural-language description of a **cryptographic protocol** and synthesizes an **Alice–Bob** style message flow. It uses a generator → optimizer loop, validates and refines the flow via a second-LLM “debate” phase, and finally renders the result as clean **ASCII**.

A second stage then performs **security evaluation** using LLM-based reasoning against the following criteria:

1. Correctness  
2. Secrecy  
3. Forward Secrecy  
4. Mutual Authentication  
5. Resistance against Key Compromise Impersonation (KCI)  
6. Resistance against Identity Mis-Binding  
7. Resistance against Replay  
8. Session Uniqueness  
9. Channel Binding  
10. Identity Hiding
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
  - **Generator → Optimizer** loop with second LLM 
  - renderer: **ASCII**
  - Command-line interface for **single** files and **batch** datasets
- Security reasoning for Alice-bob style cryptographic protocol

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
LLM_REASONING="gpt-5.1"
```

2) Run on a single example(AnB Formating):
```bash
cd src/

python -m anbprotocol.cli single --in ../dataset/natural_language/1.txt --out 1.anb
```
3)Run security reasoning:
```bash

 python -m securityReasoning.cli  --in ../dataset/anb/1.anb  --adv A --out result.txt
```
---

## CLI Usage

### Single file
Parse one natural-language description and write the result:
```bash
python -m anbprotocol.cli single --in <path-to-input>.txt --out <path-to-output>.txt
# or (depending on package name)
python -m anbprotocol.cli single --in <path-to-input>.txt --out <path-to-output>.txt
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
- `--out` *(single)*: output file (e.g., `.md`, `.txt`)  
- `--dataset` *(batch)*: directory containing input `.txt` files  
- `--output` *(batch)*: directory to write generated results  

---

## Configuration

This project reads configuration from the environment. A simple `.env` file is supported:

```dotenv
# Example — include the keys your setup requires
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
K2_THINK_API_KEY=...

### models: gpt-4.1,gpt-5.1,gpt-5-mini,gemini-2.5-pro,k2-think
LLM_OPTIMIZER="k2-think"
LLM_EVALUATOR="gpt-5.1"
LLM_REASONING="gpt-5.1"
```



---

## Output Formats

The agent can render outputs in:
- **ASCII** Alice–Bob flows
- **ASCII** Security Evaluation

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
                                        file  gt_clauses  pred_clauses  TP  FN  FP  ExactCoverage_EC  BoundedErrorRate_BER  JaccardIndex   ROUGE_L
0                                  asym.anb          12            12  12   0   0          1.000000              0.000000      1.000000  1.000000
1                                  nspk.anb          17            18  17   0   1          1.000000              0.028571      0.944444  0.979592
2                  denning-sacco_public.anb          18            20  18   0   2          1.000000              0.052632      0.900000  0.956522
3                         denning-sacco.anb          18            20  18   0   2          1.000000              0.052632      0.900000  0.956522
4                                nsl-ks.anb          20            21  19   1   2          0.950000              0.073171      0.863636  0.909091
5                                  nssk.anb          23            23  21   2   2          0.913043              0.086957      0.840000  0.928571
6                               yahalom.anb          20            20  18   2   2          0.900000              0.100000      0.818182  0.916667
7                                     2.anb          17            19  15   2   4          0.882353              0.166667      0.714286  0.863636
8                                   toy.anb          16            18  14   2   4          0.875000              0.176471      0.700000  0.789474
9                                     1.anb          16            16  14   2   2          0.875000              0.125000      0.777778  0.829268
10   isosymkeytwopassunilateralauthprot.anb          14            14  12   2   2          0.857143              0.142857      0.750000  0.866667
11                   basic-kerberos_fix.anb          20            20  17   3   3          0.850000              0.150000      0.739130  0.875000
12   isopubkeyonepassunilateralauthprot.anb          13            15  11   2   4          0.846154              0.214286      0.647059  0.800000
13                                    8.anb          19            21  16   3   5          0.842105              0.200000      0.666667  0.809524
14                                   10.anb          20            19  16   4   3          0.800000              0.179487      0.695652  0.772727
15  3isosymkeyonepassunilateralauthprot.anb          14            15  11   3   4          0.785714              0.241379      0.611111  0.717949
16           andrewsecurerpcsecrecy_fix.anb          20            22  15   5   7          0.750000              0.285714      0.555556  0.806452
17                         amended-nsck.anb          27            29  20   7   9          0.740741              0.285714      0.555556  0.789474
18       isosymkeytwopassmutualauthprot.anb          15            15  11   4   4          0.733333              0.266667      0.578947  0.750000
19               bilateral-key_exchange.anb          20            23  13   7  10          0.650000              0.395349      0.433333  0.641509
20                                    5.anb          20            19  13   7   6          0.650000              0.333333      0.500000  0.666667
21                                   cr.anb          17            18  11   6   7          0.647059              0.371429      0.458333  0.666667
22      isoccfonepassunilateralauthprot.anb          14            14   9   5   5          0.642857              0.357143      0.473684  0.666667
23     isoccftwopassmutualauthprot-corr.anb          15            15   9   6   6          0.600000              0.400000      0.428571  0.606061
24                                    4.anb          15            18   9   6   9          0.600000              0.454545      0.375000  0.604651
25                isoccfthreepassmutual.anb          17            16  10   7   6          0.588235              0.393939      0.434783  0.628571
26                                  asw.anb          18            20  10   8  10          0.555556              0.473684      0.357143  0.536585
27                                 etma.anb          20            18  11   9   7          0.550000              0.421053      0.407407  0.652174
28                                    9.anb          19            19  10   9   9          0.526316              0.473684      0.357143  0.511628
29                            signed_dh.anb          23            24  12  11  12          0.521739              0.489362      0.342857  0.555556
30   isopubkeytwopassunilateralauthprot.anb          14            18   7   7  11          0.500000              0.562500      0.280000  0.470588
31                        nonreversible.anb          18            21   9   9  12          0.500000              0.538462      0.300000  0.530612
32                                 nsck.anb          22            23  11  11  12          0.500000              0.511111      0.323529  0.480000
33             isosymkeythreepassmutual.anb          15            18   7   8  11          0.466667              0.575758      0.269231  0.457143
34                                   12.anb          25            25  11  14  14          0.440000              0.560000      0.282051  0.571429
35  isopubkeytwopassmutualauthprot-corr.anb          16            16   7   9   9          0.437500              0.562500      0.280000  0.470588
36                                   dh.anb          15            18   6   9  12          0.400000              0.636364      0.222222  0.461538
37                                 1tls.anb          24            26   9  15  17          0.375000              0.640000      0.219512  0.428571
38                                   11.anb          25            27   9  16  18          0.360000              0.653846      0.209302  0.454545
39                               pkinit.anb          25            27   8  17  19          0.320000              0.692308      0.181818  0.322581 
 
```

</details>

**Summary:**
```json
{'num_files': 40,
  'EC_weighted_by_GT': 0.6739130434782609,
  'BER_weighted': 0.34564643799472294,
  'Jaccard_weighted': 0.48627450980392156,
  'ROUGE_L_macro_avg': 0.6925366482630825
}
```

<details>
<summary>Example of Security Evaluation Report</summary>

```
=== Security Evaluation Summary ===
Adversary              : Passive Adversary Model
Total goals evaluated  : 10
Goals satisfied (yes)  : 2
Goals violated (no)    : 8
Success rate           : 20.0% yes / 80.0% no
========================================

Protocol one:

Declarations:
    senc/2;

Types:
    Agent A,B;
    Number NA;
    Data M;
    Symmetric_key Ks;

Knowledge:
    A : A, B, Ks, M;
    B : A, B, Ks;

Public:
    A,B;
Private:
    Ks, M;

Actions:
    [m1] A -> B (Na): Na;
    [m2] B -> A     : senc{Na}Ks;
    [m3] A -> B     : senc{M}Ks;

Goals:
    B -> A : NA;
    M secret between A,B;

ChannelKeys:

end

=== Security Evaluation ===
---1: Correctness ---
Result: yes
Reasoning:
Both honest parties end the protocol using the same pre-shared symmetric key Ks. The initiator A already knows Ks and uses it to encrypt Na and M; the responder B also knows Ks and successfully decrypts senc{Na}Ks in m2 and senc{M}Ks in m3. Since no key agreement or derivation occurs (Ks is fixed and shared beforehand), and all messages are unmodified, both sides necessarily use the identical key Ks for transport data, so correctness holds in the pre‑shared‑key sense.

---2: Secrecy of transport data & traffic keys ---
Result: yes
Reasoning:
NA and M are sent only either in clear over an honest endpoint channel ([m1] Na) or encrypted with the private symmetric key Ks ([m2] senc{Na}Ks, [m3] senc{M}Ks). The traffic key is Ks, which is declared private and not initially known to the attacker. Under a passive adversary and ideal cryptography, the attacker cannot derive NA from senc{Na}Ks nor M from senc{M}Ks, so Transport‑Secrecy holds.

---3: Forward Secrecy ---
Result: no
Reasoning:
Session keys are never used; all encryption uses the long-term symmetric key Ks, which is later revealed to the adversary by assumption. Once Ks is learned, the adversary can decrypt the recorded ciphertexts senc{M}Ks from past sessions and reconstruct the cleartext M, so forward secrecy does not hold.

---4: Mutual Authentication ---
Result: no
Reasoning:
Condition (1) fails: A never authenticates B. A accepts m2 = senc{Na}Ks as valid just because A also knows Ks; a passive adversary cannot break crypto, but the property requires that “if an honest initiator completes claiming to talk to B, then B must have participated.” Here, if later B’s long‑term key Ks is leaked and a passive eavesdropper had recorded traffic, the eavesdropper could have generated a matching transcript offline, so the model does not guarantee that B *actually* participated in that specific run.

no  
Condition (2) fails: There is no authentication on data M, only encryption with Ks. Under later key compromise, a passive adversary who recorded traffic and then obtained Ks could forge ciphertexts senc{M'}Ks that A would accept as from B, violating the requirement that any accepted plaintext from B must have been intentionally sent by B on that channel.

---5: Resistance against Key Compromise Impersonation (KCI) ---
Result: no
Reasoning:
B’s long-term symmetric key Ks is compromised, so the attacker can compute senc{Na}Ks and senc{M}Ks just like A. This lets the attacker impersonate A to B and make B accept protocol messages as if they were sent by A, violating resistance to KCI.

---6: Resistance against Identity mis-binding / unknown key-share ---
Result: no
Reasoning:
The protocol is not resistant to identity mis‑binding / unknown key‑share attacks.

The traffic/session key Ks is a fixed pre‑shared symmetric key and is not derived from any fresh or identity‑bound inputs (no nonces, roles, or identities are used to derive Ks). Messages:
- A → B: Na
- B → A: senc{Na}Ks
- A → B: senc{M}Ks

do not cryptographically bind the peer’s identity to the key. Under a passive adversary, the protocol runs correctly but provides no guarantee that the party holding Ks is actually the intended peer (only “whoever also knows Ks”). Thus two honest parties can share Ks while each associates it with a different peer identity, and nothing in the message flow or key derivation prevents this mis‑binding.

---7: Resistance against Replay Attack ---
Result: no
Reasoning:
B accepts any ciphertext senc{Na}Ks in m2 without linking it to a unique session or storing used nonces. An eavesdropper can record a valid m2 from an earlier run and later replay it after a fresh m1. Since B has no mechanism to detect reuse of Na or of the ciphertext, it will accept the replayed m2 as valid again.

---8: Session Uniqueness ---
Result: no
Reasoning:
The protocol does not even define a traffic key from session-dependent parameters: Ks is a fixed long-term symmetric key, reused across all runs. Thus every honest initiator (A) and responder (B) session trivially uses the same traffic key Ks, so multiple distinct sessions share the same key and (for identical Na choices) can also share the same handshake transcript.

---9: Channel Binding ---
Result: no
Reasoning:
Different sessions can produce the same final transcript but correspond to different internal states/keys. The transcript only shows Na in clear and then two ciphertexts under Ks, but Ks and M are not bound to the transcript (they are not derived from Na or other transcript values). Thus two runs with the same visible messages but different underlying Ks or M would be indistinguishable at the transcript level, violating channel binding.

---10: Identity Hiding ---
Result: no
Reasoning:
The static identities A and B are public and appear as known agents in clear; an eavesdropper can trivially tell that A and B are the communicating parties in this session, so the protocol does not hide which long-term identities participate.

```

</details>



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



