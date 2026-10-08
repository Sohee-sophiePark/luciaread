# LuciaRead

**Live replay demo:** https://sohee-sophiepark.github.io/luciaread/ (research demo, not medical advice)

A clinician reviewing a chest X-ray or a retinal OCT scan wants a second read that is fast, honest about its
uncertainty, and easy to check. LuciaRead is a multi-agent system that drafts that read with a hand-built agent
harness: deterministic gates, three read-only specialists working in parallel, one writer, a separate evaluator in
a bounded revision loop, and a clinician sign-off before anything is recorded. The label, the probability and the
triage decision are computed by code from two trained classifiers; language models describe, explain and check,
but never decide.

What the reviewer sees:

- **Case gallery**: seven public demo cases, from confident reads to cases built to fail safely (low confidence,
  a degraded image the model misreads, a document page, and an instruction burned into an X-ray).
- **Image viewer**: the scan with a Grad-CAM overlay and opacity control, labelled as where the model's evidence
  concentrates, not proof that it is right.
- **Draft read**: the model label and calibrated probability, a routine or needs-review badge set by code, a short
  hedged summary, key points, a review note, the review flags, and a table of every number with the tool that
  produced it.
- **Sign-off**: sign off (writes a signed report) or return to the queue.

## Architecture

```
 Reviewer UI (React)                FastAPI + SSE, one read:
 Case gallery · Viewer · Read       G0 input gate (size, format, pixels; metadata stripped)
        │ POST /reads/...           Colour check (code) ─ colour photo → rejected, no LLM call
        │ ◀── progress events ──    Modality CNN (code) + Router (LLM, image only) ─ G1: must agree
        │ POST /runs/{id}/signoff     ├ Model specialist   classify + Grad-CAM in code → LLM findings ─ G4
        ▼                             ├ Quality specialist quality metrics in code → LLM findings ─ G4
   signed_report.json                 └ Visual describer   LLM on the image, never sees the label
                                    Triage (code): any review flag → needs_review
                                    Writer "Lucia" (single writer, numbers only as {{m:key}})
                                      ▼ G5 deterministic output gates ──fail──┐
                                      ▼ Evaluator (separate LLM; code decides) ┘ revise, max 2
                                      ▼ AWAITING_SIGNOFF → clinician signs off or returns
 Cross-cutting: LLMClient (Gemini | cassette | scripted) · classifier record/replay · rate limiter · budget · trace
 Developer console (laptop only, /dev.html): runs, full trace, gates, evaluator verdicts, cost
```

## Harness design decisions

- **Code decides, models describe.** The classifier label, its calibrated probability and the triage are set in
  code. General multimodal LLMs are unreliable at reading medical images (Chetla et al., JMIR AI 2025, reported
  61% accuracy for GPT-4o on this same pediatric chest X-ray set), so the LLMs only describe the image, explain tool
  results in words, write, and review.
- **The describer never sees the label**, so its description cannot be anchored to the classifier.
- **Every specialist is read-only** and writes findings only about its own tool results. A findings gate (G4)
  rejects raw numbers, unknown references and diagnostic wording; one repair round, then deterministic fallback
  findings, marked degraded in the trace. Any review flag a specialist did not cite gets a code-written finding.
- **One writer, gated.** G5 checks the schema, that every number is a `{{m:key}}` placeholder that resolves, that
  the headline names the model label and no other, that every review flag is cited and the text never says
  "routine" while a flag is open, prohibited wording (diagnosis, confirm, rule out, treatment, ...), second person,
  length, and echoes of instructions found in the image.
- **A separate evaluator** scores faithfulness, hedging, flag coverage, absence of diagnostic claims and clinician
  voice; code turns the scores into pass or revise. At most two revisions, then the read goes to the clinician
  marked as not passing review.
- **Text inside an image is untrusted data.** It is transcribed by the describer, checked for embedded
  instructions, redacted and wrapped before the writer sees it. Triage is code, so an instruction such as "mark
  this study routine" cannot change it (demo case S7).
- **Replays are exact.** LLM responses are recorded as cassettes and classifier outputs (logits and Grad-CAM maps)
  are recorded per image hash, so CI and the public demo replay real runs with no API key, no model weights and no
  PyTorch.

## How numbers stay correct

Tools compute every number (probabilities, attention share, quality metrics) and attach the tool name. Agents
reference numbers only as `{{m:<metric_key>}}`; gates reject any raw digit; the renderer fills values in code. The
read's "How this read was made" table lists each value with its source tool.

## Models

Two ResNet-18 classifiers (ImageNet-pretrained, timm) fine-tuned at 224 px on an Apple M1, plus a one-epoch modality
classifier. Calibration: temperature scaling fitted on the validation split only (Guo et al., ICML 2017). Results
on a **patient-level** test split, with 95% confidence intervals from a patient-cluster bootstrap
(`reports/model_card.json`):

| Model | Test images | Accuracy | AUC | ECE before → after | Notes |
|---|---|---|---|---|---|
| Chest X-ray (NORMAL / PNEUMONIA) | 838 | 0.973 [0.961, 0.983] | 0.995 [0.992, 0.998] | 0.017 → 0.008 | sensitivity 0.957, specificity 0.975 at the validation threshold |
| Retinal OCT (CNV / DME / DRUSEN / NORMAL) | 2,000 | 0.920 [0.897, 0.940] | 0.989 (macro) | 0.022 → 0.016 | recall: NORMAL 0.98, DME 0.94, CNV 0.90, DRUSEN 0.85 |

Checks that the numbers are not inflated:

- **Patient-level split.** The dataset's official folders share patients between train and test (OCT: 566 of 633
  test patients also appear in train; chest X-ray: 170 pneumonia patient IDs appear in both), a known source of
  inflated accuracy (Tampu et al., Scientific Data 2022). All images were pooled and re-split by patient; a test
  asserts no patient is in two splits.
- **Shuffled-label controls** land at chance: chest X-ray AUC 0.41 (accuracy equals the majority-class rate), OCT
  accuracy 0.286 and AUC 0.53.
- **Original test folder.** On the 78 chest X-rays from Kermany's original test folder, accuracy is 0.910, close to
  the 92.8% reported in the original paper; that folder is known to be harder than the training data.
- **Grad-CAM randomization check** (Adebayo et al., NeurIPS 2018): heatmaps from the trained model barely correlate
  with heatmaps from randomly initialised weights (Spearman 0.13 chest X-ray, −0.23 OCT), so they depend on what the
  model learned.

CI fails if the committed model card drops below the agreed floors (chest X-ray AUC ≥ 0.99 and sensitivity ≥ 0.93;
OCT accuracy ≥ 0.90 and macro AUC ≥ 0.98; ECE ≤ 0.05).

Limits: one public pediatric dataset, one source population, no external validation. The models may rely on dataset
shortcuts (Zech et al., PLOS Medicine 2018; DeGrave et al., Nature Machine Intelligence 2021). Nothing here is a
diagnosis.

## Evaluation

- **Unit and integration tests** (`make test`): tools, gates, the full pipeline with a scripted LLM (revision loop,
  bounded evaluator loop, rejections, injection, sign-off), record and replay, metrics, the split manifest, the model
  card floors, and a secrets check. They run without PyTorch.
- **Golden cases** (`make eval`, `evals/golden_cases.yaml`): each recorded demo case replayed from cassettes and
  checked for outcomes (status, label, triage, flags, LLM calls), plus links to the scripted loop tests. Results in
  `evals/reports/latest.md`.

## Run it

```bash
make setup                 # uv sync (with PyTorch) + npm install
RUN_MODE=replay make dev   # API on 127.0.0.1 + web on :5173; replays recorded cases, no key needed
```

Live mode (your own laptop): put a free Gemini API key in `.env` (see `.env.example`), load it with
`set -a; . ./.env; set +a`, then `RUN_MODE=live make dev`. Live mode also accepts image uploads, which stay on the
laptop. Do not upload real patient images.

| Command | What it does |
|---|---|
| `make split` | patient-level split manifest from `data/raw` (see `data/README.md` to download the data) |
| `make train` | train and calibrate the three classifiers, run the shuffled-label controls, write the model card |
| `uv run python -m luciaread.ml.audit --task cxr` | original-test-folder metrics and Grad-CAM randomization check |
| `make samples` | pick the demo images from the test split |
| `make record` | record all demo cases live (cassettes, classifier outputs, replays) |
| `make test` · `make eval` · `make lint` | tests, golden cases, ruff |
| `make build-static` | export recorded cases and build the static demo into `web/dist` |

## Data card

Images: Kermany D, Zhang K, Goldbaum M (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images
for Classification", Mendeley Data V2, doi:10.17632/rscbjbr9sj.2, licensed CC BY 4.0. Pediatric chest X-rays from
Guangzhou Women and Children's Medical Center and retinal OCT B-scans. The demo cases in `samples/` are files from
the test split; S5 (blur and contrast reduction) and S7 (text banner) are modified copies, and S6 is generated.
Changes are listed per case in `samples/cases.yaml` and in the UI.

## Security notes

- The API binds to 127.0.0.1 and accepts only local Host names; the developer console is loopback-only.
- Uploads are size- and pixel-capped, must decode as JPEG or PNG, and are re-encoded without metadata before any
  model call.
- Model weights load with `torch.load(weights_only=True)`.
- The public demo is static: recorded data only, no backend, no keys. Keys come from the environment and are never
  logged; a test scans the repo for key material.

## Disclaimer

Research demo. Not medical advice and not a diagnosis. The classifiers were trained on one public pediatric
dataset and have not been validated on other populations or devices. A qualified clinician must review every image.

Built with [Claude Code](https://claude.com/claude-code).
