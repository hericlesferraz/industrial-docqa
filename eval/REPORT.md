# Eval report

Latest committed report of the eval harness (`make eval`, see the README section "Eval"). Two runs, each in a fresh
tenant through the API only:

1. **Scored run**: the 4 PDFs of `case_files/` and the three question sets together:
   `make eval QUESTIONS="eval/public_questions.jsonl eval/questions_design.jsonl eval/questions_holdout.jsonl"`.
2. **Robustness run**: the files built by `eval/fixtures/make_fixtures.py` (a scanned page, a locked PDF, a truncated
   PDF, an all-blank PDF, non-PDF files) and two questions on the scanned page: `make eval-robustness`. It runs apart so
   that its failed files don't mark the scored answers' coverage as incomplete.

Read the numbers with these caveats:

- **The judge is provisional.** `gemini-3.5-flash-lite` with prompt `j2` has not yet been measured against reference
  labels reviewed by a person. Its verdicts decide the content check, so **correct** and **unsafe** inherit its errors.
- **30 holdout reference answers are drafts**, written from the cited page while building the eval and not yet reviewed:
  `x-01` to `x-23` and `c-01` to `c-07` in `eval/questions_holdout.jsonl`. The 18 translations (`pub-0N-pt|es|de`)
  inherit the reference of the public question.
- **The design set is not independent**: its 18 questions were used to write the prompt and the verifier. The holdout
  set was not used for that, but its questions were used to choose the search (retrieval measurements).
- One run per question (`--repeat 1`): model outputs vary between runs, so a single failure or success is one sample.

## Unfixed failures

The scored run below has 11 misses out of 73, reported as measured: no reference was changed after seeing the results.
All 11 fail at `generation`: for every question with an expected page, that page was among the 8 passages sent to the
model, so none is a retrieval failure. By cause:

1. **pub-04, 5 of the 11 (the public question, its design-set copy and three translations): the answer stops at the
   product line.** The question asks for the warranty of WEG-CESTARI gear units. The warranty section (pp. 24, 50, 76)
   lists products, 12 months from the invoice, and services, 6 months; the public reference expects both. In every
   language the model gives the product line, cites it, and leaves services out: incomplete, not wrong. Rule 4 of the
   prompt already asks for every value the passages give for what was asked, but the model reads services as outside
   the question. A stronger rule ("every category the cited passage lists for the subject") has to be measured on the
   whole eval, since it can lengthen answers and add misses elsewhere, and it did not fit in the time box. Next: that
   rule, with its acceptance check fixed beforehand.
2. **x-09 and c-01 (partial): the value is right, a qualifier on the page is missing.** x-09 answers "45° maximum"
   from the figure label, while the caution text on the same page says "45° from vertical"; the judge also asks for
   "no spreader bar", a condition the question itself gives. c-01 identifies the HN10-WD plug as the hexagonal
   oil-level plug, but says neither that the oil level must reach it nor that the gear unit must be stopped in its
   working position to check it. Same family as pub-04, and the same next step.
3. **x-11 (counted unsafe): a judge false positive.** The page says "frequency not more than +5% and voltage not more
   than +10% above or below nameplate rating". The answer repeats it, and the reference writes ±5%, the usual reading
   of that sentence; the judge marked the answer contradicted. It is the only unsafe verdict in the run, and one
   reason the judge is still provisional (caveats above).
4. **c-04 (over-cautious): a case between two prompt rules.** The question asks for a sound power level and gives no
   frequency. The table on p. 47 is for 60 Hz, and its note 3 lowers 50 Hz values by 3 dB(A) for 4 poles: a default
   and an exception, which rule 5 of the prompt answers as `answered` (87 dB(A), with the 50 Hz note). The model read
   the frequency as an open condition and answered `ambiguous` with both cases: safe, but not what the rule asks.
5. **pub-02-de and x-19 (over-cautious): refusals on the safe side.** pub-02-de is the German translation of the
   public relubrication question: the expected page (LB5001 p. 2) ranked first, and the model answered that the
   passages give only grease amounts. The same question passes in English, Portuguese and Spanish, and this is one
   sample (`--repeat 1`). x-19 asks the minimum start temperature with splash lubrication; the table says "immersion
   lubrication" (+7 °C for ISO VG 320 mineral oil), the oil bath that splash lubrication uses in a gear unit, and the
   model declined because the word is not on the page. Not fixed because loosening "only what the passages say" to
   accept equivalent terms would trade these refusals for wrong answers elsewhere. Next: repeated runs, to tell noise
   from a pattern.

## Scored run: public, design and holdout sets

| | |
|---|---|
| Run | 2026-10-30 15:48 UTC to 15:56 UTC; 73 questions × 1 repeat(s), concurrency 1 |
| API | `http://api:8000`, health `ok` |
| Answer model | `gemini-3.8-flash` (from the answer records) |
| Judge | `gemini-3.5-flash-lite`, prompt `j2`, temperature 0 (**provisional** until the meta-evaluation against labels reviewed by a person) |
| PDF folder | 4 files: 4 accepted |
| Question sets | `public_questions` (7), `questions_design` (18), `questions_holdout` (48) |
| Tenant | `19193ef3-ce3c-430c-aa58-5fb297309dcb` (fresh for this run) |

### Headline

| Set | Answers | Correct | Unsafe | Over-cautious | HTTP errors |
|---|---|---|---|---|---|
| public_questions | 7 | 6 (86%) | 0 (0%) | 0 (0%) | 0 |
| questions_design | 18 | 17 (94%) | 0 (0%) | 0 (0%) | 0 |
| questions_holdout | 48 | 39 (81%) | 1 (2%) | 3 (6%) | 0 |
| **all** | 73 | 62 (85%) | 1 (1%) | 3 (4%) | 0 |

- **Correct**: the expected status, the content right according to the judge, and a citation on an expected page. For `not_found`, the status alone.
- **Unsafe**: an answer (or options) when the documents don't answer, one case picked when the question is ambiguous, or a fact of the reference contradicted (a wrong value or condition).
- **Over-cautious**, reported apart: `not_found` (the verifier's decline included) when an answer was expected, or a clarification when a direct answer was expected.

### Status matrix

| expected ↓ / got → | answered | ambiguous | not_found | error |
|---|---|---|---|---|
| answered | 54 | 1 | 1 | 0 |
| ambiguous | 0 | 9 | 1 | 0 |
| not_found | 0 | 0 | 7 | 0 |

`not_found` includes 0 verifier decline(s) (`reason: unverified`).

### By language

| lang | Answers | Correct | Unsafe | Over-cautious |
|---|---|---|---|---|
| (none) | 7 | 6 (86%) | 0 (0%) | 0 (0%) |
| de | 12 | 10 (83%) | 0 (0%) | 1 (8%) |
| en | 22 | 19 (86%) | 0 (0%) | 2 (9%) |
| es | 16 | 14 (88%) | 1 (6%) | 0 (0%) |
| pt | 16 | 13 (81%) | 0 (0%) | 0 (0%) |

### Failures by stage

| Stage | Count | Questions |
|---|---|---|
| generation | 11 | public_questions/pub-04, questions_design/pub-04, questions_holdout/pub-02-de, questions_holdout/pub-04-pt, questions_holdout/pub-04-es, questions_holdout/pub-04-de, questions_holdout/x-09, questions_holdout/x-11, questions_holdout/x-19, questions_holdout/c-01, questions_holdout/c-04 |

First failing stage in pipeline order: `ingestion` (no expected page indexed), `retrieval` (no expected page among the 30 candidates), `ranking` (among the 30, not in the 8 sent to the model), `provider` (an `llm_*` error), `api` (another HTTP error), `verification` (the verifier declined), `generation` (the model's status, content or citation).

### Questions

#### public_questions

| id | lang | expected | got | content | source | rank | stage | latency ms |
|---|---|---|---|---|---|---|---|---|
| pub-01 | - | answered | answered | correct | yes | 1 |  | 18,155 |
| pub-02 | - | ambiguous | ambiguous | correct | yes | 1 |  | 6,126 |
| pub-03 | - | answered | answered | correct | yes | 1 |  | 3,888 |
| pub-04 ✗ | - | answered | answered | partial | yes | 1 | generation | 2,721 |
| pub-05 | - | not_found | not_found | - | - | - |  | 8,368 |
| pub-06 | - | answered | answered | correct | yes | 2 |  | 2,761 |
| pub-07 | - | answered | answered | correct | yes | 1 |  | 2,727 |

Not correct:

- **pub-04** (generation): expected answered, got answered. Answer: «O prazo padrão de garantia dos produtos WEG-CESTARI é de 12 meses a contar da data de emissão da Nota Fiscal. Caso um prazo de garantia diferenciado esteja definido na proposta técnico-comercial para determinado forneci…» Judge: missing: serviços: 6 meses. Expected: «Produtos: 12 meses a partir da emissão da Nota Fiscal; serviços: 6 meses.»

#### questions_design

| id | lang | expected | got | content | source | rank | stage | latency ms |
|---|---|---|---|---|---|---|---|---|
| pub-01 | en | answered | answered | correct | yes | 1 |  | 4,364 |
| pub-02 | en | ambiguous | ambiguous | correct | yes | 1 |  | 5,435 |
| pub-03 | en | answered | answered | correct | yes | 1 |  | 2,435 |
| pub-04 ✗ | pt | answered | answered | partial | yes | 1 | generation | 3,584 |
| pub-05 | en | not_found | not_found | - | - | - |  | 2,191 |
| pub-06 | en | answered | answered | correct | yes | 2 |  | 4,386 |
| pub-07 | en | answered | answered | correct | yes | 1 |  | 3,431 |
| e-01 | en | answered | answered | correct | yes | 1 |  | 2,717 |
| a-01 | en | ambiguous | ambiguous | correct | yes | 1 |  | 9,082 |
| a-02 | en | ambiguous | ambiguous | correct | yes | 1 |  | 3,895 |
| a-03 | pt | ambiguous | ambiguous | correct | yes | 1 |  | 7,599 |
| a-04 | pt | ambiguous | ambiguous | correct | yes | 2 |  | 3,929 |
| a-05 | es | ambiguous | ambiguous | correct | yes | 3 |  | 3,030 |
| n-01 | en | not_found | not_found | - | - | - |  | 6,222 |
| n-02 | pt | not_found | not_found | - | - | - |  | 2,728 |
| n-03 | en | not_found | not_found | - | - | - |  | 2,412 |
| n-04 | pt | not_found | not_found | - | - | - |  | 2,718 |
| n-05 | es | not_found | not_found | - | - | - |  | 3,453 |

Not correct:

- **pub-04** (generation): expected answered, got answered. Answer: «O prazo padrão de garantia para produtos da WEG-CESTARI contra defeitos de fabricação e de materiais é de 12 meses a contar da data de emissão da Nota Fiscal. Caso um prazo de garantia diferenciado esteja definido na pr…» Judge: missing: Services warranty is 6 months. Expected: «Produtos: 12 meses a partir da emissão da Nota Fiscal; serviços: 6 meses.»

#### questions_holdout

| id | lang | expected | got | content | source | rank | stage | latency ms |
|---|---|---|---|---|---|---|---|---|
| pub-01-pt | pt | answered | answered | correct | yes | 1 |  | 3,586 |
| pub-01-es | es | answered | answered | correct | yes | 1 |  | 2,391 |
| pub-01-de | de | answered | answered | correct | yes | 1 |  | 2,856 |
| pub-02-pt | pt | ambiguous | ambiguous | correct | yes | 2 |  | 9,065 |
| pub-02-es | es | ambiguous | ambiguous | correct | yes | 1 |  | 6,365 |
| pub-02-de ✗ | de | ambiguous | not_found | - | - | 1 | generation | 3,505 |
| pub-03-pt | pt | answered | answered | correct | yes | 1 |  | 5,775 |
| pub-03-es | es | answered | answered | correct | yes | 1 |  | 2,499 |
| pub-03-de | de | answered | answered | correct | yes | 1 |  | 2,889 |
| pub-04-pt ✗ | pt | answered | answered | partial | yes | 1 | generation | 2,974 |
| pub-04-es ✗ | es | answered | answered | partial | yes | 1 | generation | 2,802 |
| pub-04-de ✗ | de | answered | answered | partial | yes | 1 | generation | 3,711 |
| pub-06-pt | pt | answered | answered | correct | yes | 1 |  | 3,321 |
| pub-06-es | es | answered | answered | correct | yes | 1 |  | 3,629 |
| pub-06-de | de | answered | answered | correct | yes | 1 |  | 3,948 |
| pub-07-pt | pt | answered | answered | correct | yes | 1 |  | 2,359 |
| pub-07-es | es | answered | answered | correct | yes | 1 |  | 4,804 |
| pub-07-de | de | answered | answered | correct | yes | 1 |  | 3,266 |
| x-01 | pt | answered | answered | correct | yes | 1 |  | 5,400 |
| x-02 | es | answered | answered | correct | yes | 1 |  | 2,772 |
| x-03 | de | answered | answered | correct | yes | 1 |  | 19,726 |
| x-04 | pt | answered | answered | correct | yes | 1 |  | 2,190 |
| x-05 | en | answered | answered | correct | yes | 1 |  | 2,031 |
| x-06 | pt | answered | answered | correct | yes | 1 |  | 2,766 |
| x-07 | es | answered | answered | correct | yes | 3 |  | 3,758 |
| x-08 | de | answered | answered | correct | yes | 1 |  | 2,775 |
| x-09 ✗ | pt | answered | answered | partial | yes | 4 | generation | 2,383 |
| x-10 | en | answered | answered | correct | yes | 1 |  | 2,366 |
| x-11 ⚠ | es | answered | answered | wrong | yes | 2 | generation | 2,496 |
| x-12 | en | answered | answered | correct | yes | 3 |  | 2,376 |
| x-13 | es | answered | answered | correct | yes | 2 |  | 2,693 |
| x-14 | en | answered | answered | correct | yes | 3 |  | 2,651 |
| x-15 | de | answered | answered | correct | yes | 1 |  | 2,464 |
| x-16 | es | answered | answered | correct | yes | 2 |  | 2,312 |
| x-17 | en | answered | answered | correct | yes | 2 |  | 3,363 |
| x-18 | de | answered | answered | correct | yes | 1 |  | 2,275 |
| x-19 ✗ | en | answered | not_found | - | - | 1 | generation | 8,409 |
| x-20 | es | answered | answered | correct | yes | 1 |  | 2,464 |
| x-21 | en | answered | answered | correct | yes | 1 |  | 2,470 |
| x-22 | de | answered | answered | correct | yes | 1 |  | 2,180 |
| x-23 | en | answered | answered | correct | yes | 1 |  | 2,380 |
| c-01 ✗ | en | answered | answered | partial | yes | 1 | generation | 2,292 |
| c-02 | pt | answered | answered | correct | yes | 1 |  | 2,925 |
| c-03 | es | answered | answered | correct | yes | 1 |  | 2,971 |
| c-04 ✗ | en | answered | ambiguous | - | yes | 1 | generation | 2,846 |
| c-05 | de | answered | answered | correct | yes | 1 |  | 2,417 |
| c-06 | es | answered | answered | correct | yes | 1 |  | 5,301 |
| c-07 | en | answered | answered | correct | yes | 1 |  | 7,679 |

Not correct:

- **pub-02-de** (generation, over-cautious): expected ambiguous, got not_found. Answer: «Die bereitgestellten Passagen enthalten keine Informationen darüber, wie oft ein Baldor-Motor nachgeschmiert werden sollte (sie nennen lediglich die Nachschmiermengen nach Baugröße, die Vorgehensweise sowie Wechselinter…» Expected: «Depends on frame size and rated speed (e.g. 5,500 h for frame up to 210 at 3600 RPM up to 22,000 h at 900 RPM). Should ask for / enumerate frame size and RPM.»
- **pub-04-pt** (generation): expected answered, got answered. Answer: «O prazo padrão de garantia para produtos oferecido pela WEG-CESTARI é de 12 meses a contar da data de emissão da Nota Fiscal. Caso um prazo diferenciado esteja definido na proposta técnico-comercial para determinado for…» Judge: missing: serviços: 6 meses. Expected: «Produtos: 12 meses a partir da emissão da Nota Fiscal; serviços: 6 meses.»
- **pub-04-es** (generation): expected answered, got answered. Answer: «Para los productos de WEG-CESTARI, el plazo estándar de garantía es de 12 meses, contando desde la fecha de emisión de la Factura. Sin embargo, en caso de que esté definido un plazo de garantía diferenciado en la propue…» Judge: missing: serviços: 6 meses. Expected: «Produtos: 12 meses a partir da emissão da Nota Fiscal; serviços: 6 meses.»
- **pub-04-de** (generation): expected answered, got answered. Answer: «Für Produkte von WEG-CESTARI gilt eine Standard-Garantiezeit von 12 Monaten ab dem Ausstellungsdatum der Rechnung (unabhängig vom Installationsdatum und von der Inbetriebnahme). Falls im technisch-kommerziellen Angebot …» Judge: missing: serviços: 6 meses. Expected: «Produtos: 12 meses a partir da emissão da Nota Fiscal; serviços: 6 meses.»
- **x-09** (generation): expected answered, got answered. Answer: «O ângulo máximo do cabo de içamento é de 45° (45° Maximum).» Judge: missing: Angle is measured from the vertical; missing: Applies when not using a spreader bar. Expected: «45° from vertical at most (angle of the lift rope without a spreader bar).»
- **x-11** (generation, unsafe): expected answered, got answered. Answer: «El motor de bomba sumergible funcionará adecuadamente con una variación de frecuencia de no más de +5% y una variación de tensión de no más de +10% por encima o por debajo del valor nominal de la placa (nameplate rating…» Judge: contradicted: Frequency variation within +-5%. Expected: «Frequency within ±5% and voltage within ±10% above or below the nameplate rating.»
- **x-19** (generation, over-cautious): expected answered, got not_found. Answer: «The passages do not mention splash lubrication; they only provide minimum startup temperatures for immersion lubrication and forced lubrication.» Expected: «+7 °C (mineral CLP oil, ISO VG 320, immersion/splash lubrication).»
- **c-01** (generation): expected answered, got answered. Answer: «On a WEG-CESTARI gear unit or gearmotor, the HN10-WD plug is a hexagonal oil level plug indicator (DIN 510 similar) that is different from the other plugs present on the unit.» Judge: missing: The correct oil level is at the level of the plug; missing: The gear unit must be stopped in its working position. Expected: «A hexagonal oil level plug (model HN10-WD, similar to DIN 510), different from the other plugs of the gear unit; the correct oil level is at the level of the p…»
- **c-04** (generation, over-cautious): expected answered, got ambiguous. Answer: «The maximum sound power level depends on the supply frequency (60 Hz or 50 Hz).» Expected: «87 dB(A) sound power (4 poles, frame 225, IC411, no load, 60 Hz).»

`rank` is the best fused rank of an expected page among the 30 search candidates; ✗ not correct, ⚠ unsafe.

### Citations

- Citations returned: 69 in 64 answers; on an expected page: 67.
- Text-layer pages: 44 citations; quote found in the page's text layer read by the eval with PyMuPDF (markup and whitespace ignored, table rows cell by cell): 41 (93%) of the citations, 169/172 lines.
- Transcribed pages (`transcribed: true`): 25 citations, counted apart: their text comes from the page image, so the text layer can't confirm it.
- Quotes not fully found: a-03 LB5001.pdf p2 (5/6 lines); x-09 MN414_0224.pdf p7 (0/1 lines); x-15 WEG-motores-eletricos-guia-de-especificacao-50032749-brochure-portuguese-web.pdf p50 (1/2 lines).
- Grounded numbers, re-checked from the API's answer alone (every number of the answer must be in a quote of the answer or in the question, any locale reading): 64 (100%) of the answers pass.

### Retrieval

- Answers with expected pages and a record: 66. Expected page among the 8 passages sent to the model: 66 (100%); among the 30 candidates: 66 (100%); MRR 0.89 (fused rank of the first chunk on an expected page).

### Judge

- Model `gemini-3.5-flash-lite` (**provisional**), prompt `j2`, temperature 0, reference-guided: the model lists the facts of the reference (essential or example) and marks each `stated`, `missing` or `contradicted`; the verdict is computed by code (correct: every essential fact stated and none contradicted; partial: nothing contradicted, an essential fact missing; wrong: something contradicted or no essential fact stated).
- Judged answers: 63 (only when the status matches and is not `not_found`): correct 55, partial 7, wrong 1; from the cache: 1.
- Numeric cross-check (numbers of the reference that the question doesn't hold, looked up in the answer): verdicts to review: questions_holdout/x-09: judged partial, but every number of the reference appears in the answer; questions_holdout/x-11: judged contradicted, but every number of the reference appears in the answer; questions_holdout/c-01: judged partial, but every number of the reference appears in the answer.

### Latency

- Pipeline (from the answer record, scope check to response): p50 2,925 ms, p95 8,671, max 19,726 ms.
- HTTP round trip measured by the harness: p50 2,931 ms, p95 8,677, max 19,731 ms.
- Samples: 73.
- Answer model calls: 74; verifier retries: 1 (c-06); provider retries: 0.

### Cost

Measured tokens (OCR from `GET /v1/documents/{id}`, answers from the answer records, judge from its calls) times the list prices of 2026-10-30.

| Part | Model | Calls | Input tokens | Output tokens | US$ |
|---|---|---|---|---|---|
| Answers | `gemini-3.8-flash` | 74 | 231,976 | 10,645 | 0.2353 |
| **Product total** | | | | | **0.2353** |
| Judge | `gemini-3.5-flash-lite` | 62 | 34,918 | 11,484 | 0.0392 |
| **Judge total** | | | | | **0.0392** |

- Answers: US$ 0.00322 per question on average (73 answers).
- OCR: 54 transcribed pages, 54 of them from the shared OCR cache (an earlier transcription of the same file: no call, no cost in this run).
- Embeddings run locally (BGE-M3 int8): no per-token cost.

Prices (US$ per million tokens, input / output, thinking billed as output), Gemini API pricing page, paid tier (standard, not batch), as of 2026-10-30: `gemini-3.5-flash-lite` 0.3 / 2.5; `gemini-3.8-flash` 0.825 / 4.125; `gemini-embedding-2` 0.2 / 0.

### Ingestion

One batch, 4 files, complete after 102 s.

| File | Upload | Status | Pages | Indexed | Layer / OCR | OCR: flagged / transcribed / cache | Failed pages | Error | Final after s |
|---|---|---|---|---|---|---|---|---|---|
| LB5001.pdf | accepted (202) | ready | 2 | 2 | 2 / 0 | 0 / 0 / 0 | - | - | 4 |
| MN414_0224.pdf | accepted (202) | ready | 16 | 15 | 15 / 0 | 0 / 0 / 0 | - | - | 22 |
| WEG-CESTARI-manual-iom-guia-consulta-rapida-50111652-pt-en-… | accepted (202) | ready | 84 | 81 | 28 / 53 | 53 / 53 / 53 | - | - | 52 |
| WEG-motores-eletricos-guia-de-especificacao-50032749-brochu… | accepted (202) | ready | 68 | 68 | 67 / 1 | 1 / 1 / 1 | - | - | 102 |

Coverage: no question was asked while a document was still queued or processing (failed documents don't count: they are listed above).

## Robustness run: broken and scanned files, own tenant

| | |
|---|---|
| Run | 2026-10-30 15:56 UTC to 15:57 UTC; 2 questions × 1 repeat(s), concurrency 1 |
| API | `http://api:8000`, health `ok` |
| Answer model | `gemini-3.8-flash` (from the answer records) |
| Judge | `gemini-3.5-flash-lite`, prompt `j2`, temperature 0 (**provisional** until the meta-evaluation against labels reviewed by a person) |
| PDF folder | 7 files: 4 accepted, 3 rejected |
| Question sets | `questions_robustness` (2) |
| Tenant | `27d2eb35-e0dd-4db5-ab59-f605a9f9c7fe` (fresh for this run) |

### Headline

| Set | Answers | Correct | Unsafe | Over-cautious | HTTP errors |
|---|---|---|---|---|---|
| questions_robustness | 2 | 2 (100%) | 0 (0%) | 0 (0%) | 0 |

- **Correct**: the expected status, the content right according to the judge, and a citation on an expected page. For `not_found`, the status alone.
- **Unsafe**: an answer (or options) when the documents don't answer, one case picked when the question is ambiguous, or a fact of the reference contradicted (a wrong value or condition).
- **Over-cautious**, reported apart: `not_found` (the verifier's decline included) when an answer was expected, or a clarification when a direct answer was expected.

### Status matrix

| expected ↓ / got → | answered | ambiguous | not_found | error |
|---|---|---|---|---|
| answered | 2 | 0 | 0 | 0 |

`not_found` includes 0 verifier decline(s) (`reason: unverified`).

### By language

| lang | Answers | Correct | Unsafe | Over-cautious |
|---|---|---|---|---|
| en | 1 | 1 (100%) | 0 (0%) | 0 (0%) |
| pt | 1 | 1 (100%) | 0 (0%) | 0 (0%) |

### Failures by stage

No failures.

### Questions

#### questions_robustness

| id | lang | expected | got | content | source | rank | stage | latency ms |
|---|---|---|---|---|---|---|---|---|
| scan-01 | en | answered | answered | correct | yes | 1 |  | 2,845 |
| scan-02 | pt | answered | answered | correct | yes | 1 |  | 3,643 |

`rank` is the best fused rank of an expected page among the 30 search candidates; ✗ not correct, ⚠ unsafe.

### Citations

- Citations returned: 2 in 2 answers; on an expected page: 2.
- Text-layer pages: 0 citations; quote found in the page's text layer read by the eval with PyMuPDF (markup and whitespace ignored, table rows cell by cell): 0 of the citations, 0/0 lines.
- Transcribed pages (`transcribed: true`): 2 citations, counted apart: their text comes from the page image, so the text layer can't confirm it.
- Grounded numbers, re-checked from the API's answer alone (every number of the answer must be in a quote of the answer or in the question, any locale reading): 2 (100%) of the answers pass.

### Retrieval

- Answers with expected pages and a record: 2. Expected page among the 8 passages sent to the model: 2 (100%); among the 30 candidates: 2 (100%); MRR 1.00 (fused rank of the first chunk on an expected page).

### Judge

- Model `gemini-3.5-flash-lite` (**provisional**), prompt `j2`, temperature 0, reference-guided: the model lists the facts of the reference (essential or example) and marks each `stated`, `missing` or `contradicted`; the verdict is computed by code (correct: every essential fact stated and none contradicted; partial: nothing contradicted, an essential fact missing; wrong: something contradicted or no essential fact stated).
- Judged answers: 2 (only when the status matches and is not `not_found`): correct 2; from the cache: 0.
- Numeric cross-check (numbers of the reference that the question doesn't hold, looked up in the answer): no disagreement.

### Latency

- Pipeline (from the answer record, scope check to response): p50 3,244 ms, max 3,643 ms.
- HTTP round trip measured by the harness: p50 3,249 ms, max 3,648 ms.
- Samples: 2 (p95 shown from 20 samples on).
- Answer model calls: 2; verifier retries: 0; provider retries: 0.

### Cost

Measured tokens (OCR from `GET /v1/documents/{id}`, answers from the answer records, judge from its calls) times the list prices of 2026-10-30.

| Part | Model | Calls | Input tokens | Output tokens | US$ |
|---|---|---|---|---|---|
| Answers | `gemini-3.8-flash` | 2 | 3,546 | 249 | 0.0040 |
| **Product total** | | | | | **0.0040** |
| Judge | `gemini-3.5-flash-lite` | 2 | 1,132 | 432 | 0.0014 |
| **Judge total** | | | | | **0.0014** |

- Answers: US$ 0.00198 per question on average (2 answers).
- OCR: 1 transcribed pages, 1 of them from the shared OCR cache (an earlier transcription of the same file: no call, no cost in this run).
- Embeddings run locally (BGE-M3 int8): no per-token cost.

Prices (US$ per million tokens, input / output, thinking billed as output), Gemini API pricing page, paid tier (standard, not batch), as of 2026-10-30: `gemini-3.5-flash-lite` 0.3 / 2.5; `gemini-3.8-flash` 0.825 / 4.125; `gemini-embedding-2` 0.2 / 0.

### Ingestion

One batch, 7 files, complete after 4 s.

| File | Upload | Status | Pages | Indexed | Layer / OCR | OCR: flagged / transcribed / cache | Failed pages | Error | Final after s |
|---|---|---|---|---|---|---|---|---|---|
| blank.pdf | accepted (202) | failed | 1 | 0 | 0 / 0 | 0 / 0 / 0 | - | no_text | 2 |
| empty.pdf | rejected (422) | - | - | - | 0 / 0 | 0 / 0 / 0 | - | empty_file | - |
| lb5001-p2-scan.pdf | accepted (202) | ready | 1 | 1 | 0 / 1 | 1 / 1 / 1 | - | - | 4 |
| locked.pdf | accepted (202) | failed | - | 0 | 0 / 0 | 0 / 0 / 0 | - | pdf_encrypted | 4 |
| not-a-pdf.pdf | rejected (422) | - | - | - | 0 / 0 | 0 / 0 / 0 | - | not_a_pdf | - |
| notes.txt | rejected (422) | - | - | - | 0 / 0 | 0 / 0 / 0 | - | not_a_pdf | - |
| truncated.pdf | accepted (202) | failed | - | 0 | 0 / 0 | 0 / 0 / 0 | - | pdf_corrupt | 4 |

Coverage: no question was asked while a document was still queued or processing (failed documents don't count: they are listed above).
