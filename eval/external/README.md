# External eval set

Ten public industrial manuals from outside the four sample manuals, with 15 questions about them. The detector, the prompt, the
verifier and the search were tuned on the four sample PDFs in `case_files/`; real use brings other documents of the same kind. This
set is the check of how the pipeline does on documents it was not tuned on: scans without a text layer, a 454-page
catalog, manuals in German, Spanish, French and Italian, and one file in seven languages. One search setting (the share
of the corpus under which a term counts as rare) was later chosen between two values with this set, so it is no longer
fully held out. The reference answer of o11-12 was also trimmed after the first run: it asked for the 8-hour level
check, which the question does not ask about, and the judge marked a complete answer partial.

It runs apart from `make eval`, which only ever uses `case_files/` and `eval/public_questions.jsonl`.

| File | What it is |
|---|---|
| `manifest.json` | Each manual: file name, direct URL, source page, full sha256, size, pages, languages, license, what it tests |
| `fetch.py` | Downloads each manual into `pdfs/`, checks its sha256, cuts the TM 5-5060 excerpt |
| `questions_external.jsonl` | The 15 questions, in the format of `eval/public_questions.jsonl` plus `lang` (language of the question) |
| `pdfs/`, `originals/` | Downloaded files, git-ignored. `pdfs/` is the folder the eval uploads; `originals/` keeps the whole TM 5-5060 |

## Run

```sh
make eval-external-fetch          # download and check only (about 110 MB, no model calls)
make eval-external                # download if needed, then the eval on pdfs/ with questions_external.jsonl; report in eval/out/
make eval-external ARGS="--no-judge"
python -I eval/external/fetch.py  # the download step without Docker (needs PyMuPDF)
```

A file already in `pdfs/` with the recorded sha256 is not downloaded again. A failed download or a file whose sha256
differs from the manifest (the publisher replaced it) is listed with the reason and left out of `pdfs/`; the others go
on, and the questions about the missing file show `file missing from folder` in the report. `fetch.py` exits 1 only when
no file is available.

**Cost**, measured on 30 Oct 2026 with the default models: the first run transcribes 161 pages (the three scans, the
photo and drawing pages of NORD and SKF, one Bonfiglioli table), about US$0.25 of OCR, plus about US$0.05 for the 15
answers and US$0.01 for the judge; plan for US$0.40. Ingestion takes about 15 minutes (extraction and indexing run one
document at a time). Each run creates a new tenant: transcribed pages come from the OCR cache, but pages the OCR refused
are tried again (about US$0.08 per run).

**TM 5-5060 excerpt.** Only pages 1–60 of the 229-page TM 5-5060 go to `pdfs/`, with the original page numbers (rule and
hashes in `manifest.json`): OCR of the whole scan was projected above the budget, and these pages hold the
specifications, the operator instructions and the lubrication section. The OCR model's recitation filter refuses its
prose pages (36 of the 60 on 30 Oct 2026), so the document ends `partial`; question o11-11 is about one of those pages.

## Why the PDFs are not in the repository

Six of the manuals are the manufacturers' copyrighted documents (SKF, NORD twice, Leroy-Somer, Bonfiglioli, ABB),
offered as public downloads but not for redistribution. `fetch.py` downloads every manual from its original source and
checks it against the sha256 recorded when the set was built, so a run uses the same bytes or says which file changed.
The others (the two US Army manuals and the DTIC report, public domain or approved for public release, and the
1943 Ingersoll-Rand scan, marked public domain by its uploader) go the same way.

## Questions

Pages are 1-based positions in the PDF, as the eval counts them; the printed page number follows in parentheses when it
differs. For the scanned manuals the values were checked on the page image.

| id | lang | Manual | Expected | Page and text the reference answer rests on | What it tests |
|---|---|---|---|---|---|
| o11-01 | pt | NORD B 1000, DE and ES | answered | DE p. 49, Tabelle 10: "Bei Betriebstemperaturen bis 80 °C alle 10000 Betriebsstunden, mindestens alle 2 Jahre • Öl wechseln"; ES p. 48, Tabla 10: "Cada 10.000 horas de servicio al menos cada 2 años • Cambiar el aceite" | One fact in two documents and two languages (both are sources); a table cell split over two rows |
| o11-02 | en | NORD B 1000, DE and ES | answered | DE p. 76, Tabelle 14 "Schrauben-Anziehdrehmomente [Nm]": "M16 \| 230 \| 335 \| 390 \| 207" (classes 8.8, 10.9, 12.9, V2A-70); ES p. 75, Tabla 14 | A German torque table, almost only numbers, asked in English |
| o11-03 | pt | ABB | answered | p. 10, Table 4, ball bearings: "200 \| 40 \| 4000 \| 6000 \| 11000 \| 13000 \| 17000 \| 21000" (frame, grams per bearing, hours at 3600 to 500–900 r/min); the same table on pp. 22, 34, 46, 58, 70, 82 | Interval and grease amount from a table repeated in seven languages |
| o11-04 | en | Bonfiglioli HDP/HDO | answered | p. 80 (78), tab. 21: "65 < to < 80 \| 4000 \| 15000 \| 15000" (mineral EP, PAO, PAG oil); "(*) = Sostituzione comunque entro 1 anno" | An Italian oil-change table by temperature, with its footnote |
| o11-05 | pt | Leroy-Somer 4850 | ambiguous | p. 6, "Roulements regraissables": grade 2 grease, "supérieure à 1 an inférieure à 5 ans" → "Démonter le roulement - Le nettoyer - Renouveler la graisse en totalité"; grade 3, "supérieure à 1 an inférieure à 2 ans" → "Procéder à un regraissage avant la mise en service" | The answer depends on the grease grade, which the question leaves out |
| o11-06 | en | SKF handbook | answered | p. 407 (403), Appendix F-2: "100 \| 120 \| 0,050 \| 0,060 \| 0,79 \| 0,95 \| 2,00 \| 2,35" (bore over/incl., clearance reduction min/max, axial drive-up for taper 1:12 min/max, 1:30) | A long table split into chunks under a title row; decimal commas in English |
| o11-07 | en | TM 5-692-1 | answered | p. 88 (11-6), Table 11-2, chilled water system pumps: "Inspect shaft alignment and clearances of impeller and shaft. Readjust as required. 3 mos"; "Replace mechanical pump seal. yr" | One table among dozens of similar ones in a 261-page manual |
| o11-08 | pt | DTIC blower (scan) | answered | p. 15, motor chart: "Motor is equipped with alemite fitting… Use 1 to 2 strokes of high quality ball bearing grease"; "Medium \| Lithium \| Shell Dolium R"; "Continual Normal Application \| 1 year" | A table read by OCR |
| o11-09 | en | DTIC blower (scan) | answered | p. 10 (4-1): "The filter should be checked weekly for the first 2 months of operation… When the vacuum drop across the filter is above 15 inches of water, a dirty filter element should be suspected"; p. 11 (4-2): "Filter \| Check once per month, wash or replace as necessary" | An answer spread over two scanned pages |
| o11-10 | pt | Ingersoll-Rand (scan) | answered | p. 8 (7): "Oil is ordinarily fed at the rate of three to four drops per minute. However, for the first two weeks of operation, it should be fed at a rate of fifteen drops per minute." | An old low-resolution scan; numbers spelled out in words |
| o11-11 | en | TM 5-5060 excerpt (scan) | answered | p. 49 (31), "Table of capacities and lubricants to be used": "+32°F. to 0°F. \| OE SAE 10" (engine and compressor crankcases) | A page the OCR refuses (recitation): the expected failure is at ingestion. The reference keeps what the manual says, so the miss is counted: a `not_found` here is over-cautious, not unsafe |
| o11-12 | pt | TM 5-5060 excerpt (scan) | answered | p. 48 (30), War Department Lubrication Guide: "128 OE Engine crankcase (See Table) Drain, refill (Note 3) Check level every 8 hours"; "Compressor crankcase OE 256 (See Table) Drain, refill (Note 4) Check level every 8 hours" | A scanned lubrication chart: a drawing with labels |
| o11-13 | pt | TM 5-692-1 | not_found | p. 143 (18-4), Table 18-3: "Perform megohm resistance test between pump body and power leads. Resistance should be in the megohm. yr" | Near miss: the test is named, its minimum value is not |
| o11-14 | en | NORD B 1000, DE and ES | not_found | DE pp. 74–75: "Die erforderliche Schmierstoffmenge und Schmierstoffart sind auf dem Typenschild angegeben"; ES p. 74: "Encontrará la cantidad y el tipo de lubricante necesario en la placa de características"; p. 19 in both | Near miss: the manual sends the reader to the nameplate instead of giving quantities |
| o11-15 | en | TM 5-5060 excerpt (scan) | answered | p. 9 (V), Specifications: "Spark Plug Gap……025" to .030"", "Valve Clearance (Engine Hot)……015"" | A decimal point glued to a dot leader (.015") |
