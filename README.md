# Resume Tailor

A command-line tool that takes a job posting link and produces a one-page,
ATS-safe resume tailored to that job, built **only from verified facts** in
`master_resume.yaml`. It never invents or inflates experience: truthfulness is
enforced in code (`src/validate.py`) and by a human approval checkpoint, not
only by prompts.

> **Status: under construction.** Built milestone by milestone; see
> "Build progress" below. This README is completed in the final milestone.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate    # Python 3.11+
pip install -r requirements.txt
brew install --cask libreoffice                        # PDF conversion (macOS)
cp .env.example .env                                   # then put your key in .env
python tailor.py --check                               # verifies key, models, data files
```

The API key is read from `.env` only. `.env` and `output/` are gitignored.

## Build progress

| # | Milestone | Status |
|---|---|---|
| 1 | Scaffold, config, schemas, startup checks (incl. model verification) | done |
| 2 | Job posting extraction + paste fallback | done |
| 3 | Job analysis + eligibility checking | done |
| 4 | Master resume fact bank, `skills_status.yaml`, evidence-based validation | done |
| 5 | Resume tailoring (normal + `--strict`) | done |
| 6 | DOCX/PDF rendering, page-fit optimization, ATS checks | done |
| 7 | Independent judge + code-computed scoring | done |
| 8 | Iterative revision loop | done |
| 9 | Company research with source tiers | done |
| 10 | Interview system + human approval checkpoint | done |
| 11 | Reports, archive, application tracking, cost monitoring | done |
| 12 | Application intelligence dashboard | |
| 13 | Batch processing + optional cover letter | |

## Tests

```bash
pytest
```

All Claude calls are mocked; tests never use the network or an API key.
