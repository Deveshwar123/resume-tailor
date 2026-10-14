# Resume Tailor

Give it a job posting link. It gives back:

1. an **eligibility check** against the posting's requirements,
2. a **one-page, ATS-safe resume tailored to the job**, built only from verified facts about you,
3. three separate scores: **Resume Quality** (target 9/10), **Job Match**, and **Keyword Coverage**,
4. reports on what changed, what you approved, the score history, and your skill gaps,
5. an archived snapshot of exactly what was produced, and
6. over time, a **dashboard** showing which skills would unlock the most postings.

**It never exaggerates.** The tool may select, reorder, reword, and emphasize facts from
`master_resume.yaml`. It cannot add or enlarge an employer, title, date, number, skill, tool, scope,
or level of ownership ("assisted" never becomes "led"). That is enforced in code
(`src/validate.py`) and by an approval step, not only by prompts. If a 9/10 is not reachable
honestly, it stops and tells you why.

```
link -> fetch -> analyze -> eligibility -> research -> tailor -> validate
                                                         ^          |
                                              revise <- judge <- score      (loop until 9.0 or no honest fix is left)
     -> interview (optional) -> approval -> render + page fit -> ATS check -> reports + archive
```

## Setup

Needs Python 3.11+ and LibreOffice (for PDF conversion).

```bash
cd resume-tailor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install --cask libreoffice          # macOS; on Linux: apt install libreoffice
cp .env.example .env                     # then put your Anthropic API key in .env
python tailor.py --check                 # verifies the key, both model IDs, and your data files
```

- The API key is read from `.env` **only**. `.env` and `output/` are gitignored.
- `--check` asks the API whether the two configured models exist. If one does not, it prints the
  model IDs your key can use and stops. Change models in `config.py` or with `FAST_MODEL=` /
  `WRITE_MODEL=` lines in `.env`.
- Without LibreOffice, `pip install docx2pdf` works as a fallback if Microsoft Word is installed.

Before your first real run, **review `master_resume.yaml`** (see "Writing facts" below). The header
of that file lists what needs your confirmation.

## Using it

```bash
python tailor.py <job_url>                     # tailor for one posting
python tailor.py --text job.txt                # the posting saved as text (use when a site blocks fetching)
python tailor.py <job_url> --strict            # keep my approved wording; rewrites are only proposals
python tailor.py <job_url> --no-research       # skip the company web research
python tailor.py <job_url> --no-interview      # never ask me follow-up questions
python tailor.py <job_url> --max-rounds 5 --target 9.0
python tailor.py <job_url> --cover-letter      # also write a cover letter from the same facts
python tailor.py --batch links.txt             # many postings, no prompts
python tailor.py --insights                    # rebuild the dashboard (no API key needed)
python tailor.py --find "Company"              # look up what I produced for a company (no API key needed)
python tailor.py --check                       # startup checks only
```

| Flag | What it does |
|---|---|
| `<job_url>` | One request to that link with a normal browser User-Agent; the main text is extracted. It never crawls. Results are cached for 24 hours. |
| `--text FILE` | Read the posting from a file. LinkedIn, Workday, and Handshake usually block fetching; when that happens the tool asks you to paste the description (finish with a blank line), or you can use this flag. |
| `--strict` | Your `approved: true` bullets are used word for word. A rewrite is only proposed alongside its original, and only your approval puts it on the resume. |
| `--no-research` | Skip the web research about the company. |
| `--no-interview` | Do not ask questions when the review thinks you may have unlisted experience. |
| `--max-rounds N` | Maximum revision rounds (default 5). |
| `--target X` | Resume Quality target out of 10 (default 9.0). |
| `--cover-letter` | Also write `cover_letter.md/.docx/.pdf`. Same facts, same validator, same approval rules. |
| `--batch FILE` | One link per line (`#` starts a comment). Nothing prompts: blocked links are skipped, eligibility flags are recorded and the run continues, unapproved rewrites revert to verified wording, and the dashboard is rebuilt at the end. |
| `--insights` | Rebuild `output/insights.html` and `insights.md`. |
| `--find "Company"` | List archived runs whose company or role matches. |
| `--check` | Run the startup checks and exit. |

### What a run produces

`output/<Company>_<Role>_<YYYY-MM-DD>/`

| File | Contents |
|---|---|
| `resume.pdf`, `resume.docx`, `<Name>_Resume.pdf` | The tailored resume (the last is the same PDF under a filename fit to send). |
| `eligibility.md` | Requirement table with the posting's own sentences; flags first. |
| `changes.md` | Every bullet: original vs. final, cited fact ids, change type, approval decision, flags; the validation log; page-fit steps. |
| `scores.md` | Resume Quality per round with category scores, why the loop stopped, Job Match, Keyword Coverage, every ATS check, Claude usage and cost. |
| `gaps.md` | Requirements you do not clearly meet, real gaps, open questions, and why the target was missed if it was. |
| `posting.txt`, `master_resume.snapshot.yaml`, `tailored_resume.json`, `scores.json`, `company_brief.json`, `eligibility.json`, `job_analysis.json`, `usage.json`, `archive.json` | The archive: exactly what went in and what came out. |

Each run also appends a row to `data/applications.csv`. Its `status` starts as `drafted`; change
it by hand to `applied`, `interview`, `offer`, or `rejected` and the dashboard's funnel fills in.

## Writing facts in `master_resume.yaml`

This file is the only source of truth. Store **detailed factual material**, not just polished
bullets: Claude builds bullets from facts and picks different ones for software, data, finance,
and research roles.

```yaml
experience:
  - id: exp_acme                    # unique, never reused
    title: Software Engineering Intern
    org: Acme Corp
    location: Austin, TX
    dates: {start: 2025-06, end: 2025-08}     # YYYY-MM, YYYY, or "present"
    type: internship                # job | internship | research | program | leadership | club | volunteer
    section: experience             # optional: experience | leadership (where it is listed)
    ownership: contributor          # member < contributor < lead < founder / owner - the HIGHEST TRUE level
    tags: [software, data]          # help ranking; never printed
    skills: [Python, SQL]
    facts:
      - id: exp_acme_f01
        kind: responsibility        # responsibility | achievement | metric | tool_use | scope
        text: Assisted a 4-person team in developing a Python application
        ownership: contributor      # optional; overrides the role's level for this fact
        metrics: [4-person]         # the exact number strings allowed in output
        skills: [Python]            # every skill this fact genuinely shows
    bullets:                        # optional phrasings you have signed off
      - id: exp_acme_b01
        text: Assisted a 4-person team in developing a Python application.
        fact_ids: [exp_acme_f01]
        approved: true
```

Rules of thumb:

- **One claim per fact.** "Built X and raised Y" is two facts. Atomic facts let the tool cite
  exactly what a bullet rests on.
- **Numbers go in `metrics`, exactly as you would write them.** `20+` is not `20`; a bullet may say
  "20+" only if a cited fact does. The tool never adds, rounds, or "improves" a number.
- **Tag every skill a fact shows.** A bullet may name a skill only if a cited fact names it in its
  text or lists it under `skills`, and the skill is `verified` in `skills_status.yaml`.
- **Set ownership honestly.** It is a ceiling on verbs. With `contributor`, "led", "managed",
  "founded" are rejected; "built" and "created" are held for your approval. If a fact says
  "Assisted ...", a rewrite that drops "assisted" is flagged.
- **`approved: true` bullets are trusted wording.** They pass validation as written, and `--strict`
  uses them verbatim. Rewrites you approve can be saved back as approved bullets automatically.
- **`skills:` at the bottom** is the master list: the only skills that may appear in the resume's
  skills section. `awards:` and `certifications:` are optional.
- Education fields (`school`, `degree`, `major`, `grad`, `class_year`, `gpa`) and
  `work_authorization` are what the eligibility check compares against postings.

The tool rewrites this file only in two cases, each with a timestamped backup in `data/backups/`:
when you approve a fact in the interview, and when you choose to save approved rewrites. Comments
below the header block are not preserved by those rewrites.

## `skills_status.yaml`

```yaml
verified: [Python, Java, ...]     # may appear on a resume
learning: [SQL, ...]              # never on a resume; half credit in Job Match
missing:  [AWS, ...]              # a skill listed nowhere is also treated as missing
```

Only `verified` skills can be printed. All three statuses feed the dashboard, so adding the skills
that postings keep asking for under `learning` or `missing` makes the unlock analysis more useful.

## How truthfulness is enforced

`src/validate.py` is deterministic (no AI) and runs on every draft and every revision. Each bullet
is checked against the facts it cites.

| Check | Rule | If it fails |
|---|---|---|
| A structure | Cited facts and bullets must exist and belong to that experience. Employer, title, dates, and location are always copied from the master resume. | A fabricated or altered selection is removed and replaced with the next-most-relevant verified content. |
| B1 metrics | Every number, %, $, or count must appear in a cited fact. | hard |
| B2 skills | Every named skill or tool must be supported by a cited fact and be `verified`. | hard (a posting term or new name the facts never mention: approval) |
| B3 ownership | Verbs may not exceed the cited facts' ownership level. | hard; ambiguous verbs and dropped "assisted" wording: approval |
| B4 scope | "company-wide", "end-to-end", "production", "all", ... must be in the cited facts. | approval |
| B5 skills list | Only `verified` skills from the master list. | removed |
| C wording | Added technical-complexity or result wording; a dropped metric. | approval |

- **Hard** violations get exactly one retry (Claude is sent the specific problem). If the retry
  fails, the bullet reverts to an approved bullet or the plain fact text, or is dropped.
- **Soft** flags, any substantial rewording, any bullet built from a new interview fact, and (in
  `--strict`) every rewrite go to the **approval checkpoint**: original and rewrite side by side,
  with the reason, and `[A]pprove / [E]dit / [R]evert`. Enter reverts. An edit is re-validated.
- The change type (`unchanged` / `minor_wording` / `reworded` / `combined`) is measured in code, so
  a rewrite cannot be passed off as minor. Minor wording changes are auto-approved and still listed
  in `changes.md`.
- Research and the posting change **emphasis only**. If a company uses PyTorch, PyTorch does not
  appear unless your facts support it.

No validator catches every exaggeration, which is why anything ambiguous comes to you.

## How each score works

Claude never produces a total; `src/scoring.py` computes all three.

**Resume Quality (0-10)** is the only score that drives revisions.

| Category | Weight | Scored by |
|---|---|---|
| Impact (results, strong accurate verbs) | 25% | judge |
| Relevance of the chosen content | 25% | judge |
| Clarity and concision | 20% | judge |
| ATS formatting | 15% | code (`ats_check.py`) |
| Use of the posting's language, where truthful | 15% | judge |

The judge is a separate, cheaper model acting as a strict recruiter. It sees only the posting, the
trusted parts of the company brief, and the resume text (with your name and contact details
redacted). It never sees earlier scores or drafts. It judges writing separately from
qualifications: a missing qualification is reported as a `REAL_GAP` and does not lower Impact or
Clarity. Its fixes are typed `REWRITE` (fixable with existing facts), `ASK_USER` (a question for
you), or `REAL_GAP`.

The loop revises while the score is below target, and stops when: the target is reached; no
`REWRITE` fixes are left (only questions and gaps); `--max-rounds` is reached; or the last 2 rounds
each gained less than 0.3. The best-scoring draft is kept. If it stopped below target and the
judge has questions, the interview asks them; facts you approve are added (marked
`source: interview`), the resume is re-tailored, and the loop runs once more.

During the loop the ATS part uses the checks that need no file. The final score uses every check
on the real PDF and DOCX: contact info, PDF text extraction matching the intended text (missing
text, garbled characters, reading order), standard headings, odd characters, links, keyword
stuffing, reverse-chronological order, document structure, and exactly one page.

**Job Match (0-100)** measures *you*, not the writing, so rewriting cannot change it: 55% weighted
coverage of required skills (verified = 1, learning = 0.5), 15% preferred skills, 30% eligibility
(meets = 1, partial or unknown = 0.5, doesn't meet = 0). Unstated requirements are left out.

**Keyword Coverage (0-100)** is the weighted share of the posting's keywords found in a text, with
a synonym map (ML = machine learning, Postgres = PostgreSQL, ...). It is reported for your whole
master resume and for the final resume. Keywords missing from the whole master resume are things
you have no fact for; they are never added to raise the number.

## Company research and source tiers

Research uses Claude's server-side web search (at most 5 searches). Every finding is tagged **in
code** from its source URL: `official` (the company's own domain), `posting` (the job description
itself), `third_party` (reputable news), or `unverified` (forums, aggregators, and any source the
search did not actually return). The tailor weights official and posting items heavily, news
lightly, and never sees unverified ones. Research is cached per company for 24 hours.

## Editing the template

`template.docx` is filled with [docxtpl](https://docxtpl.readthedocs.io). Open it in Word and
change the paragraph styles named **Resume Name / Contact / Heading / Entry / Line / Bullet**
(font, size, spacing). Keep it ATS-safe: one column, no tables, no text boxes, no images, nothing
in the header or footer, and leave the `{{ ... }}` and `{%p ... %}` lines exactly as they are.
`python scripts/build_template.py` regenerates the original.

If the resume runs past one page the tool tries, in order, re-rendering after each step:
(1) Claude tightens the longest bullets, accepted only if re-validation raises nothing new;
(2) spacing, then margins, then font size, never below the limits in `config.py` (0.5" margins,
10pt body); (3) removing the lowest-relevance bullet, then project. Every step is in `changes.md`.

Section headings, the order of sections, skill-line labels, and the GPA label are in `config.py`.

## The dashboard

`python tailor.py --insights` writes `output/insights.html` and `output/insights.md`, computed in
code from `data/postings/`, `data/applications.csv`, and `skills_status.yaml`:

- **Skill demand:** each skill, how many postings require or prefer it, and your status.
- **Unlock analysis:** for each skill you are missing or learning, the postings where it is the
  *only* unmet required skill (or one of at most two). The top three are the skills that would make
  you eligible for the most additional postings.
- **Eligibility patterns:** "X% of postings require junior standing or above", minimum GPAs,
  citizenship and sponsorship rules, graduation windows that exclude you.
- **Scores over time** and an **outcome funnel** from the status column.

## Cost and privacy

- Token usage is logged per call (`usage.json`), summarized per step in `scores.md`, and the
  estimated cost is printed at the end of each run. Prices live in `config.py`.
- The fact-bank prompt is identical across the tailor, fix, revise, tighten, and cover-letter
  calls, and the judge's posting prompt is identical every round, so those are cached.
- Your name, phone, and email are never sent to Claude: the tailor works from the fact bank
  without the contact block, and the judge reads a redacted copy.
- Rate-limit and overload errors are retried with exponential backoff (5 retries).

## Configuration

Everything tunable is in `config.py`: models and effort per step, `TARGET_QUALITY`, `MAX_ROUNDS`,
`PLATEAU_DELTA`, `PLATEAU_ROUNDS`, `QUALITY_WEIGHTS`, `JOB_MATCH_WEIGHTS`, page-fit limits, the
ownership verb tiers (`OWNERSHIP_VERBS`), `SCOPE_WORDS`, `SYNONYMS`, the tool lexicon, the news
allowlist, and extra official domains per company.

## Tests

```bash
pytest          # about 315 tests, under a minute; no API key and no network needed
```

Claude is mocked everywhere. Covered: the validator (including "Assisted a team in developing a
Python application" -> "Led development of a Python application" **must fail**, invented metrics,
unverified skills, scope inflation, altered employers and dates), scoring math and synonym
matching, eligibility (graduation year, GPA, unknowns), all four loop stop conditions, page-fit
order, the ATS text-extraction comparison, the unlock analysis, and end-to-end runs that render
real PDFs with LibreOffice.

## Layout

```
tailor.py               CLI entry point
config.py               models, thresholds, weights, lexicons, paths
master_resume.yaml      verified fact bank (the source of truth)
skills_status.yaml      verified / learning / missing
template.docx           ATS-safe resume design
SPEC.md                 the original build spec
src/
  startup.py            .env, model verification, data-file validation
  llm.py                the one wrapper around the Claude API (structured output, web search, usage)
  fetch.py              link -> job text, cache, paste fallback
  analyze.py            job text -> requirements and keywords
  eligibility.py        requirement-by-requirement check
  research.py           web research -> source-tiered company brief
  tailor_resume.py      first draft
  validate.py           deterministic anti-fabrication checks
  judge.py              rubric scoring (category scores only)
  scoring.py            Resume Quality, Job Match, Keyword Coverage
  revise.py             applies REWRITE fixes
  interview.py          ASK_USER questions -> new verified facts
  approval.py           human approval checkpoint
  render.py             DOCX -> PDF, page fit
  ats_check.py          ATS compatibility checks
  report.py             eligibility / changes / scores / gaps, applications.csv
  archive.py            snapshots, --find
  insights.py           dashboard
  cover_letter.py       --cover-letter
  loop.py               orchestrator, batch
  models.py store.py textutil.py relevance.py ui.py errors.py
data/postings/          every analyzed posting (JSON)
data/applications.csv   one row per run
output/                 results (gitignored)
```

Beyond the original spec (all additive): `llm.py`, `store.py`, `textutil.py`, `relevance.py`,
`ui.py`, `cover_letter.py`; an `awards` list and an experience type `program` (competitions,
workshops) in the fact bank; an optional `section:` per experience; a JSON-LD fallback when a job
page renders with JavaScript; a `<Name>_Resume.pdf` copy; `--check`; and a final re-judge when
approval or page fitting changed the text after the loop.

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
| 12 | Application intelligence dashboard | done |
| 13 | Batch processing + optional cover letter | done |
Every Claude call is exercised by tests with scripted responses. The live API has not been called
yet from this checkout: `python tailor.py --check` is the first thing to run once `.env` has a key.

## Troubleshooting

- **"No .env file" / "ANTHROPIC_API_KEY is missing"** - copy `.env.example` to `.env` and add the key.
- **"These configured models are not available"** - use one of the IDs it lists.
- **"blocked the request" / "only N words were extracted"** - the site needs a login. Paste the
  description when asked, or save it to a file and use `--text`.
- **"No PDF converter found"** - install LibreOffice.
- **`master_resume.yaml failed validation`** - the message lists each problem (duplicate id, a
  bullet citing a fact from another entry, a metric that is not in its fact's text, a bad date).
- **The PDF font is not the one in the template** - `scores.md` notes this under "text extraction".
  On macOS the tool points LibreOffice at the system fonts; on Linux install the font.
