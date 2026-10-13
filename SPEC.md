# MASTER PROMPT v2: Resume Tailor & Application Intelligence System (Python + Claude API)

> The build spec for this project, saved verbatim on 2026-10-13 so later sessions
> can pick the build back up. `README.md` documents what was actually built.

You are an expert Python engineer. Build the complete project described below, one milestone at a time, following the build order at the bottom. Ask me before making any design change that contradicts this spec. Write clean, modular, well-commented code with type hints and pydantic models.

---

## 1. Goal

A command-line tool where I give a job posting link and get back:

1. An **eligibility check**: do I meet this posting's requirements?
2. A **one-page, ATS-safe resume tailored to the job**, built only from verified facts about me.
3. Three separate scores: **Resume Quality** (with a target of 9/10), **Job Match**, and **Keyword Coverage**.
4. A report covering what changed, what I approved, the score history, and my skill gaps.
5. An archived snapshot of exactly what I submitted.
6. Over time, an **application intelligence dashboard** showing which skills would unlock the most internships for me.

**Context:** I'm a freshman Data Engineering major (Class of 2030) applying to Summer 2027 internships and research programs in CS, data, and finance.

---

## 2. Core principle: never exaggerate

- The tool may only **select, reorder, reword, and emphasize** facts that exist in `master_resume.yaml`.
- It must never invent or inflate experience, skills, tools, employers, titles, dates, metrics, scope, or **level of ownership** (e.g. "assisted" must never become "led").
- Truthfulness is enforced **in code** (`validate.py`) and by **my approval** (`approval.py`), not only by prompts.
- Company research and job postings influence **emphasis only**. They never add claims about me. If a company uses PyTorch, PyTorch does not appear on my resume unless my master resume supports it.
- No deterministic validator can catch every exaggeration, so anything ambiguous goes to me for approval.
- If a 9/10 Resume Quality score isn't reachable honestly, the tool stops and reports why.

---

## 3. Tech stack

- Python 3.11+
- `anthropic` (official Claude API SDK)
- `requests` and `trafilatura` (job page fetching and text extraction)
- `PyYAML` (master resume), `pydantic` (schemas)
- `docxtpl` (template filling)
- PDF: LibreOffice headless (`soffice --headless --convert-to pdf`), with `docx2pdf` as a fallback
- `pypdf` (page count and text extraction for ATS checks)
- `python-dotenv`, `rich`, `pytest`
- Optional for the dashboard: `pandas`, `matplotlib`, or a static HTML report

**Models (configurable and verified)**
- Defaults in `config.py`: `FAST_MODEL = "claude-haiku-5-5"`, `WRITE_MODEL = "claude-sonnet-5-5"`.
- **Don't assume these IDs are valid.** At startup, verify both are available (list models via the API, or make a tiny test call). If either is unavailable, print a clear error and the available model IDs, and stop.
- Use tool use or structured output for **every** Claude call that returns data, and validate the results with pydantic.

---

## 4. Project layout

```
resume-tailor/
├── tailor.py                 # CLI entry point
├── config.py                 # models, thresholds, weights, paths
├── master_resume.yaml        # source of truth: verified facts + optional bullets
├── template.docx             # ATS-safe resume design (docxtpl placeholders)
├── skills_status.yaml        # my status per skill: verified / learning / missing
├── .env / .gitignore / requirements.txt / README.md
├── src/
│   ├── models.py             # all pydantic schemas
│   ├── startup.py            # env + model verification
│   ├── fetch.py              # link → clean job text
│   ├── analyze.py            # job text → structured requirements + keywords
│   ├── eligibility.py        # requirement-by-requirement eligibility check
│   ├── research.py           # web research → source-tiered company brief
│   ├── tailor_resume.py      # first tailored draft
│   ├── validate.py           # deterministic evidence-based anti-fabrication
│   ├── judge.py              # rubric scoring (category scores only)
│   ├── scoring.py            # weighted totals + keyword coverage + job match (code)
│   ├── revise.py             # applies REWRITE fixes
│   ├── interview.py          # ASK_USER questions → new verified facts
│   ├── approval.py           # human approval checkpoint
│   ├── render.py             # docx → pdf, page-fit optimization
│   ├── ats_check.py          # ATS compatibility checks
│   ├── report.py             # changes, scores, gaps
│   ├── archive.py            # resume version snapshots
│   ├── insights.py           # application intelligence dashboard
│   └── loop.py               # orchestrator
├── tests/
├── data/
│   ├── applications.csv
│   └── postings/             # every analyzed JobAnalysis as JSON (for insights)
└── output/
    └── <Company>_<Role>_<YYYY-MM-DD>/
```

---

## 5. Master resume: verified fact bank (`master_resume.yaml`)

Store **detailed factual material** about each experience, not just polished bullets. Claude builds bullets from these facts and emphasizes different facts for different job types: technical work for software roles, analytical and business work for finance, methodology for research.

```yaml
contact:
  name: ""
  email: ""
  phone: ""
  location: ""
  linkedin: ""
  github: ""

education:
  - id: edu_tamu
    school: "Texas A&M University"
    degree: "B.S. Engineering, Data Engineering"
    grad: "May 2030"
    class_year: 2030
    gpa: 4.0
    honors: ["University Honors Program"]
    coursework: []

work_authorization: ""      # e.g. "US citizen" — used only by eligibility.py

experience:                 # jobs, research, leadership, clubs
  - id: exp_example
    title: ""
    org: ""
    location: ""
    dates: { start: "2025-08", end: "present" }
    type: leadership        # job | internship | research | leadership | club | volunteer
    ownership: contributor  # member | contributor | lead | founder | owner — highest TRUE level for this role
    tags: [leadership, data, finance]
    skills: [python, sql]
    facts:                  # atomic, verified statements, each with its own ID
      - id: exp_example_f01
        kind: responsibility      # responsibility | achievement | metric | tool_use | scope
        text: "Assisted a 4-person team in developing a Python application"
        ownership: contributor    # ownership for THIS fact (overrides role-level)
        metrics: ["4-person"]     # exact strings allowed to appear in output
        skills: [python]
    bullets:                # optional pre-approved phrasings
      - id: exp_example_b01
        text: "Assisted a 4-person team in developing a Python application"
        fact_ids: [exp_example_f01]
        approved: true

projects:
  - id: proj_example
    name: ""
    link: ""
    dates: { start: "", end: "" }
    ownership: owner
    tags: []
    skills: []
    facts: []
    bullets: []

skills:                     # master list — the ONLY skills allowed on output
  languages: []
  tools: []
  concepts: []

certifications: []
```

Create `skills_status.yaml` listing each skill as `verified`, `learning`, or `missing`. Only `verified` skills may appear on the resume. All three statuses feed the dashboard.

---

## 6. CLI

```
python tailor.py <job_url>
python tailor.py --text job.txt              # paste fallback
python tailor.py <job_url> --strict          # original wording unless I approve rewrites
python tailor.py <job_url> --no-research
python tailor.py <job_url> --no-interview
python tailor.py <job_url> --max-rounds 5 --target 9.0
python tailor.py --batch links.txt
python tailor.py --insights                  # regenerate the dashboard
python tailor.py --find "Company"            # look up archived submissions
```

---

## 7. Module specs

### `startup.py`
- Load `.env`, verify the API key exists, verify the model IDs (see §3), and validate `master_resume.yaml` against the schema (unique IDs, required fields). Fail fast with a clear message.

### `fetch.py`
- `fetch_job(url) -> str`: one request with a normal browser User-Agent, main text extracted with `trafilatura`.
- If the text is under about 300 words or looks like a login or captcha wall (LinkedIn, Workday, and Handshake often are), prompt me to paste the description (end with a blank line), or accept `--text`.
- Fetch only the URL I give, never crawl. Cache by URL for 24 hours.

### `analyze.py` (Haiku)
Output `JobAnalysis`:
- `company`, `role`, `location`, `job_type` (software / data / finance / research / other)
- `required_skills[]` and `preferred_skills[]`, each with a weight (1–3)
- `tools[]`, `key_phrases[]`, `responsibilities[]`, `keywords[]` (flat and de-duplicated)
- `requirements`: `grad_year_range`, `class_standing`, `degree_fields[]`, `min_gpa`, `work_authorization`, `prior_experience_required`, and `other[]`. Each includes the exact source sentence from the posting, or null if unstated.

Save every `JobAnalysis` to `data/postings/` for the dashboard.

### `eligibility.py` (pure Python where possible)
- Compare `JobAnalysis.requirements` with my master resume and output a table:

  | Requirement | Status | Evidence |
  |---|---|---|
  | Graduation year / class standing | Meets / Doesn't meet / Unknown | quote from posting |
  | Degree field | … | … |
  | GPA | … | … |
  | Work authorization | … | … |
  | Required skills | Meets / Partial / Doesn't meet (list missing) | … |
  | Prior internship/research experience | … | … |

- **Flag, never auto-reject.** Show the table first, then ask "Continue tailoring? [Y/n]". In batch mode, continue but put the flags at the top of the report.
- Stated requirements are compared in code. Use Claude only to interpret vague wording (e.g. "rising juniors"), and label those results "interpreted".

### `research.py` (Sonnet + the Anthropic server-side web search tool)
- Check the current Anthropic docs for the exact web search tool type string and parameters. Cap it at about 5 searches.
- Output `CompanyBrief` with every item tagged by source tier:
  1. `official`: the company's own site, careers pages, or engineering blog
  2. `posting`: the job description itself
  3. `third_party`: reputable news or reporting
  4. `unverified`: forums, aggregators, anything else
- Fields: `summary`, `tech_stack[]`, `values_language[]`, `intern_work[]`, `recent_news[]`, `sources[]`, each tagged with its tier.
- The tailor weights tiers 1–2 heavily, tier 3 lightly, and ignores tier 4 when choosing emphasis.
- **Research may only change what gets emphasized. It may never add claims, skills, or tools to the resume.**

### `tailor_resume.py` (Sonnet)
- Input: master resume, `JobAnalysis`, `CompanyBrief`, and the mode (normal or strict).
- Output `TailoredResume`. For each bullet: `rewritten_text`, `fact_ids[]` (the facts it's built from), `source_bullet_id` (if adapted from an existing bullet), `change_type` (`unchanged` / `minor_wording` / `reworded` / `combined`), and `rationale`.
- Also outputs: section order, selected experiences and projects, and `skills[]` (verified skills only).
- Prompt rules:
  - Each bullet may only state what its cited facts support.
  - Keep ownership verbs at or below each fact's `ownership` level.
  - Mirror the posting's language only where true. Use strong but accurate verbs. Never add numbers.
  - Choose facts that fit the `job_type` (technical for software, analytical for finance, methodology for research).
- **`--strict` mode:** use `approved: true` bullets verbatim. A new rewrite is proposed only alongside its original, and only my approval puts it in the resume.

### `validate.py` (deterministic, no AI)
Run on every draft and every revision.

**A. Structural checks**
- Every `fact_id` and `source_bullet_id` exists.
- Employer, title, dates, and location are **always copied from the master resume** and never come from the AI.
- **Invalid selections** (a nonexistent experience, an altered employer or title or date, a fabricated ID) are **removed and replaced** with the next-most-relevant verified content. Falling back to "original bullet text" isn't enough when the selection itself is invalid.

**B. Evidence checks per bullet**
1. **Metrics:** every number, %, $, or count must match a `metrics` string, or appear in the text, of a cited fact.
2. **Skills/tools:** every named skill or tool must be in the cited facts' `skills` or text, **and** be `verified` in `skills_status.yaml`.
3. **Ownership:** maintain a verb-tier lexicon in `config.py`, for example:
   - `member`: participated, attended, supported
   - `contributor`: assisted, contributed, helped, collaborated, developed (as part of a team)
   - `lead`: led, managed, directed, headed, coordinated, supervised
   - `founder/owner`: founded, created, launched, built (solo), established, owned

   If a bullet's verbs sit at a tier **above** its cited facts' `ownership`, that's a violation.
4. **Scope inflation:** flag scope words (e.g. "company-wide", "end-to-end", "full-stack", "large-scale", "production", "multiple", "all") that don't appear in the cited facts.
5. **Skills list:** output `skills[]` must be a subset of the verified skills.

**C. Outcomes**
- **Hard violations** (A, or B1, B2, B5) trigger one retry with the specific violation sent to Claude. If it fails again, revert to an approved bullet or the plain fact text, or drop the bullet.
- **Soft flags** (B3 edge cases, B4, and any `change_type` of `reworded` or `combined`) go to `approval.py`.
- Log everything to `changes.md`.

### `scoring.py` (pure Python, computes all totals)
Three separate scores. **Claude never produces a total.**

1. **Resume Quality** (the only score with a 9/10 target): the weighted average of the judge's category scores, computed in Python.

   | Category | Weight |
   |---|---|
   | Impact (results, strong accurate verbs) | 25% |
   | Relevance of chosen content to this role | 25% |
   | Clarity and concision | 20% |
   | ATS formatting (from `ats_check.py`, in code) | 15% |
   | Use of the posting's language (where truthful) | 15% |

   ATS formatting is scored by code, not the judge.
2. **Job Match** (informational): how well my verified experience covers the job's weighted requirements, computed from `JobAnalysis` vs. the master resume and eligibility results. It doesn't change when the resume gets rewritten, because it measures *me*, not the writing.
3. **Keyword Coverage** (informational): the weighted percentage of job keywords present in the resume text, with a synonym map (ML = machine learning, Postgres = PostgreSQL, etc.). Report it for the master resume and the final resume.

### `judge.py` (Haiku, independent)
- It sees only the job posting, the tier-1/2 parts of the company brief, and the current resume text. It never sees previous scores, drafts, or tailor reasoning.
- It acts as a strict technical recruiter for this role.
- Returns `category_scores` (1–10 per judged category, with a one-line justification each) and `fixes[]`. Each fix has:
  - `type`:
    - `REWRITE`: fixable using existing verified facts (reword, reorder, swap facts)
    - `ASK_USER`: I might have relevant experience not yet in the master resume, phrased as a question to me
    - `REAL_GAP`: a requirement I don't meet
  - `target`, `detail`, and `expected_gain`
- The judge must judge **writing quality** separately from **qualifications**. Missing qualifications go in `REAL_GAP` and don't lower the Impact or Clarity scores.

### `revise.py` (Sonnet)
- Applies only `REWRITE` fixes, using only verified facts (it may swap in unselected facts and bullets).
- Re-validated every round.

### `loop.py` (orchestrator)
```
startup checks
job_text    = fetch
analysis    = analyze(job_text)          → save to data/postings/
eligibility = check(analysis, master)    → show table, ask to continue
brief       = research(company)          unless --no-research
draft       = tailor(...) → validate
judgment    = judge(draft); quality = scoring.quality(judgment, ats)
best        = draft

while quality < TARGET:
    rewrite_fixes = [f for f in judgment.fixes if f.type == REWRITE]
    if not rewrite_fixes: break                    # only gaps/questions left
    if rounds >= MAX_ROUNDS: break
    if last PLATEAU_ROUNDS rounds each gained < PLATEAU_DELTA: break
    draft = revise(draft, rewrite_fixes) → validate
    judgment = judge(draft); quality = scoring.quality(...)
    keep best-scoring draft

if quality < TARGET and ASK_USER fixes exist and interview enabled:
    interview → new facts (with my OK) → re-tailor → loop once more

approval checkpoint (approval.py)
render → page-fit → ats_check
compute job_match + keyword_coverage
report → archive → append applications.csv
print summary
```
- Only **Resume Quality** drives the loop. Job Match and Keyword Coverage never trigger revisions.
- Show live progress with `rich`.

### `interview.py`
- Asks `ASK_USER` questions one at a time, and I can skip any of them.
- Turns my answers into **structured facts** (kind, text, ownership, metrics, skills). Shows me the proposed fact for **Approve / Edit / Reject**.
- Back up `master_resume.yaml` with a timestamp before any write. New facts are marked `source: interview` with the date.

### `approval.py` (human checkpoint)
- Before rendering, show each flagged change side by side: **Original fact/bullet → Rewritten**, with the reason it was flagged.
- Options: **[A]pprove / [E]dit / [R]evert**.
- Changes that **always** require approval:
  - Changes to ownership or responsibility wording
  - Changes to technical complexity
  - Anything touching metrics or results
  - Changes to project scope
  - Bullets built from new interview facts
  - Any soft flag from `validate.py`
  - In `--strict`, every rewrite
- Minor wording and formatting changes are auto-approved but listed in `changes.md`.
- Approved rewrites can be saved back to `master_resume.yaml` as `approved: true` bullets, if I choose, so future runs need fewer approvals.

### `render.py` (page-fit optimization)
Fill `template.docx` (single column, no tables or text boxes for content, standard headings, common font), then convert to PDF. If it's over one page, try these steps **in order**, re-rendering after each:
1. Ask Claude to tighten wording on the longest bullets (no meaning change, re-validated).
2. Adjust spacing and margins **within ATS-safe limits** set in `config.py` (e.g. margins ≥ 0.5", body font ≥ 10pt).
3. Remove the lowest-relevance content (bullet, then project), one at a time.

Log every step.

### `ats_check.py` (pure Python)
On the final PDF and DOCX:
- Contact info present (name, email, phone).
- **PDF text extraction matches the intended content:** extract with `pypdf`, normalize, and compare to the rendered source text. Report mismatches like missing sections, garbled characters, and wrong reading order.
- Required section headings present (Education, Experience, Skills; Projects if used).
- No unsupported or odd characters (fancy bullets, emoji, ligature issues).
- Links are well-formed. Optionally check that they resolve (HEAD request to my own links only).
- No excessive keyword repetition (flag any keyword used more than about 3 times).
- Entries in reverse chronological order within each section.
- Exactly one page.

The results feed the ATS category of Resume Quality and appear in the report.

### `report.py`
In the output folder:
- `eligibility.md`: the requirement table.
- `changes.md`: per bullet, the original vs. final text, cited fact IDs, change type, approval decision, validation flags, and page-fit removals.
- `scores.md`: Resume Quality per round with category breakdowns, Job Match, Keyword Coverage (master vs. final), and the ATS check results.
- `gaps.md`: `REAL_GAP`s and unmet requirements.
- Append to `data/applications.csv`: date, company, role, url, quality, job_match, keyword_coverage, eligibility_flags, rounds, output_folder, and status (applied / interview / offer / rejected, which I edit by hand).

### `archive.py` (resume versions)
For every run, save into the output folder:
- the job posting text and URL
- a **snapshot of `master_resume.yaml`** at that moment
- the final `resume.docx` and `resume.pdf`
- the `TailoredResume` JSON, the score breakdown, the company brief, eligibility, and the date

`--find "Company"` lists matching archived submissions so I can pull up exactly what I sent if a recruiter contacts me months later.

### `insights.py` (application intelligence dashboard)
All numbers are **computed in code** from `data/postings/`, `applications.csv`, and `skills_status.yaml`. Claude never guesses them.
- **Skill demand table:** each skill, the number of postings requiring or preferring it, and my status (Verified / Learning / Missing).
- **Unlock analysis:** for each skill I'm missing or learning, count the postings where it's the *only* (or one of ≤2) unmet required skills. Rank the **top 3 skills that would make me eligible for the most additional postings**.
- Eligibility patterns, e.g. "X% of postings require junior+ standing".
- Average Resume Quality, Job Match, and Keyword Coverage over time, plus an outcome funnel once I fill in statuses.
- Output as a static `output/insights.html` (simple charts) plus `insights.md`.

---

## 8. Config defaults (`config.py`)
```
TARGET_QUALITY = 9.0
MAX_ROUNDS = 5
PLATEAU_DELTA = 0.3
PLATEAU_ROUNDS = 2
MAX_RESEARCH_SEARCHES = 5
MAX_PAGES = 1
MIN_MARGIN_IN = 0.5
MIN_FONT_PT = 10
KEYWORD_REPEAT_LIMIT = 3
FAST_MODEL = "claude-haiku-5-5"     # verified at startup
WRITE_MODEL = "claude-sonnet-5-5"   # verified at startup
QUALITY_WEIGHTS = {impact: .25, relevance: .25, clarity: .20, ats: .15, language: .15}
OWNERSHIP_VERBS = {...}              # tiered lexicon (see validate.py)
SCOPE_WORDS = [...]
```

---

## 9. Engineering requirements
- The API key is loaded from `.env` only, and `.env` and `output/` are gitignored.
- Exponential backoff on rate-limit and overload errors.
- Log token usage per call and print the run cost at the end.
- Cache fetches and research for 24 hours.
- Clear error messages, no raw stack traces for expected failures.
- **Tests (`pytest`), with mocked Claude responses, for:**
  - `validate.py`, especially the ownership-inflation case: "Assisted a team in developing a Python application" → "Led development of a Python application" **must fail**. Also invented metrics, unverified skills, scope inflation, and altered employers or dates.
  - `scoring.py` (weighted math, synonym matching)
  - `eligibility.py` (grad year, GPA, unknowns)
  - the loop's stop conditions (target reached, no REWRITE fixes, max rounds, plateau)
  - page-fit order
  - ATS text-extraction comparison
  - `insights.py` unlock analysis
- The README covers setup, writing facts in `master_resume.yaml`, `skills_status.yaml`, editing the template, every CLI flag, and how each score works.

---

## 10. Build order (one milestone at a time; show me each and wait before continuing)

| # | Milestone |
|---|---|
| 1 | Project scaffold, config, pydantic schemas, startup checks (incl. model verification) |
| 2 | Job posting extraction (`fetch.py`) + paste fallback |
| 3 | Job analysis + eligibility checking |
| 4 | Master resume fact bank, `skills_status.yaml`, evidence-based `validate.py` (with tests) |
| 5 | Resume tailoring (normal + `--strict`) |
| 6 | DOCX/PDF rendering, page-fit optimization, ATS checks |
| 7 | Independent judge + code-computed scoring (3 scores) |
| 8 | Iterative revision loop with all stop conditions |
| 9 | Company research with source tiers |
| 10 | Interview system + human approval checkpoint |
| 11 | Reports, resume version archive, application tracking, cost monitoring |
| 12 | Application intelligence dashboard |
| 13 | Batch processing + optional `--cover-letter` (reuses the same facts, validation, and approval) |

> Standing instruction from 2026-10-13: build straight through all milestones
> without waiting for sign-off between them; save progress to the Obsidian vault
> and push to GitHub after each milestone.
