"""Central configuration: models, thresholds, weights, lexicons, and paths.

Everything a user might reasonably tune lives here. Nothing in this file calls
the network or reads the master resume; it is plain data so tests can import
it freely.

Environment overrides (read from `.env` by `src/startup.py`):
    FAST_MODEL, WRITE_MODEL   - override the model IDs below
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MASTER_RESUME_PATH = ROOT / "master_resume.yaml"
SKILLS_STATUS_PATH = ROOT / "skills_status.yaml"
TEMPLATE_PATH = ROOT / "template.docx"
ENV_PATH = ROOT / ".env"

DATA_DIR = ROOT / "data"
POSTINGS_DIR = DATA_DIR / "postings"
APPLICATIONS_CSV = DATA_DIR / "applications.csv"
CACHE_DIR = DATA_DIR / "cache"            # gitignored
BACKUP_DIR = DATA_DIR / "backups"         # timestamped master_resume.yaml backups
OUTPUT_DIR = ROOT / "output"              # gitignored

# --------------------------------------------------------------------------
# Models (verified against the API at startup; see src/startup.py)
# --------------------------------------------------------------------------
FAST_MODEL = os.environ.get("FAST_MODEL", "claude-haiku-5-5")    # analyze, judge, interpret
WRITE_MODEL = os.environ.get("WRITE_MODEL", "claude-sonnet-5-5")  # tailor, revise, research

# Effort per task. Both models think adaptively; effort is the only depth knob.
# (Sampling parameters such as temperature are rejected by these models.)
EFFORT = {
    "analyze": "medium",
    "interpret": "low",
    "judge": "high",        # strict instruction following matters most here
    "tailor": "medium",
    "revise": "medium",
    "fix": "low",           # single-bullet repair after a validation failure
    "tighten": "low",       # page-fit wording tightening
    "research": "medium",
    "brief": "low",         # structuring research notes into a CompanyBrief
    "interview": "low",
    "cover_letter": "medium",
}

MAX_TOKENS = 16000          # per call; thinking counts toward this cap
API_MAX_RETRIES = 5         # SDK-level exponential backoff on 429 / 5xx / 529
API_TIMEOUT_SECONDS = 300.0

# Server-side refusal fallback. Supported by WRITE-tier models on the Claude
# API only; Haiku has no server-side fallback, so it is never sent there.
ENABLE_REFUSAL_FALLBACK = True
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_CAPABLE_PREFIXES = ("claude-sonnet-5-5", "claude-opus-5", "claude-fable-5")

# Server-side web search tool used by research.py (see README, "Research").
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
MAX_RESEARCH_SEARCHES = 5
MAX_PAUSE_CONTINUATIONS = 3

# --------------------------------------------------------------------------
# Pricing (USD per million tokens) for the end-of-run cost report.
# Cached 2026-10; edit here if Anthropic's price list changes.
# --------------------------------------------------------------------------
PRICING = {
    # model-id prefix: (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Haiku 5.5 bills prompts longer than 100K tokens on a higher rate card.
LONG_PROMPT_THRESHOLD = 100_000
LONG_PROMPT_PRICING = {"claude-haiku-5-5": (0.50, 2.50)}
CACHE_READ_MULTIPLIER = 0.10    # cache reads cost 0.1x the input rate
CACHE_WRITE_MULTIPLIER = 1.25   # 5-minute cache writes cost 1.25x
WEB_SEARCH_COST_PER_SEARCH = 0.01   # $10 per 1,000 searches (estimate)

# --------------------------------------------------------------------------
# Loop thresholds
# --------------------------------------------------------------------------
TARGET_QUALITY = 9.0
MAX_ROUNDS = 5
PLATEAU_DELTA = 0.3
PLATEAU_ROUNDS = 2

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
QUALITY_WEIGHTS = {
    "impact": 0.25,      # results, strong accurate verbs          (judge)
    "relevance": 0.25,   # chosen content fits this role           (judge)
    "clarity": 0.20,     # clear and concise                       (judge)
    "ats": 0.15,         # ATS formatting                          (code: ats_check.py)
    "language": 0.15,    # truthful use of the posting's language  (judge)
}
JUDGED_CATEGORIES = ("impact", "relevance", "clarity", "language")

# Job Match: how well *I* cover the job (never changes with rewriting).
JOB_MATCH_WEIGHTS = {"required_skills": 0.55, "preferred_skills": 0.15, "eligibility": 0.30}
LEARNING_SKILL_CREDIT = 0.5      # partial credit for a skill marked "learning"
UNKNOWN_ELIGIBILITY_CREDIT = 0.5

# Keyword Coverage: preferred skills count for less than required ones.
PREFERRED_KEYWORD_FACTOR = 0.5
PLAIN_KEYWORD_WEIGHT = 1.0

# ATS check weights (sum need not be 1; ats_check normalizes).
ATS_CHECK_WEIGHTS = {
    "contact_info": 2.0,
    "text_extraction": 3.0,
    "section_headings": 2.0,
    "odd_characters": 1.5,
    "links": 0.5,
    "keyword_repetition": 0.5,
    "reverse_chronological": 1.0,
    "page_count": 2.0,
}
KEYWORD_REPEAT_LIMIT = 3
EXTRACTION_MATCH_THRESHOLD = 0.97   # share of source text that must survive PDF extraction

# A rewrite at least this similar to its verified source (word-level) counts as
# "minor_wording" and is auto-approved unless another check flags it.
MINOR_WORDING_SIMILARITY = 0.80

# --------------------------------------------------------------------------
# Rendering / page fit (ATS-safe limits)
# --------------------------------------------------------------------------
MAX_PAGES = 1
MIN_MARGIN_IN = 0.5
MIN_FONT_PT = 10
DEFAULT_MARGIN_IN = 0.75
DEFAULT_FONT_PT = 11.0
DEFAULT_SPACING_SCALE = 1.0     # multiplies paragraph spacing in the template
MIN_SPACING_SCALE = 0.4
MARGIN_STEP_IN = 0.05
FONT_STEP_PT = 0.5
SPACING_STEP = 0.2
BODY_FONT = "Times New Roman"
TARGET_BULLETS = 15             # bullet budget suggested to the tailor for one page
MAX_BULLETS_PER_ENTRY = 3
MIN_BULLETS_PER_ENTRY = 1
REPLACEMENT_MAX_BULLETS = 2     # bullets used when validate.py swaps in a verified entry
TIGHTEN_BULLET_COUNT = 5        # how many of the longest bullets Claude may tighten
TIGHTEN_MIN_CHARS = 110         # bullets shorter than this are left alone
PAGE_FIT_MAX_STEPS = 60
PDF_CONVERT_TIMEOUT_SECONDS = 120
SOFFICE_CANDIDATES = (
    "soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/opt/homebrew/bin/soffice",
    "/usr/bin/soffice",
    "/usr/local/bin/soffice",
)

# Section layout. `SECTION_FOR_TYPE` decides which heading an experience lands under.
SECTION_HEADINGS = {
    "education": "EDUCATION",
    "experience": "EXPERIENCE",
    "projects": "PROJECTS",
    "leadership": "LEADERSHIP",
    "awards": "AWARDS AND RECOGNITION",
    "skills": "SKILLS",
    "certifications": "CERTIFICATIONS",
}
DEFAULT_SECTION_ORDER = ["education", "experience", "projects", "leadership", "skills", "awards", "certifications"]
REQUIRED_SECTIONS = ("education", "experience", "skills")
SECTION_FOR_TYPE = {
    "job": "experience",
    "internship": "experience",
    "research": "experience",
    "program": "experience",
    "leadership": "leadership",
    "club": "leadership",
    "volunteer": "leadership",
}
# Experience types that count as "prior internship / research experience".
PRIOR_EXPERIENCE_TYPES = {
    "internship": {"internship"},
    "research": {"research"},
    "work": {"job", "internship"},
    "any": {"job", "internship", "research"},
}
SKILL_CATEGORY_LABELS = {"languages": "Languages", "tools": "Tools and Platforms", "concepts": "Skills"}
GPA_LABEL = "Cumulative GPA (GPR)"
GPA_SCALE = 4.0

# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
FETCH_TIMEOUT_SECONDS = 25
MIN_JOB_WORDS = 300
CACHE_TTL_HOURS = 24
# Phrases that mean we got a login / bot wall instead of the posting.
WALL_MARKERS = (
    "sign in to view", "sign in to continue", "log in to continue", "please log in",
    "join now to see", "join linkedin", "create an account to", "verify you are human",
    "verify you are a human", "are you a robot", "captcha", "enable javascript",
    "javascript is required", "access denied", "unusual activity", "checking your browser",
    "cloudflare", "authwall", "session has expired",
)
# Hosts that usually need a login or render with JavaScript only.
WALLED_HOSTS = ("linkedin.com", "myworkdayjobs.com", "workday.com", "joinhandshake.com", "handshake.com", "indeed.com", "glassdoor.com")

# --------------------------------------------------------------------------
# Research source tiers
# --------------------------------------------------------------------------
# Job boards / ATS hosts: never treated as a company's "official" domain.
JOB_BOARD_HOSTS = (
    "linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "joinhandshake.com",
    "greenhouse.io", "lever.co", "myworkdayjobs.com", "workday.com", "ashbyhq.com",
    "smartrecruiters.com", "icims.com", "jobvite.com", "wellfound.com", "simplify.jobs",
    "builtin.com", "dice.com", "monster.com", "levels.fyi", "teamblind.com",
)
REPUTABLE_NEWS_DOMAINS = (
    "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com", "ft.com", "cnbc.com",
    "apnews.com", "bbc.com", "bbc.co.uk", "washingtonpost.com", "economist.com",
    "forbes.com", "fortune.com", "businessinsider.com", "axios.com", "techcrunch.com",
    "theverge.com", "wired.com", "arstechnica.com", "venturebeat.com", "theinformation.com",
    "npr.org", "marketwatch.com", "barrons.com", "sec.gov", "hbr.org", "nature.com",
    "ieee.org", "acm.org", "zdnet.com", "cnn.com", "theguardian.com", "politico.com",
)
# Manual overrides: company slug -> extra official domains (e.g. {"goldmansachs": ["gs.com"]}).
OFFICIAL_DOMAIN_OVERRIDES: dict[str, list[str]] = {
    "goldmansachs": ["gs.com"],
    "jpmorganchase": ["jpmorgan.com", "chase.com"],
    "alphabet": ["google.com", "abc.xyz"],
    "meta": ["meta.com", "metacareers.com", "fb.com"],
}
# Tiers the tailor may use when choosing emphasis, with their relative weight.
TAILOR_TIER_WEIGHTS = {"official": "heavy", "posting": "heavy", "third_party": "light"}
JUDGE_TIERS = ("official", "posting")

# --------------------------------------------------------------------------
# Ownership lexicon (validate.py, B3)
# --------------------------------------------------------------------------
OWNERSHIP_TIERS = {"member": 1, "contributor": 2, "lead": 3, "founder": 4, "owner": 4}

# Verb lemmas by the LOWEST ownership tier that may honestly use them.
OWNERSHIP_VERBS = {
    "member": ["participate", "attend", "support", "join", "observe", "shadow"],
    "contributor": ["assist", "contribute", "help", "collaborate", "develop", "co-develop", "aid"],
    "lead": ["lead", "co-lead", "manage", "direct", "head", "coordinate", "supervise",
             "oversee", "spearhead", "chair", "captain"],
    "founder": ["found", "co-found", "create", "launch", "build", "establish", "own",
                "pioneer", "originate"],
}
# Irregular or restricted surface forms. Verbs not listed get regular
# inflection (lemma, +s, +ed/+d, +ing) from src/textutil.py.
VERB_FORMS = {
    "lead": ["lead", "leads", "led", "leading"],
    "co-lead": ["co-lead", "co-leads", "co-led", "co-leading"],
    "build": ["build", "builds", "built", "building"],
    "found": ["founded", "founding", "founds"],          # bare "found" is usually "find"
    "co-found": ["co-founded", "co-founding", "cofounded"],
    "oversee": ["oversee", "oversees", "oversaw", "overseen", "overseeing"],
    "direct": ["directed", "directing", "directs"],      # bare "direct" is usually an adjective
    "head": ["headed", "heading up", "heads up"],
    "own": ["owned", "owns", "owning"],                   # bare "own" is usually a determiner
    "chair": ["chaired", "chairing", "chairs"],
    "captain": ["captained", "captaining"],
    "support": ["support", "supports", "supported", "supporting"],
    "aid": ["aided", "aiding"],
    "shadow": ["shadowed", "shadowing"],
    "observe": ["observed", "observing"],
    "help": ["help", "helps", "helped", "helping"],
    "join": ["joined", "joining"],
}
# A surface form followed by one of these words is not an ownership claim
# ("leading to a 20% gain", "lead generation", "supporting documents").
VERB_FORM_EXCLUSIONS = {
    "leading": ["to"],
    "lead": ["generation", "time", "times"],
    "supporting": ["documents", "materials", "evidence"],
    "building": ["blocks", "codes"],
}
# Verbs whose ownership meaning depends on context ("built" solo vs. as part of
# a team). Above-tier use is a SOFT flag (approval) rather than a hard failure.
AMBIGUOUS_OWNERSHIP_VERBS = {"build", "create", "develop", "co-develop", "coordinate"}
# Words that mark a helper role. If the evidence has one and the rewrite drops
# every one of them, the bullet is flagged ("Assisted in building" -> "Built").
DOWNGRADE_QUALIFIERS = {"assist", "help", "support", "contribute", "participate",
                        "aid", "shadow", "observe", "attend", "collaborate"}
# Title nouns that claim a tier ("team lead", "founder").
OWNERSHIP_NOUNS = {
    "lead": ["team lead", "tech lead", "project lead", "team leader", "leader", "manager",
             "supervisor", "director", "head of", "captain", "chairperson", "president"],
    "founder": ["founder", "co-founder", "cofounder", "owner", "creator"],
}

# --------------------------------------------------------------------------
# Scope, complexity, and result lexicons (validate.py B4 + approval triggers)
# --------------------------------------------------------------------------
SCOPE_WORDS = [
    "company-wide", "organization-wide", "enterprise-wide", "firm-wide", "school-wide",
    "university-wide", "campus-wide", "district-wide", "state-wide", "statewide",
    "nationwide", "national", "international", "global", "worldwide",
    "end-to-end", "full-stack", "large-scale", "at scale", "production", "enterprise",
    "multiple", "all", "entire", "every", "numerous", "extensive", "comprehensive",
    "cross-functional", "mission-critical", "major", "significant", "significantly",
    "complete", "fully", "several", "various", "many", "countless", "widespread",
    "organization", "department-wide", "single-handedly", "independently", "solely",
]
COMPLEXITY_WORDS = [
    "advanced", "complex", "sophisticated", "scalable", "robust", "optimized", "optimised",
    "architected", "engineered", "algorithm", "algorithms", "pipeline", "pipelines",
    "framework", "frameworks", "distributed", "real-time", "high-performance",
    "production-grade", "automated", "automation", "deep learning", "ai-powered",
    "model", "models", "modeling", "statistical", "quantitative", "predictive",
    "infrastructure", "backend", "back-end", "frontend", "front-end", "database",
    "databases", "microservices", "cloud", "neural", "system", "systems", "architecture",
    "state-of-the-art", "cutting-edge", "proprietary", "novel",
]
RESULT_WORDS = [
    "improve", "increase", "reduce", "decrease", "boost", "grow", "save", "cut",
    "accelerate", "achieve", "deliver", "drive", "generate", "raise", "win", "earn",
    "exceed", "outperform", "optimize", "streamline", "enhance", "double", "triple",
    "maximize", "minimize", "secure", "award",
]
RESULT_PHRASES = ["resulting in", "resulted in", "leading to", "led to", "which led"]

# Vague "more than" markers. A metric written with one of these must be backed
# by a fact that also says so ("20+" in a fact supports "over 20"; "20" does not).
MORE_THAN_MARKERS = ["over", "more than", "at least", "upwards of", "exceeding", "above", "nearly", "almost", "approximately", "about", "around", "roughly"]
STRICT_MORE_MARKERS = ["over", "more than", "at least", "upwards of", "exceeding", "above"]

# --------------------------------------------------------------------------
# Synonyms: canonical term -> other surface forms. Used by keyword coverage,
# skill matching, and validation. Upper-case variants match case-sensitively.
# --------------------------------------------------------------------------
SYNONYMS = {
    "machine learning": ["ML"],
    "artificial intelligence": ["AI"],
    "natural language processing": ["NLP"],
    "large language models": ["LLM", "LLMs", "large language model"],
    "postgresql": ["postgres", "psql"],
    "javascript": ["JS"],
    "typescript": ["TS"],
    "python": ["python3", "python 3"],
    "sql": ["structured query language"],
    "aws": ["amazon web services"],
    "gcp": ["google cloud platform", "google cloud"],
    "azure": ["microsoft azure"],
    "ci/cd": ["cicd", "continuous integration", "continuous delivery", "continuous deployment"],
    "ui/ux": ["ui ux", "ux/ui", "user interface", "user experience", "UI", "UX"],
    "excel": ["microsoft excel", "ms excel"],
    "data visualization": ["data viz", "dataviz", "data visualisation"],
    "data analysis": ["data analytics", "analyzing data", "analyze data", "data analyses"],
    "etl": ["extract transform load", "extract, transform, load"],
    "kubernetes": ["k8s"],
    "node.js": ["nodejs", "node js"],
    "react": ["reactjs", "react.js"],
    "c++": ["cpp"],
    "c#": ["csharp", "c sharp"],
    "object-oriented programming": ["OOP", "object oriented programming", "object-oriented design"],
    "api": ["apis", "application programming interface"],
    "rest": ["rest api", "rest apis", "restful"],
    "power bi": ["powerbi"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "cad": ["computer-aided design", "computer aided design"],
    "financial modeling": ["financial modelling", "financial models"],
    "dcf": ["discounted cash flow"],
    "test automation": ["automated testing", "automation testing", "test automation suites"],
    "git": ["version control"],
    "github": ["git hub"],
    "html": ["html5"],
    "css": ["css3"],
    "statistics": ["statistical analysis", "statistical methods"],
    "technical analysis": ["financial technical analysis"],
    "public speaking": ["presenting", "presentations", "presentation skills"],
    "teaching": ["tutoring", "instruction"],
    "communication": ["communication skills", "written and verbal communication"],
    "teamwork": ["team player", "collaboration"],
    "problem solving": ["problem-solving"],
    "pine script": ["pinescript"],
    "tradingview": ["trading view"],
    "playwright": ["playwright automation"],
}

# Terms that collide with ordinary English words: matched case-sensitively.
CASE_SENSITIVE_TERMS = {
    "Go", "Rust", "Swift", "Spark", "Excel", "Oracle", "Looker", "REST", "SAS", "R", "C",
    "Chef", "Puppet", "Flask", "Spring", "Access", "Word", "Teams", "Notion", "Slack",
    "Unity", "Ruby", "Dart", "Julia", "Hive", "Pig", "Storm", "Kafka", "Ray", "Dash",
    "Stata", "Alteryx", "Git", "Jira", "Linux", "Azure", "Canva", "Figma", "Scala",
    "Shell", "Bash", "Make", "Ant", "Maven", "Snowflake", "Airflow", "Bloomberg",
}

# Built-in vocabulary of named tools / technologies / hard skills. A bullet that
# names one of these must cite a fact that supports it (validate.py B2).
TECH_LEXICON = [
    # languages
    "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go", "Rust", "Ruby",
    "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB", "SQL", "NoSQL", "HTML", "CSS",
    "Bash", "Shell", "Pine Script", "VBA", "Julia", "Dart", "Perl", "Haskell", "Solidity",
    # frameworks / libraries
    "React", "Angular", "Vue", "Next.js", "Node.js", "Express", "Django", "Flask",
    "FastAPI", "Spring", "Spring Boot", ".NET", "Rails", "Flutter", "React Native",
    "PyTorch", "TensorFlow", "Keras", "scikit-learn", "pandas", "NumPy", "SciPy",
    "Matplotlib", "Seaborn", "Plotly", "OpenCV", "LangChain", "Hugging Face", "XGBoost",
    # data
    "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Snowflake", "BigQuery", "Redshift",
    "Databricks", "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Oracle",
    "SQL Server", "DynamoDB", "Elasticsearch", "Hive", "ETL",
    # cloud / devops
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "Git",
    "GitHub", "GitLab", "GitHub Actions", "GitHub Pages", "CI/CD", "Linux", "Jira",
    # analytics / finance
    "Tableau", "Power BI", "Excel", "Looker", "SAS", "Stata", "SPSS", "Alteryx",
    "Bloomberg", "Bloomberg Terminal", "FactSet", "Capital IQ", "TradingView",
    "StockCharts", "Investopedia Simulator", "QuickBooks", "DCF", "financial modeling",
    "valuation", "technical analysis", "financial statement analysis",
    # testing / design / hardware
    "Playwright", "Selenium", "Cypress", "pytest", "JUnit", "Figma", "Canva",
    "Raspberry Pi", "Arduino", "SolidWorks", "AutoCAD", "CAD", "ROS",
    # concepts
    "machine learning", "deep learning", "natural language processing", "computer vision",
    "large language models", "data analysis", "data visualization", "data engineering",
    "data science", "statistics", "A/B testing", "REST", "GraphQL", "API",
    "object-oriented programming", "test automation", "agile", "scrum",
]
