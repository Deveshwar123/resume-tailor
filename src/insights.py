"""Application intelligence dashboard.

Every number here is computed in code from three files - Claude never guesses:

    data/postings/*.json     every JobAnalysis saved by analyze.py
    data/applications.csv    one row per run (I fill in the status column)
    skills_status.yaml       my status per skill

Outputs `output/insights.html` (static page, inline SVG charts, no network)
and `output/insights.md`:

* Skill demand: each skill, how many postings require or prefer it, my status.
* Unlock analysis: for each skill I am missing or learning, the postings where
  it is the ONLY unmet required skill (or one of at most two). The top three
  are the skills that would make me eligible for the most additional postings.
* Eligibility patterns: "X% of postings require junior+ standing", ...
* Average Resume Quality, Job Match, and Keyword Coverage over time.
* An outcome funnel, once I fill in statuses.
"""

from __future__ import annotations

import html
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import config
from src import analyze, eligibility, report, textutil, ui
from src.models import ClassStanding, JobAnalysis, MasterResume, SkillsStatus, SkillStatus, WorkAuthKind

TOP_SKILLS_IN_CHART = 15
TOP_UNLOCKS_IN_CHART = 8
FUNNEL_STAGES = ["drafted", "applied", "interview", "offer"]
# A row's status places it at this stage or later.
_REACHED = {
    "drafted": {"drafted", "applied", "interview", "offer", "rejected"},
    "applied": {"applied", "interview", "offer", "rejected"},
    "interview": {"interview", "offer"},
    "offer": {"offer"},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SkillDemand:
    name: str
    required: int
    preferred: int
    status: str                  # verified | learning | missing
    listed: bool                 # False when the skill is not in skills_status.yaml at all

    @property
    def total(self) -> int:
        return self.required + self.preferred


@dataclass
class Unlock:
    name: str
    status: str
    sole: int                    # postings where this is the ONLY unmet required skill
    within_two: int              # postings where it is one of at most two unmet required skills
    required_in: int
    postings: list[str] = field(default_factory=list)    # "Role at Company" for the sole-blocker postings


@dataclass
class Pattern:
    label: str
    count: int
    total: int
    note: str = ""

    @property
    def pct(self) -> float:
        return 100.0 * self.count / self.total if self.total else 0.0


@dataclass
class Period:
    month: str                   # YYYY-MM
    count: int
    quality: float
    job_match: float
    keyword_coverage: float


@dataclass
class Insights:
    generated_on: str
    postings: int = 0
    applications: int = 0
    fully_covered: int = 0                       # postings whose required skills I already meet
    demand: list[SkillDemand] = field(default_factory=list)
    unlocks: list[Unlock] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    periods: list[Period] = field(default_factory=list)
    averages: dict[str, float] = field(default_factory=dict)
    statuses: dict[str, int] = field(default_factory=dict)
    funnel: list[tuple[str, int]] = field(default_factory=list)
    job_types: dict[str, int] = field(default_factory=dict)

    def top_unlocks(self, n: int = 3) -> list[Unlock]:
        return [u for u in self.unlocks if u.within_two > 0][:n]


# ---------------------------------------------------------------------------
# Computation (pure)
# ---------------------------------------------------------------------------


def dedupe_postings(postings: list[JobAnalysis]) -> list[JobAnalysis]:
    """One entry per distinct posting text: re-running a job must not double-count it."""
    latest: dict[str, JobAnalysis] = {}
    for posting in postings:
        digest = posting.id.rsplit("_", 1)[-1]
        key = digest if re.fullmatch(r"[0-9a-f]{8}", digest) else f"{posting.company}|{posting.role}"
        latest[key] = posting                     # postings arrive oldest first
    return list(latest.values())


def _status(status: SkillsStatus, name: str) -> tuple[str, bool]:
    mine = status.status_of(name)
    return (mine.value if mine else SkillStatus.missing.value, mine is not None)


def skill_demand(postings: list[JobAnalysis], status: SkillsStatus) -> list[SkillDemand]:
    table: dict[str, SkillDemand] = {}
    for posting in postings:
        for group, attribute in ((posting.required_skills, "required"), (posting.preferred_skills, "preferred")):
            seen: set[str] = set()
            for skill in group:
                key = textutil.canon(skill.name)
                if key in seen or not eligibility.assessable(skill.name, skill.kind, status):
                    continue
                seen.add(key)
                if key not in table:
                    mine, listed = _status(status, skill.name)
                    table[key] = SkillDemand(name=skill.name, required=0, preferred=0, status=mine, listed=listed)
                setattr(table[key], attribute, getattr(table[key], attribute) + 1)
    return sorted(table.values(), key=lambda d: (-d.total, -d.required, d.name.lower()))


def unmet_required(posting: JobAnalysis, status: SkillsStatus) -> dict[str, str]:
    """{canonical: display name} of required skills I cannot claim as verified."""
    unmet: dict[str, str] = {}
    for skill in posting.required_skills:
        if eligibility.assessable(skill.name, skill.kind, status) and not status.is_verified(skill.name):
            unmet.setdefault(textutil.canon(skill.name), skill.name)
    return unmet


def unlock_analysis(postings: list[JobAnalysis], status: SkillsStatus) -> tuple[list[Unlock], int]:
    """Rank the skills that would unlock the most postings. Returns (ranking, fully covered count)."""
    table: dict[str, Unlock] = {}
    fully_covered = 0
    for posting in postings:
        unmet = unmet_required(posting, status)
        if not unmet:
            fully_covered += 1 if posting.required_skills else 0
            continue
        for key, name in unmet.items():
            if key not in table:
                table[key] = Unlock(name=name, status=_status(status, name)[0], sole=0, within_two=0, required_in=0)
            entry = table[key]
            entry.required_in += 1
            if len(unmet) == 1:
                entry.sole += 1
                entry.postings.append(f"{posting.role} at {posting.company}")
            if len(unmet) <= 2:
                entry.within_two += 1
    ranking = sorted(table.values(), key=lambda u: (-u.sole, -u.within_two, -u.required_in, u.name.lower()))
    return ranking, fully_covered


def eligibility_patterns(postings: list[JobAnalysis], master: MasterResume | None = None) -> list[Pattern]:
    total = len(postings)
    if not total:
        return []
    juniors_up = {ClassStanding.junior, ClassStanding.senior, ClassStanding.graduate}
    standing = [p.requirements.class_standing.allowed for p in postings]
    gpas = [p.requirements.min_gpa.value for p in postings if p.requirements.min_gpa.value is not None]
    auth = [p.requirements.work_authorization.kind for p in postings]
    patterns = [
        Pattern("require junior standing or above", sum(1 for a in standing if a and set(a) <= juniors_up), total),
        Pattern("are explicitly open to freshmen", sum(1 for a in standing if ClassStanding.freshman in a), total),
        Pattern("state a minimum GPA", len(gpas), total,
                f"median minimum {statistics.median(gpas):.2f}" if gpas else ""),
        Pattern("require U.S. citizenship", auth.count(WorkAuthKind.citizen_required), total),
        Pattern("require a U.S. person (citizen or permanent resident)", auth.count(WorkAuthKind.us_person_required), total),
        Pattern("do not sponsor visas", auth.count(WorkAuthKind.no_sponsorship), total),
        Pattern("require prior internship, research, or work experience",
                sum(1 for p in postings if p.requirements.prior_experience_required.required), total),
    ]
    grad = eligibility.candidate_grad(master) if master is not None else None
    if grad is not None:
        excluded = 0
        for posting in postings:
            window = posting.requirements.grad_year_range
            low = eligibility._bound(window.earliest, end=False)
            high = eligibility._bound(window.latest, end=True)
            if (low and grad < low) or (high and grad > high):
                excluded += 1
        patterns.append(Pattern(f"have a graduation window that excludes mine ({grad[0]})", excluded, total))
    return patterns


def _number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _mean(values: list[float | None]) -> float:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 2) if kept else 0.0


def score_trends(applications: list[dict[str, str]]) -> tuple[list[Period], dict[str, float]]:
    by_month: dict[str, list[dict[str, str]]] = {}
    for row in applications:
        month = (row.get("date") or "")[:7]
        if re.fullmatch(r"\d{4}-\d{2}", month):
            by_month.setdefault(month, []).append(row)
    periods = [
        Period(month=month, count=len(rows),
               quality=_mean([_number(r.get("quality")) for r in rows]),
               job_match=_mean([_number(r.get("job_match")) for r in rows]),
               keyword_coverage=_mean([_number(r.get("keyword_coverage")) for r in rows]))
        for month, rows in sorted(by_month.items())
    ]
    averages = {
        "quality": _mean([_number(r.get("quality")) for r in applications]),
        "job_match": _mean([_number(r.get("job_match")) for r in applications]),
        "keyword_coverage": _mean([_number(r.get("keyword_coverage")) for r in applications]),
    }
    return periods, averages


def outcome_funnel(applications: list[dict[str, str]]) -> tuple[dict[str, int], list[tuple[str, int]]]:
    statuses: dict[str, int] = {}
    for row in applications:
        status = (row.get("status") or report.INITIAL_STATUS).strip().lower() or report.INITIAL_STATUS
        statuses[status] = statuses.get(status, 0) + 1
    funnel = [(stage, sum(n for status, n in statuses.items() if status in _REACHED[stage])) for stage in FUNNEL_STAGES]
    return statuses, funnel


def compute(postings: list[JobAnalysis], applications: list[dict[str, str]], status: SkillsStatus,
            master: MasterResume | None = None, today: date | None = None) -> Insights:
    postings = dedupe_postings(postings)
    unlocks, fully_covered = unlock_analysis(postings, status)
    periods, averages = score_trends(applications)
    statuses, funnel = outcome_funnel(applications)
    job_types: dict[str, int] = {}
    for posting in postings:
        job_types[posting.job_type.value] = job_types.get(posting.job_type.value, 0) + 1
    return Insights(
        generated_on=(today or date.today()).isoformat(),
        postings=len(postings), applications=len(applications), fully_covered=fully_covered,
        demand=skill_demand(postings, status), unlocks=unlocks,
        patterns=eligibility_patterns(postings, master), periods=periods, averages=averages,
        statuses=statuses, funnel=funnel, job_types=job_types,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _unlock_sentence(unlock: Unlock) -> str:
    return (f"**{unlock.name}** ({unlock.status}) - the only missing required skill in {unlock.sole} posting(s); "
            f"one of at most two in {unlock.within_two}")


def render_markdown(ins: Insights) -> str:
    parts = ["# Application insights", "",
             f"Generated {ins.generated_on} from {ins.postings} analyzed posting(s) and {ins.applications} logged "
             "application(s). Every number is computed in code.", ""]
    if not ins.postings:
        parts.append("No postings have been analyzed yet. Run `python tailor.py <job_url>` a few times first.")
        return "\n".join(parts) + "\n"

    parts += ["## Skills that would unlock the most postings", ""]
    top = ins.top_unlocks(3)
    if top:
        parts += [f"{i}. {_unlock_sentence(u)}" for i, u in enumerate(top, 1)]
    else:
        parts.append("No single skill is blocking a posting right now.")
    parts += ["", f"I already meet every required skill in {ins.fully_covered} of {ins.postings} posting(s).", ""]
    if ins.unlocks:
        parts += [report.table(["Skill", "My status", "Only missing skill in", "One of <= 2 missing in", "Required by"],
                               [[u.name, u.status, u.sole, u.within_two, u.required_in] for u in ins.unlocks]), ""]

    parts += ["## Skill demand", "",
              report.table(["Skill", "Required by", "Preferred by", "Total", "My status"],
                           [[d.name, d.required, d.preferred, d.total,
                             d.status.title() + ("" if d.listed else " (not listed)")] for d in ins.demand]), "",
              "_A skill that is not in skills_status.yaml counts as missing. Soft skills are only counted when listed there._", ""]

    parts += ["## Eligibility patterns", ""]
    parts += [f"- {p.pct:.0f}% of postings {p.label} ({p.count} of {p.total})" + (f"; {p.note}" if p.note else "")
              for p in ins.patterns]
    parts += ["", "## Scores over time", ""]
    if ins.periods:
        parts += [report.table(["Month", "Applications", "Avg Resume Quality", "Avg Job Match", "Avg Keyword Coverage"],
                               [[p.month, p.count, f"{p.quality:.2f}", f"{p.job_match:.1f}", f"{p.keyword_coverage:.1f}%"]
                                for p in ins.periods]), "",
                  f"Overall: Resume Quality {ins.averages['quality']:.2f}, Job Match {ins.averages['job_match']:.1f}, "
                  f"Keyword Coverage {ins.averages['keyword_coverage']:.1f}%.", ""]
    else:
        parts += ["No applications have been logged yet.", ""]

    parts += ["## Outcome funnel", ""]
    if ins.applications:
        parts += [report.table(["Stage reached", "Applications"], [[stage.title(), n] for stage, n in ins.funnel]), ""]
        parts.append("Statuses: " + ", ".join(f"{n} {status}" for status, n in sorted(ins.statuses.items())) + ".")
        if set(ins.statuses) <= {report.INITIAL_STATUS}:
            parts.append("\nEvery row is still `drafted`. Edit the `status` column of data/applications.csv "
                         "(applied / interview / offer / rejected) to fill in this funnel.")
    else:
        parts.append("No applications have been logged yet.")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# HTML (static, inline SVG; palette and mark specs follow the dataviz reference)
# ---------------------------------------------------------------------------

CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--strong:#2a78d6;--soft:#86b6ef;--track:#cde2fb}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--strong:#3987e5;--soft:#184f95;--track:#0d366b}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--strong:#3987e5;--soft:#184f95;--track:#0d366b}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
min-height:100vh;padding:32px 20px 64px}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;font-weight:650}
h2{font-size:17px;margin:0 0 2px;font-weight:650}
.sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px 22px;margin:0 0 18px}
.card>p{color:var(--ink2);margin:0 0 14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:0 0 18px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.tile .label{color:var(--ink2);font-size:13px}
.tile .value{font-size:30px;font-weight:650;line-height:1.2}
.tile .note{color:var(--muted);font-size:12.5px}
.unlock .value{font-size:22px}
.rank{color:var(--muted);font-size:12.5px;letter-spacing:.02em}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font:12.5px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink2)}
svg .name{fill:var(--ink);font-size:13px}
svg .tag,svg .tick{fill:var(--muted);font-size:12px}
svg .tick{font-variant-numeric:tabular-nums}
svg .val{fill:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.strong{fill:var(--strong)}.soft{fill:var(--soft)}
.line{fill:none;stroke:var(--strong);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{fill:var(--strong);stroke:var(--surface);stroke-width:2}
.mark{cursor:default;outline:none}
.mark:hover,.mark:focus-visible{filter:brightness(1.12)}
.hit{fill:transparent}
.cross{stroke:var(--axis);stroke-width:1;visibility:hidden}
.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:0 0 10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.shape{fill:var(--ink2);stroke:var(--ink2);stroke-width:1.5}
.shape.hollow{fill:none}
.meter{display:grid;grid-template-columns:minmax(220px,1fr) minmax(120px,280px) 54px;gap:12px;align-items:center;
padding:7px 0;border-top:1px solid var(--grid)}
.meter:first-of-type{border-top:0}
.meter .track{height:10px;border-radius:5px;background:var(--track);overflow:hidden}
.meter .fill{height:100%;background:var(--strong);border-radius:5px 0 0 5px}
.meter .pct{text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.meter small{color:var(--muted);display:block}
.multiples{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}
.multiples h3{font-size:14px;margin:0;font-weight:600}
.multiples .now{font-size:22px;font-weight:650}
details{margin-top:12px}
summary{cursor:pointer;color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:13.5px}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.empty{color:var(--ink2)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);padding:7px 10px;border-radius:6px;
font-size:12.5px;line-height:1.35;max-width:280px;opacity:0;transition:opacity .08s;z-index:10}
#tip b{display:block;font-size:14px}
#tip span{opacity:.85}
footer{color:var(--muted);font-size:12.5px;margin-top:8px}
@media (max-width:640px){.meter{grid-template-columns:1fr 90px 46px}}
"""

JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(value,label,x,y){
    tip.textContent='';
    var b=document.createElement('b');b.textContent=value;tip.appendChild(b);
    var s=document.createElement('span');s.textContent=label;tip.appendChild(s);
    var w=tip.offsetWidth,h=tip.offsetHeight;
    tip.style.left=Math.max(8,Math.min(window.innerWidth-w-8,x+14))+'px';
    tip.style.top=Math.max(8,y-h-12)+'px';tip.style.opacity=1;
  }
  function hide(){tip.style.opacity=0;}
  document.querySelectorAll('.mark').forEach(function(m){
    m.addEventListener('pointermove',function(e){show(m.dataset.v,m.dataset.l,e.clientX,e.clientY);});
    m.addEventListener('pointerleave',hide);
    m.addEventListener('focus',function(){var r=m.getBoundingClientRect();show(m.dataset.v,m.dataset.l,r.right,r.top+r.height/2);});
    m.addEventListener('blur',hide);
  });
  document.querySelectorAll('svg[data-points]').forEach(function(svg){
    var pts=JSON.parse(svg.dataset.points),cross=svg.querySelector('.cross');
    function at(clientX){
      var box=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal,x=(clientX-box.left)/box.width*vb.width,best=pts[0];
      pts.forEach(function(p){if(Math.abs(p.x-x)<Math.abs(best.x-x))best=p;});
      cross.setAttribute('x1',best.x);cross.setAttribute('x2',best.x);cross.style.visibility='visible';
      show(best.v,best.l,box.left+best.x/vb.width*box.width,box.top+best.y/vb.height*box.height);
    }
    svg.addEventListener('pointermove',function(e){at(e.clientX);});
    svg.addEventListener('pointerleave',function(){cross.style.visibility='hidden';hide();});
  });
})();
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _nice_step(maximum: float, target: int = 5) -> int:
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if maximum / step <= target:
            return step
    return max(1, int(maximum // target))


def _bar_path(x: float, y: float, width: float, height: float, round_end: bool) -> str:
    """A horizontal bar segment: square at the baseline side, 4px rounded at the data end."""
    radius = min(4.0, height / 2, width) if round_end else 0.0
    right = x + width
    if radius <= 0:
        return f"M{x:.1f},{y:.1f}H{right:.1f}V{y + height:.1f}H{x:.1f}Z"
    return (f"M{x:.1f},{y:.1f}H{right - radius:.1f}Q{right:.1f},{y:.1f} {right:.1f},{y + radius:.1f}"
            f"V{y + height - radius:.1f}Q{right:.1f},{y + height:.1f} {right - radius:.1f},{y + height:.1f}H{x:.1f}Z")


_STATUS_SHAPE = {
    # A shape plus a word carries my status, so it never depends on color.
    "verified": '<circle cx="0" cy="0" r="4" class="shape"/>',
    "learning": '<path d="M0,-4A4,4 0 0 1 0,4Z" class="shape"/><circle cx="0" cy="0" r="4" class="shape hollow"/>',
    "missing": '<circle cx="0" cy="0" r="4" class="shape hollow"/>',
}


def _stacked_bars(rows: list[dict], strong_label: str, soft_label: str) -> str:
    """Horizontal stacked bars: `rows` = [{name, tag, status, strong, soft, strong_tip, soft_tip}]."""
    label_w, tag_w, right_pad, row_h, bar_h, top = 210, 86, 46, 30, 14, 8
    width = 900
    plot_x, plot_w = label_w + tag_w, width - label_w - tag_w - right_pad
    maximum = max((r["strong"] + r["soft"] for r in rows), default=1) or 1
    step = _nice_step(maximum)
    scale_max = step * (int((maximum - 1) // step) + 1)
    height = top + row_h * len(rows) + 26
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(strong_label)} and {_e(soft_label)} by skill">']
    for tick in range(0, scale_max + 1, step):
        x = plot_x + plot_w * tick / scale_max
        out.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 22}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{height - 6}" text-anchor="middle">{tick}</text>')
    for index, row in enumerate(rows):
        y = top + index * row_h + (row_h - bar_h) / 2
        mid = top + index * row_h + row_h / 2
        out.append(f'<text class="name" x="0" y="{mid + 4:.1f}">{_e(row["name"])}</text>')
        out.append(f'<g transform="translate({label_w + 6},{mid:.1f})">{_STATUS_SHAPE.get(row["status"], "")}</g>')
        out.append(f'<text class="tag" x="{label_w + 16}" y="{mid + 4:.1f}">{_e(row["tag"])}</text>')
        x = float(plot_x)
        segments = [("strong", row["strong"], row["strong_tip"]), ("soft", row["soft"], row["soft_tip"])]
        drawn = [s for s in segments if s[1] > 0]
        for position, (css, value, tip) in enumerate(drawn):
            seg_w = plot_w * value / scale_max
            gap = 2.0 if position > 0 else 0.0          # 2px surface gap between stacked segments
            path = _bar_path(x + gap, y, max(seg_w - gap, 1.0), bar_h, round_end=position == len(drawn) - 1)
            out.append(f'<path class="mark {css}" tabindex="0" d="{path}" data-v="{_e(value)}" data-l="{_e(tip)}"/>')
            x += seg_w
        out.append(f'<text class="val" x="{x + 7:.1f}" y="{mid + 4:.1f}">{row["strong"] + row["soft"]}</text>')
    out.append("</svg>")
    return "".join(out)


def _line_chart(points: list[tuple[str, float]], y_max: float, unit: str, title: str, digits: int = 1) -> str:
    """One small-multiple line chart with its own y-axis (never a shared second axis)."""
    width, height, left, right, top, bottom = 320, 150, 34, 14, 12, 24
    plot_w, plot_h = width - left - right, height - top - bottom
    step = y_max / 2
    coords = []
    for index, (label, value) in enumerate(points):
        x = left + (plot_w * index / (len(points) - 1) if len(points) > 1 else plot_w / 2)
        y = top + plot_h * (1 - min(value, y_max) / y_max)
        coords.append({"x": round(x, 1), "y": round(y, 1), "v": f"{value:.{digits}f}{unit}", "l": f"{title}, {label}"})
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_e(title)} by month" '
           f"data-points='{_e(json.dumps(coords))}'>"]
    for tick in (0, step, y_max):
        y = top + plot_h * (1 - tick / y_max)
        out.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{y + 4:.1f}" text-anchor="end">{tick:g}</text>')
    out.append(f'<line class="cross" x1="0" x2="0" y1="{top}" y2="{top + plot_h}"/>')
    if len(coords) > 1:
        out.append('<polyline class="line" points="' + " ".join(f'{c["x"]},{c["y"]}' for c in coords) + '"/>')
    for c in coords:
        out.append(f'<circle class="dot" cx="{c["x"]}" cy="{c["y"]}" r="4"/>')
    for index in sorted({0, len(points) - 1}):
        anchor = "start" if index == 0 and len(points) > 1 else ("end" if len(points) > 1 else "middle")
        out.append(f'<text class="tick" x="{coords[index]["x"]}" y="{height - 6}" text-anchor="{anchor}">{_e(points[index][0])}</text>')
    out.append(f'<rect class="hit" x="0" y="0" width="{width}" height="{height}"/></svg>')
    return "".join(out)


def _table(headers: list[tuple[str, bool]], rows: list[list[object]]) -> str:
    numeric_attr = ' class="n"'
    head = "".join(f"<th{numeric_attr if numeric else ''}>{_e(text)}</th>" for text, numeric in headers)
    body = "".join(
        "<tr>" + "".join(f"<td{numeric_attr if headers[i][1] else ''}>{_e(value)}</td>" for i, value in enumerate(row)) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(ins: Insights) -> str:
    body: list[str] = ['<div class="wrap">', "<h1>Application insights</h1>",
                       f'<p class="sub">Generated {_e(ins.generated_on)} from {ins.postings} analyzed posting(s) and '
                       f"{ins.applications} logged application(s). Every number is computed in code.</p>"]
    if not ins.postings:
        body.append('<div class="card"><p class="empty">No postings have been analyzed yet. Run '
                    "<code>python tailor.py &lt;job_url&gt;</code> a few times, then regenerate this page.</p></div>")
    else:
        # Headline numbers are stat tiles, not charts.
        tiles = [("Postings analyzed", str(ins.postings), ", ".join(f"{n} {t}" for t, n in sorted(ins.job_types.items()))),
                 ("I meet every required skill in", f"{ins.fully_covered}", f"of {ins.postings} postings")]
        if ins.applications:
            tiles += [("Avg Resume Quality", f"{ins.averages['quality']:.1f}", f"out of 10, {ins.applications} application(s)"),
                      ("Avg Job Match", f"{ins.averages['job_match']:.0f}", "out of 100"),
                      ("Avg Keyword Coverage", f"{ins.averages['keyword_coverage']:.0f}%", "of each posting's keywords")]
        body.append('<div class="tiles">' + "".join(
            f'<div class="tile"><div class="label">{_e(label)}</div><div class="value">{_e(value)}</div>'
            f'<div class="note">{_e(note)}</div></div>' for label, value, note in tiles) + "</div>")

        # Unlock analysis: the top three as tiles, the ranking as bars, and a table view.
        body.append('<div class="card"><h2>Skills that would unlock the most postings</h2>'
                    "<p>For each skill I cannot claim yet: how many postings list it as the only required skill "
                    "I am missing, and how many more list it as one of two.</p>")
        top = ins.top_unlocks(3)
        if top:
            body.append('<div class="tiles">' + "".join(
                f'<div class="tile unlock"><div class="rank">#{i}</div><div class="value">{_e(u.name)}</div>'
                f'<div class="note">only missing skill in {u.sole} posting(s); one of two in {u.within_two - u.sole} more '
                f"&middot; my status: {_e(u.status)}</div></div>" for i, u in enumerate(top, 1)) + "</div>")
            rows = [{"name": u.name, "tag": u.status.title(), "status": u.status, "strong": u.sole,
                     "soft": u.within_two - u.sole,
                     "strong_tip": f"{u.name}: postings where it is the only missing required skill",
                     "soft_tip": f"{u.name}: postings where it is one of two missing required skills"}
                    for u in ins.unlocks[:TOP_UNLOCKS_IN_CHART] if u.within_two > 0]
            body.append('<div class="legend"><span><i style="background:var(--strong)"></i>Only missing required skill</span>'
                        '<span><i style="background:var(--soft)"></i>One of two missing</span></div>')
            body.append(_stacked_bars(rows, "Only missing required skill", "One of two missing"))
            body.append("<details><summary>Table view</summary>" + _table(
                [("Skill", False), ("My status", False), ("Only missing skill in", True), ("One of <= 2 missing in", True),
                 ("Required by", True)],
                [[u.name, u.status, u.sole, u.within_two, u.required_in] for u in ins.unlocks]) + "</details>")
        else:
            body.append('<p class="empty">No single skill is blocking a posting right now.</p>')
        body.append("</div>")

        # Skill demand.
        body.append('<div class="card"><h2>Skill demand</h2><p>How many analyzed postings require or prefer each skill, '
                    "with my current status beside it.</p>")
        rows = [{"name": d.name, "tag": d.status.title(), "status": d.status, "strong": d.required, "soft": d.preferred,
                 "strong_tip": f"{d.name}: postings that require it", "soft_tip": f"{d.name}: postings that prefer it"}
                for d in ins.demand[:TOP_SKILLS_IN_CHART]]
        body.append('<div class="legend"><span><i style="background:var(--strong)"></i>Required</span>'
                    '<span><i style="background:var(--soft)"></i>Preferred</span>'
                    '<span><svg width="12" height="12" viewBox="-6 -6 12 12" style="display:inline;width:12px;vertical-align:-1px">'
                    '<circle r="4" class="shape"/></svg> verified &nbsp; '
                    '<svg width="12" height="12" viewBox="-6 -6 12 12" style="display:inline;width:12px;vertical-align:-1px">'
                    '<path d="M0,-4A4,4 0 0 1 0,4Z" class="shape"/><circle r="4" class="shape hollow"/></svg> learning &nbsp; '
                    '<svg width="12" height="12" viewBox="-6 -6 12 12" style="display:inline;width:12px;vertical-align:-1px">'
                    '<circle r="4" class="shape hollow"/></svg> missing</span></div>')
        body.append(_stacked_bars(rows, "Required", "Preferred"))
        if len(ins.demand) > TOP_SKILLS_IN_CHART:
            body.append(f'<p class="empty" style="margin-top:10px">Showing the top {TOP_SKILLS_IN_CHART} of {len(ins.demand)} skills; '
                        "the table has all of them.</p>")
        body.append(_table([("Skill", False), ("Required by", True), ("Preferred by", True), ("Total", True), ("My status", False)],
                           [[d.name, d.required, d.preferred, d.total, d.status.title() + ("" if d.listed else " (not listed)")]
                            for d in ins.demand]))
        body.append("</div>")

        # Eligibility patterns: single ratios, so meters.
        body.append('<div class="card"><h2>Eligibility patterns</h2><p>Share of analyzed postings with each requirement.</p>')
        for p in ins.patterns:
            body.append(f'<div class="meter"><div>{_e(p.label[0].upper() + p.label[1:])}'
                        f'<small>{p.count} of {p.total}{"; " + _e(p.note) if p.note else ""}</small></div>'
                        f'<div class="track" role="img" aria-label="{p.pct:.0f} percent"><div class="fill" style="width:{p.pct:.1f}%"></div></div>'
                        f'<div class="pct">{p.pct:.0f}%</div></div>')
        body.append("</div>")

    # Scores over time: three small multiples, one measure and one axis each.
    body.append('<div class="card"><h2>Scores over time</h2>')
    if ins.periods:
        body.append("<p>Monthly averages across logged applications. Each measure has its own chart and scale.</p>"
                    '<div class="multiples">')
        for title, key, y_max, unit, digits in (("Resume Quality", "quality", 10.0, "", 2),
                                                ("Job Match", "job_match", 100.0, "", 0),
                                                ("Keyword Coverage", "keyword_coverage", 100.0, "%", 0)):
            points = [(p.month, getattr(p, key)) for p in ins.periods]
            body.append(f'<div><h3>{title}</h3><div class="now">{points[-1][1]:.{digits}f}{unit}</div>'
                        + _line_chart(points, y_max, unit, title, digits) + "</div>")
        body.append("</div><details><summary>Table view</summary>" + _table(
            [("Month", False), ("Applications", True), ("Avg Resume Quality", True), ("Avg Job Match", True),
             ("Avg Keyword Coverage", True)],
            [[p.month, p.count, f"{p.quality:.2f}", f"{p.job_match:.1f}", f"{p.keyword_coverage:.1f}%"] for p in ins.periods])
            + "</details>")
    else:
        body.append('<p class="empty">No applications have been logged yet.</p>')
    body.append("</div>")

    # Outcome funnel: one series, one color; the value sits at each bar's tip.
    body.append('<div class="card"><h2>Outcome funnel</h2>')
    if ins.applications:
        body.append("<p>How many applications reached each stage, from the status column of data/applications.csv.</p>")
        rows = [{"name": stage.title(), "tag": "", "status": "", "strong": n, "soft": 0,
                 "strong_tip": f"applications that reached the {stage} stage", "soft_tip": ""} for stage, n in ins.funnel]
        body.append(_stacked_bars(rows, "Applications", "none"))
        body.append("<details><summary>Table view</summary>" + _table(
            [("Status", False), ("Applications", True)], [[s, n] for s, n in sorted(ins.statuses.items())]) + "</details>")
        if set(ins.statuses) <= {report.INITIAL_STATUS}:
            body.append('<p class="empty" style="margin-top:12px">Every row is still "drafted". Edit the status column '
                        "(applied / interview / offer / rejected) to fill in this funnel.</p>")
    else:
        body.append('<p class="empty">No applications have been logged yet.</p>')
    body.append("</div>")
    body.append("<footer>Built by resume-tailor from data/postings, data/applications.csv, and skills_status.yaml. "
                "A skill not listed in skills_status.yaml counts as missing.</footer></div>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Application insights</title><style>{CSS}</style></head>"
            f'<body><div class="viz-root">{"".join(body)}</div><div id="tip" role="status"></div>'
            f"<script>{JS}</script></body></html>")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate(status: SkillsStatus, master: MasterResume | None = None, *, today: date | None = None,
             quiet: bool = False) -> tuple[Path, Path]:
    """Recompute everything and write output/insights.html and output/insights.md."""
    ins = compute(analyze.load_postings(), report.read_applications(), status, master, today)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path, md_path = config.OUTPUT_DIR / "insights.html", config.OUTPUT_DIR / "insights.md"
    html_path.write_text(render_html(ins), encoding="utf-8")
    md_path.write_text(render_markdown(ins), encoding="utf-8")
    (config.OUTPUT_DIR / "insights.json").write_text(json.dumps(asdict(ins), indent=2), encoding="utf-8")
    if not quiet:
        ui.success(f"insights: {ins.postings} posting(s), {ins.applications} application(s)")
        for number, unlock in enumerate(ins.top_unlocks(3), 1):
            ui.info(f"  {number}. {unlock.name} ({unlock.status}): only missing skill in {unlock.sole} posting(s), "
                    f"one of two in {unlock.within_two - unlock.sole} more")
        ui.info(f"  {html_path}\n  {md_path}")
    return html_path, md_path
