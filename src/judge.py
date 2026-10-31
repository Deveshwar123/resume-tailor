"""Independent rubric judge (Haiku).

The judge sees only three things: the job posting, the tier-1/2 (official and
posting) parts of the company brief, and the current resume text. It never sees
earlier scores, earlier drafts, the fact bank, or the tailor's reasoning, so
each round is scored fresh.

It returns four category scores (1-10, one decimal allowed, each with a
one-line justification), integer scores for every numbered bullet, and a list
of fixes. It never returns a total: scoring.py computes Resume Quality.

Why per-bullet scores: a revision usually rewords two or three bullets. A
whole-resume score in whole numbers did not move for that, so a real
improvement looked like a tie and the loop stopped. scoring.quality() blends
the average of the bullet scores into each category, so every bullet that got
better moves the total a little.
"""

from __future__ import annotations

import config
from src.models import BulletScore, CompanyBrief, Fix, FixType, Judgment

MAX_FIXES = 12

JUDGE_SYSTEM = """You are a strict technical recruiter screening resumes for the role below. You have read thousands of intern resumes and you do not inflate scores.

You will be given one resume. Score how well it is WRITTEN and how well its content is CHOSEN for this role, then list concrete fixes.

Judge writing quality separately from qualifications. This candidate may simply not have a skill or experience the posting asks for. That is a qualification gap: report it as a REAL_GAP fix, and do not let it lower Impact or Clarity. Score those two on the bullets that are on the page, as writing.

Categories (1-10 each, with a one-line justification; use one decimal place, such as 7.5, when the resume sits between two anchors)
- impact: do the bullets lead with strong, accurate verbs and show results, scale, or outcome wherever the content gives them room to? Vague duty lists score low; specific, outcome-led bullets score high.
- relevance: is the content on the page the right content for this role, and is the most relevant material the most prominent (order of sections, entries, and bullets)? Space spent on material this role does not care about lowers the score.
- clarity: is every bullet clear, concise, and specific? Consistent tense, no filler, no jargon without substance, no bullet that needs rereading.
- language: does the resume use the posting's own terminology where it naturally and truthfully fits, without keyword stuffing?

Score anchors: 9-10 means you would send it on with no edits. 7-8 is solid with a few fixable weaknesses. 5-6 has several weak or off-target bullets. 4 or below is mostly weak. Reserve 9 and 10 for resumes that truly earn them.

Bullet scores
- The bullets are numbered [1], [2], ... For EVERY numbered bullet, return `bullet` (its number) and integer 1-10 scores for impact, relevance, clarity, and language, judged on that bullet alone with the same anchors. Relevance here means how much this bullet matters for this role.
- Score each bullet on its own wording, not on its neighbors.

Fixed layout rules (do not ask to change these; they are applied by code for applicant-tracking systems):
- Within each section, entries are always listed newest first. Never ask to move an entry above or below another entry in the same section. To give an entry more or less weight, ask to change its bullets, cut it, or change the order of whole sections.
- The page is one page; section headings and the date format are fixed.

Fixes (most valuable first)
- REWRITE: an improvement that needs no new information about the candidate - rewording a bullet, cutting filler, reordering sections, entries, or bullets, leading with a stronger verb that the bullet already supports, replacing low-relevance content with more relevant content already implied on the page. Never ask for a number, tool, or responsibility that is not on the page as a REWRITE.
- ASK_USER: the candidate might have relevant experience or detail that is not on the resume (a metric, a tool they used, a result). Write `detail` as a direct question to the candidate.
- REAL_GAP: a requirement of the posting that the resume shows no evidence of meeting.
For each fix give `target` (the section, entry, or the first words of the bullet it concerns), `detail` (exactly what to change, or the question, or the gap), and `expected_gain` (your estimate of how many points, 0.1 to 1.0, the overall resume would gain on a 10-point scale).

<posting>
{posting}
</posting>
{brief}"""

BRIEF_BLOCK = """
<company_context note="from the company's own pages and the posting">
{items}
</company_context>"""


def brief_text(brief: CompanyBrief | None) -> str:
    """Only the official and posting tiers reach the judge."""
    if brief is None:
        return ""
    lines = [f"- [{name}] {item.text}" for name, items in brief.items(config.JUDGE_TIERS).items() for item in items]
    return BRIEF_BLOCK.format(items="\n".join(lines)) if lines else ""


def build_system(posting_text: str, brief: CompanyBrief | None) -> str:
    """Stable for the whole run (posting + brief), so every round reads it from cache."""
    return JUDGE_SYSTEM.format(posting=posting_text.strip(), brief=brief_text(brief))


def clean(judgment: Judgment, bullet_count: int | None = None) -> Judgment:
    """Clamp scores to 1-10 and tidy the fix list. No totals are computed here.

    Category scores keep one decimal. Bullet scores are clamped to whole numbers
    1-10; a bullet number that is not on the page (or repeats) is dropped.
    """
    scores = judgment.category_scores
    for name in config.JUDGED_CATEGORIES:
        category = getattr(scores, name)
        category.score = round(max(1.0, min(10.0, float(category.score))), 1)
        category.justification = category.justification.strip()
    kept: dict[int, BulletScore] = {}
    for item in judgment.bullet_scores:
        if item.bullet < 1 or (bullet_count is not None and item.bullet > bullet_count) or item.bullet in kept:
            continue
        for name in config.JUDGED_CATEGORIES:
            setattr(item, name, int(max(1, min(10, round(float(getattr(item, name)))))))
        kept[item.bullet] = item
    judgment.bullet_scores = [kept[n] for n in sorted(kept)]
    fixes: list[Fix] = []
    for fix in judgment.fixes:
        if not fix.detail.strip():
            continue
        fix.expected_gain = round(max(0.0, min(3.0, float(fix.expected_gain))), 2)
        fix.target, fix.detail = fix.target.strip(), fix.detail.strip()
        fixes.append(fix)
    judgment.fixes = sorted(fixes, key=lambda f: -f.expected_gain)[:MAX_FIXES]
    return judgment


def judge_resume(llm, posting_text: str, brief: CompanyBrief | None, resume_text: str,
                 bullet_count: int | None = None) -> Judgment:
    """Score one resume. The user message is the resume text and nothing else.

    `resume_text` should number its bullets (`ResumeView.plain_text(number_bullets=True)`);
    `bullet_count` lets clean() drop bullet scores for numbers that are not on the page.
    """
    judgment = llm.parse(
        purpose="judge",
        model=config.FAST_MODEL,
        system=build_system(posting_text, brief),
        user=f"<resume>\n{resume_text.strip()}\n</resume>",
        schema=Judgment,
    )
    return clean(judgment, bullet_count)


def judge_view(llm, posting_text: str, brief: CompanyBrief | None, view) -> tuple[Judgment, list[str]]:
    """Judge a rendered view (contact redacted, bullets numbered). Returns the judgment and the bullet texts."""
    bullets = view.bullet_texts()
    text = view.plain_text(redact_contact=True, number_bullets=True)
    return judge_resume(llm, posting_text, brief, text, len(bullets)), bullets


def summarize(judgment: Judgment) -> str:
    counts = {t: len(judgment.fixes_of(t)) for t in FixType}
    return (f"{counts[FixType.REWRITE]} rewrite fix(es), {counts[FixType.ASK_USER]} question(s), "
            f"{counts[FixType.REAL_GAP]} real gap(s)")
