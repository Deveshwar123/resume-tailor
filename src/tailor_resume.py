"""First tailored draft (Sonnet), immediately validated.

Claude selects, reorders, rewords, and emphasizes facts from the master resume.
It never gets the last word: `validate.enforce()` checks every bullet against
its cited facts, sends hard violations back for one retry (`make_retry_fn`),
and reverts or drops whatever still fails.

The system prompt built here (rules + fact bank + job + company brief) is
shared by the tailor, reword, revise, fix, and tighten calls. It is identical
across a run, so it is cached and later calls read it at a fraction of the cost.

Why tailoring is pushed so hard toward rewording: in live runs the tailor
returned the approved bullets word for word (10 of 10, 9 of 9, 6 of 7), which
made "tailoring" only a choice and order of bullets. The approved bullets are
the candidate's general-purpose wording, so the normal-mode rules now ask for a
rewrite per bullet and name the words the checker treats as new claims. If the
draft still comes back mostly verbatim, `reword_verbatim()` asks once more, for
those bullets only, with one candidate and a reason each; a candidate is kept
only if it passes the validator with no hard violation and uses at least as
much of the posting's language as the wording it replaces.
"""

from __future__ import annotations

import json

import config
from src import scoring, textutil, ui, validate
from src.errors import LLMError
from src.models import (
    BulletDraft,
    BulletFix,
    BulletFixBatch,
    BulletFixRequest,
    ChangeType,
    RewordBatch,
    Severity,
    CompanyBrief,
    EntryDraft,
    MasterResume,
    Mode,
    SectionName,
    TailoredResumeDraft,
)
from src.validate import ValidationContext, ValidationOutcome

RULES = """You tailor a student's resume to one job posting. You work only from the verified fact bank below.

Why the rules are strict: this resume goes to real employers, and anything on it that the fact bank does not support is a fabrication the candidate would have to answer for. Every bullet you write is checked by code against the facts it cites. A bullet that fails is sent back once and then replaced with the original wording, so a claim the facts do not support never reaches the page - it only costs the candidate a better-written bullet.

What you may do: select, reorder, reword, and emphasize facts from the fact bank.
What you may not do: add or enlarge an employer, title, date, number, skill, tool, scope, or level of ownership.

Writing each bullet
- A bullet states only what its cited facts support. Cite every fact you draw on in `fact_ids`, and cite only facts that belong to the same experience or project as the bullet.
- Numbers: use a number only if it appears in a cited fact, exactly as written there ("20+" stays "20+"; never "over 20", never rounded up). Never add a number.
- Skills and tools: name one only if a cited fact names it or lists it under `skills`. The job posting and the company research tell you what to emphasize; they never give you new skills to claim. If the company uses a tool the fact bank does not mention, it does not appear.
- Ownership: each fact has an ownership level (member < contributor < lead < founder/owner). Keep the verbs at or below that level. A "contributor" fact that says "Assisted a team in developing" cannot become "Led development" or "Built"; keep the helper wording ("assisted", "helped", "supported") when the fact has it.
- Scope: do not add words that enlarge the work ("company-wide", "end-to-end", "large-scale", "production", "multiple", "all") unless the cited fact uses them.
- Mirror the posting's language only where it is true of the fact. Prefer strong, accurate verbs over inflated ones.
- Style: start with a verb, no first person, one sentence, at most about 200 characters, ending with a period. Use past tense for roles that have ended and present tense for current ones.
- Words the checker treats as new claims unless a cited fact already uses them (a bullet that adds one goes to the candidate for approval, and reverts to the original when no one can approve it):
  - scope: {scope_words}
  - technical complexity: {complexity_words}
  - results (any form): {result_words}

Choosing content
- Pick the experiences, projects, and facts that best fit this job type: technical work for software roles, analytical and business work for finance roles, methodology and rigor for research roles, data handling and analysis for data roles.
- The resume must fit on ONE page: select at most {target_bullets} bullets in total and at most {max_per_entry} per entry. Leave out content that does not help for this job.
- Company research is graded by source: "heavy" items (the company's own pages and the posting) should shape emphasis; "light" items (news) may nudge it. Research changes what you emphasize, never what you claim.

Filling the output
- `source_id` is the id of the experience or project. Copy `org`, `title`, `start`, and `end` exactly as they appear in the fact bank; code uses the fact bank's values and treats any difference as an alteration.
- `source_bullet_id`: the id of the approved bullet you kept or adapted, or null when you wrote the bullet from facts.
- `change_type`: unchanged (identical to an approved bullet or a fact), minor_wording (a few words changed), reworded (substantially rewritten), combined (merges facts that no single approved bullet covers).
- `rationale`: one short sentence on why this bullet helps for this job.
- `skills`: only items from "allowed_skills", most relevant to this job first.
- `honors`, `coursework`: items copied exactly from the education entry that are worth the space for this job (either may be empty).
- `award_ids`: ids of awards worth including (may be empty).
- `section_order`: the order of the sections for this job (education, experience, projects, leadership, skills, awards).
- Layout is fixed by code: within a section, entries are always listed newest first, whatever order you return. Give the most relevant work its weight through which entries you include, how many bullets each gets, the order of its bullets, the wording, and the order of sections. Leave out entries that add little for this job (for example, a membership whose only fact is attending meetings) unless the page would otherwise be thin.
{mode_rules}"""

NORMAL_RULES = """
Mode: normal. Tailoring the wording is the main job.
- The approved bullets are the candidate's general-purpose wording. They show what is true; they are not written for this posting. A draft that repeats them word for word is not tailored.
- For every bullet, start from its cited facts (and the approved bullet, if any) and rewrite it for this posting: lead with the part this role cares about most (its `responsibilities`, `key_phrases`, and `required_skills`), use the posting's own words for what the facts already show, and cut what this role does not need. If a word would add a claim, drop that word, not the rewrite.
- Keep an approved bullet or a fact word for word only when it already leads with what this role values, in the posting's language, and say so in `rationale`. Expect to rewrite most bullets.
- `rationale`: name the posting's need the bullet now speaks to."""

STRICT_RULES = """
Mode: STRICT. The candidate wants their own approved wording.
- Where an experience has approved bullets, use them word for word: `rewritten_text` identical to the approved bullet, `source_bullet_id` set, `change_type` "unchanged".
- For a fact with no approved bullet, use the fact text word for word, one fact per bullet.
- You may propose a rewrite of a bullet only when it would clearly help for this job. Put the proposal in `rewritten_text`, keep `source_bullet_id` pointing at the original, and explain it in `rationale`. The original stays on the resume unless the candidate approves your proposal, so propose sparingly."""

TAILOR_USER = "Write the tailored resume draft for this job."

REWORD_USER = """These bullets are still word for word the candidate's general-purpose wording. Rewrite each one for this posting.

- Lead with what this role cares about most (`responsibilities`, `key_phrases`, `required_skills` in <job>), and use the posting's own words for what the cited facts already show. List those words in `posting_terms`.
- State only what the cited facts support: the same numbers written the same way, only skills and tools the facts name, ownership verbs at or below each fact's level, and keep helper wording such as "assisted" or "helped" when a fact has it.
- Avoid the words listed under "Words the checker treats as new claims" unless a cited fact uses them.
- One sentence, starting with a verb, ending with a period, at most about 200 characters.
- `reason`: one short sentence on what the rewrite does for this posting.
- Keep each `key`. If no truthful rewrite would serve this posting better, return the current text and say why in `reason`.

{items}"""

FIX_USER = """These bullets failed validation against the fact bank. Rewrite each one so that it states only what its cited facts support and no longer has the problem listed. If a problem cannot be fixed by rewording, return the fact's own wording. Keep each `key` unchanged and cite only facts of that bullet's own entry.

{requests}"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def allowed_skills(ctx: ValidationContext) -> list[str]:
    """Skills that may appear in the skills section: master list AND verified."""
    return [s for s in ctx.master.skills.all() if ctx.status.is_verified(s)]


def fact_bank(ctx: ValidationContext) -> dict:
    """The master resume as compact, deterministic JSON for the prompt."""
    master: MasterResume = ctx.master

    def entry_block(entry) -> dict:
        return {
            "source_id": entry.id,
            "org": entry.display_org,
            "title": entry.display_title,
            "start": entry.dates.start,
            "end": entry.dates.end,
            "type": getattr(entry, "type", "project"),
            "tags": entry.tags,
            "facts": [
                {
                    "id": fact.id,
                    "ownership": master.effective_ownership(entry, fact).value,
                    "kind": fact.kind.value,
                    "text": fact.text,
                    "metrics": fact.metrics,
                    "skills": fact.skills,
                    **({"new": True} if fact.id in ctx.new_fact_ids else {}),
                }
                for fact in entry.facts
            ],
            "approved_bullets": [
                {"id": b.id, "text": b.text, "fact_ids": b.fact_ids} for b in entry.bullets if b.approved
            ],
        }

    return {
        "education": [
            {"school": e.school, "degree": e.degree, "major": e.major, "grad": e.grad,
             "honors": e.honors, "coursework": e.coursework}
            for e in master.education
        ],
        "allowed_skills": allowed_skills(ctx),
        "awards": [{"id": a.id, "text": a.text} for a in master.awards],
        "experiences": [entry_block(e) for e in master.experience],
        "projects": [entry_block(p) for p in master.projects],
    }


def job_block(ctx: ValidationContext) -> dict:
    analysis = ctx.analysis
    if analysis is None:
        return {}
    return {
        "company": analysis.company,
        "role": analysis.role,
        "job_type": analysis.job_type.value,
        "required_skills": [{"name": s.name, "weight": s.weight} for s in analysis.required_skills],
        "preferred_skills": [{"name": s.name, "weight": s.weight} for s in analysis.preferred_skills],
        "tools": analysis.tools,
        "key_phrases": analysis.key_phrases,
        "responsibilities": analysis.responsibilities,
        "keywords": analysis.keywords,
    }


def brief_block(brief: CompanyBrief | None) -> dict:
    """Company research the tailor may use for emphasis. Unverified items are left out."""
    if brief is None or brief.is_empty():
        return {}
    out: dict[str, list[dict]] = {}
    for name, items in brief.items(tuple(config.TAILOR_TIER_WEIGHTS)).items():
        out[name] = [{"text": i.text, "weight": config.TAILOR_TIER_WEIGHTS[i.tier.value]} for i in items]
    return out


def build_system(ctx: ValidationContext, brief: CompanyBrief | None = None) -> str:
    """The shared, cacheable system prompt for tailor / revise / fix / tighten."""
    rules = RULES.format(
        target_bullets=config.TARGET_BULLETS,
        max_per_entry=config.MAX_BULLETS_PER_ENTRY,
        mode_rules=STRICT_RULES if ctx.mode == Mode.strict else NORMAL_RULES,
        scope_words=", ".join(config.SCOPE_WORDS),
        complexity_words=", ".join(config.COMPLEXITY_WORDS),
        result_words=", ".join(config.RESULT_WORDS + config.RESULT_PHRASES),
    )
    parts = [
        rules,
        "<fact_bank>\n" + json.dumps(fact_bank(ctx), indent=1, ensure_ascii=False) + "\n</fact_bank>",
        "<job>\n" + json.dumps(job_block(ctx), indent=1, ensure_ascii=False) + "\n</job>",
    ]
    research = brief_block(brief)
    if research:
        parts.append("<company_research>\n" + json.dumps(research, indent=1, ensure_ascii=False) + "\n</company_research>")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Retry (the single re-ask after a hard violation)
# ---------------------------------------------------------------------------


def make_retry_fn(llm, system: str, ctx: ValidationContext) -> validate.RetryFn:
    """The callback validate.enforce() uses for its one retry per failing bullet."""

    def retry(requests: list[BulletFixRequest]) -> list[BulletFix]:
        payload = []
        for request in requests:
            facts = [ctx.facts[fid][1] for fid in request.fact_ids if fid in ctx.facts]
            payload.append({
                "key": request.key,
                "entry": request.entry_id,
                "failed_text": request.text,
                "problems": request.violations,
                "cited_facts": [{"id": f.id, "text": f.text, "metrics": f.metrics, "skills": f.skills} for f in facts],
            })
        try:
            batch = llm.parse(
                purpose="fix", model=config.WRITE_MODEL, system=system,
                user=FIX_USER.format(requests=json.dumps(payload, indent=1, ensure_ascii=False)),
                schema=BulletFixBatch,
            )
        except LLMError as exc:
            ui.warn(f"Validation retry could not run ({exc.message}); reverting those bullets to verified wording.")
            return []
        return batch.fixes

    return retry


# ---------------------------------------------------------------------------
# Second pass: bullets that came back word for word
# ---------------------------------------------------------------------------


def verbatim_share(outcome: ValidationOutcome) -> tuple[int, int]:
    """(bullets that are word for word an approved bullet or a fact, all bullets)."""
    bullets = [b for _e, _i, b in outcome.resume.bullets()]
    return sum(1 for b in bullets if b.change_type == ChangeType.unchanged), len(bullets)


def reword_verbatim(llm, system: str, ctx: ValidationContext, outcome: ValidationOutcome) -> int:
    """Rewrite the verbatim bullets once more for this posting. Returns how many rewrites were kept.

    Runs only in normal mode and only when more than REWORD_MIN_VERBATIM_SHARE
    of the bullets are verbatim. Each candidate is judged in code before it may
    replace the verified wording:
      * `validate.check_bullet` must find no hard violation (a soft flag sends the
        rewrite to approval, exactly like a rewrite from the first call);
      * it must use at least as much of the posting's language
        (`scoring.posting_overlap`) as the wording it replaces.
    Anything else keeps the verified wording, so this pass can only add
    rewrites that the validator accepts. Mutates `outcome` in place.
    """
    if ctx.mode != Mode.normal:
        return 0
    verbatim, total = verbatim_share(outcome)
    if not total or verbatim / total <= config.REWORD_MIN_VERBATIM_SHARE:
        return 0
    targets = {f"{entry.source_id}#{index}": (entry, index, bullet) for entry, index, bullet in outcome.resume.bullets()
               if bullet.change_type == ChangeType.unchanged}
    payload = []
    for key, (entry, _index, bullet) in targets.items():
        facts = [ctx.facts[fid][1] for fid in bullet.fact_ids if fid in ctx.facts]
        payload.append({"key": key, "entry": entry.source_id, "current_text": bullet.text,
                        "cited_facts": [{"id": f.id, "text": f.text, "metrics": f.metrics, "skills": f.skills,
                                         "ownership": ctx.master.effective_ownership(ctx.master.entry(entry.source_id), f).value}
                                        for f in facts]})
    try:
        batch = llm.parse(purpose="reword", model=config.WRITE_MODEL, system=system,
                          user=REWORD_USER.format(items=json.dumps(payload, indent=1, ensure_ascii=False)),
                          schema=RewordBatch)
    except LLMError as exc:
        ui.warn(f"The rewording pass could not run ({exc.message}); keeping the verified wording.")
        return 0
    kept = 0
    for candidate in batch.rewrites:
        target = targets.get(candidate.key)
        if target is None:
            continue
        entry, index, bullet = target
        new_text = textutil.plain(candidate.rewritten_text)
        if not new_text or textutil.same_text(new_text, bullet.text):
            continue
        if any(textutil.same_text(new_text, other.text) for other in entry.bullets if other is not bullet):
            continue
        source_entry = ctx.master.entry(entry.source_id)
        facts = [ctx.facts[fid][1] for fid in bullet.fact_ids if fid in ctx.facts]
        source = ctx.bullets[bullet.source_bullet_id][1] if bullet.source_bullet_id in ctx.bullets else None
        if source_entry is None or not facts:
            continue
        found = validate.check_bullet(new_text, validate.gather_evidence(source_entry, facts, source), ctx,
                                      entry.source_id, index)
        hard = [v for v in found if v.severity == Severity.hard]
        if hard:
            for v in hard:
                v.resolution = "reword candidate rejected; the verified wording was kept"
            outcome.violations.extend(hard)
            continue
        if scoring.posting_overlap(ctx.analysis, new_text) < scoring.posting_overlap(ctx.analysis, bullet.text):
            bullet.history.append(f'a reword candidate used less of the posting\'s language and was not kept: "{new_text}"')
            continue
        soft = [v for v in found if v.severity == Severity.soft]
        rebuilt = validate.build_bullet(new_text, facts, source, ChangeType.reworded, candidate.reason, soft, ctx,
                                        [*bullet.history, "reworded for this posting in the second pass"])
        if rebuilt.approval.value == "pending":
            for v in soft:
                v.resolution = "sent to approval"
            outcome.violations.extend(soft)
        entry.bullets[index] = rebuilt
        kept += 1
    return kept


# ---------------------------------------------------------------------------
# Deterministic baseline (no AI)
# ---------------------------------------------------------------------------


def baseline_draft(ctx: ValidationContext, max_bullets: int | None = None) -> TailoredResumeDraft:
    """A draft made only of verified wording, chosen by relevance - no Claude.

    Used when Claude returns nothing usable, and by the offline demo.
    """
    budget = max_bullets or config.TARGET_BULLETS
    experiences: list[EntryDraft] = []
    projects: list[EntryDraft] = []
    for entry in ctx.relevance.rank_entries(list(ctx.master.entries())):
        if budget <= 0:
            break
        approved = sorted((b for b in entry.bullets if b.approved), key=lambda b: -ctx.relevance.score_text(b.text))
        if approved:
            chosen = [BulletDraft(rewritten_text=b.text, fact_ids=list(b.fact_ids), source_bullet_id=b.id,
                                  change_type=ChangeType.unchanged, rationale="approved wording")
                      for b in approved[: min(config.MAX_BULLETS_PER_ENTRY, budget)]]
        else:
            chosen = [BulletDraft(rewritten_text=f.text, fact_ids=[f.id], source_bullet_id=None,
                                  change_type=ChangeType.unchanged, rationale="fact text")
                      for f in ctx.relevance.rank_facts(entry.facts)[: min(config.MAX_BULLETS_PER_ENTRY, budget)]]
        if not chosen:
            continue
        budget -= len(chosen)
        draft = EntryDraft(source_id=entry.id, org=entry.display_org, title=entry.display_title,
                           start=entry.dates.start, end=entry.dates.end, bullets=chosen)
        (experiences if entry.kind == "experience" else projects).append(draft)
    skills = sorted(allowed_skills(ctx), key=lambda s: -ctx.relevance.score_text(s))
    education = ctx.master.education[0] if ctx.master.education else None
    return TailoredResumeDraft(
        section_order=[SectionName(s) for s in config.DEFAULT_SECTION_ORDER if s != "certifications"],
        experiences=experiences, projects=projects, skills=skills,
        honors=list(education.honors) if education else [],
        coursework=list(education.coursework) if education else [],
        award_ids=[],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def tailor(llm, ctx: ValidationContext, brief: CompanyBrief | None = None) -> tuple[ValidationOutcome, str]:
    """Ask Claude for the first draft and validate it.

    Returns the validation outcome and the system prompt (reused by later calls).
    """
    system = build_system(ctx, brief)
    draft = llm.parse(purpose="tailor", model=config.WRITE_MODEL, system=system,
                      user=TAILOR_USER, schema=TailoredResumeDraft)
    if not draft.experiences and not draft.projects:
        ui.warn("Claude selected no experiences; starting from the verified baseline instead.")
        draft = baseline_draft(ctx)
    outcome = validate.enforce(draft, ctx, retry_fn=make_retry_fn(llm, system, ctx))
    outcome.reworded = reword_verbatim(llm, system, ctx, outcome)
    return outcome, system


def summarize(outcome: ValidationOutcome) -> str:
    """One-line description of a validation pass for the progress display."""
    bullets = sum(len(e.bullets) for e in outcome.resume.entries())
    pending = len(outcome.resume.pending())
    verbatim, _total = verbatim_share(outcome)
    second = f" ({outcome.reworded} in the second pass)" if outcome.reworded else ""
    return (f"{len(list(outcome.resume.entries()))} entries, {bullets} bullets, {bullets - verbatim} reworded{second}; "
            f"{len(outcome.hard())} hard violation(s) handled, {pending} bullet(s) need approval")
