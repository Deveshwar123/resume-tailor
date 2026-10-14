"""Revision (Sonnet): apply the judge's REWRITE fixes, then re-validate.

Only REWRITE fixes are ever passed in - questions for me (ASK_USER) and real
gaps (REAL_GAP) cannot be fixed by rewording. The reviser works from the same
fact bank as the tailor (it may swap in facts and approved bullets that were
not selected before), and its output goes through `validate.enforce()` again,
every round.
"""

from __future__ import annotations

import json

import config
from src import validate
from src.models import Fix, FixType, TailoredResume, TailoredResumeDraft
from src.tailor_resume import make_retry_fn
from src.validate import ValidationContext, ValidationOutcome

REVISE_USER = """A recruiter reviewed the current draft and asked for the changes below. Apply them and return the full revised draft.

- Make each requested change using only the fact bank. You may bring in facts or approved bullets that are not in the current draft, and drop content the recruiter found weak or off-target.
- If a requested change would need a number, skill, tool, scope, or level of ownership that the fact bank does not support, leave that bullet as it is. An honest bullet that scores lower is the right outcome.
- Keep everything the recruiter did not mention as it is, including bullets marked unchanged.
- The one-page limits still apply.

<requested_changes>
{fixes}
</requested_changes>

<current_draft>
{draft}
</current_draft>"""


def rewrite_fixes(fixes: list[Fix]) -> list[Fix]:
    return [f for f in fixes if f.type == FixType.REWRITE]


def revise(llm, system: str, ctx: ValidationContext, current: TailoredResume, fixes: list[Fix]) -> ValidationOutcome:
    """One revision round. `system` is the tailor's cached system prompt."""
    applied = rewrite_fixes(fixes)
    if not applied:
        raise ValueError("revise() needs at least one REWRITE fix")
    fix_lines = "\n".join(f"{i}. [{f.target}] {f.detail}" for i, f in enumerate(applied, 1))
    draft_json = json.dumps(current.to_draft(ctx.master).model_dump(mode="json"), indent=1, ensure_ascii=False)
    draft = llm.parse(
        purpose="revise", model=config.WRITE_MODEL, system=system,
        user=REVISE_USER.format(fixes=fix_lines, draft=draft_json),
        schema=TailoredResumeDraft,
    )
    if not draft.experiences and not draft.projects:
        # A revision that selects nothing is unusable; keep the current draft.
        draft = current.to_draft(ctx.master)
    return validate.enforce(draft, ctx, retry_fn=make_retry_fn(llm, system, ctx))
