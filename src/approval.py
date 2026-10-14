"""Human approval checkpoint.

Before anything is rendered, every flagged change is shown side by side
(original fact / bullet -> rewritten) with the reason it was flagged, and I
choose: [A]pprove / [E]dit / [R]evert.

When nobody is at the keyboard (batch mode, pipes), nothing can be approved, so
every pending change reverts to its verified original wording.
"""

from __future__ import annotations

from src.models import Approval, TailoredResume


def revert_pending(resume: TailoredResume, reason: str = "not approved (no one at the keyboard)") -> int:
    """Revert every bullet still waiting for approval to its verified wording."""
    count = 0
    for _entry, _index, bullet in resume.pending():
        bullet.final_text = bullet.original_text
        bullet.approval = Approval.reverted
        bullet.history.append(f"reverted to the verified wording: {reason}")
        count += 1
    return count
