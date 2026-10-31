"""Human approval checkpoint.

Before anything is rendered, every flagged change is shown side by side
(original fact / bullet -> rewritten) with the reason it was flagged, and I
choose: [A]pprove / [E]dit / [R]evert.

What always needs approval (the flags come from validate.py):
  * changes to ownership or responsibility wording
  * changes to technical complexity
  * anything touching metrics or results
  * changes to project scope
  * bullets built from new interview facts
  * any other soft flag from validate.py
  * in --strict mode, every rewrite

Minor wording changes are auto-approved (and still listed in changes.md).
Approved rewrites can be saved back to master_resume.yaml as `approved: true`
bullets so future runs need fewer approvals.

When nobody is at the keyboard (batch mode, pipes), nothing can be approved, so
every pending change reverts to its verified original wording.

--auto (an app calling the tool) is the one exception, by design: a rewrite
whose only reason for review is that it is a substantial rewording (the
"change" flag) raised no validator flag at all, so it is used and marked
`accepted`. Anything a validator check flagged still reverts. `settle()` applies
these rules inside the revision loop, so the judge scores exactly the text that
will be rendered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rich.table import Table

from src import store, textutil, ui, validate
from src.models import Approval, Bullet, MasterResume, Severity, TailoredBullet, TailoredEntry, TailoredResume
from src.validate import ValidationContext


@dataclass
class Decision:
    """One approval decision, kept for changes.md."""

    entry_id: str
    original: str
    proposed: str
    final: str
    action: str                       # approved | edited | reverted | override | accepted (--auto)
    reasons: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    automatic: bool = False           # True when nobody was asked (batch mode)
    saved_as: str | None = None       # id of the approved bullet written to master_resume.yaml


REVIEW_ONLY_FLAGS = {"change"}        # "substantially reworded": a reason to show a person, not a validator finding
_REVERT_NOTE = "reverted to the verified wording: "


def revert_pending(resume: TailoredResume, reason: str = "not approved (no one at the keyboard)") -> list[Decision]:
    """Revert every bullet still waiting for approval to its verified wording."""
    decisions: list[Decision] = []
    for entry, _index, bullet in resume.pending():
        decisions.append(Decision(
            entry_id=entry.source_id, original=bullet.original_text, proposed=bullet.rewritten_text,
            final=bullet.original_text, action="reverted", reasons=[f.message for f in bullet.flags],
            fact_ids=list(bullet.fact_ids), automatic=True,
        ))
        bullet.final_text = bullet.original_text
        bullet.approval = Approval.reverted
        bullet.history.append(_REVERT_NOTE + reason)
    return decisions


def settle(resume: TailoredResume, *, accept_clean: bool, reason: str) -> None:
    """Resolve every pending bullet without a person (batch and --auto).

    `accept_clean` (--auto): a rewrite whose flags are only REVIEW_ONLY_FLAGS is
    used. Every other pending bullet, and every pending bullet in batch mode,
    reverts to its verified wording. Strict-mode proposals carry a "strict" flag
    and so always revert.
    """
    for _entry, _index, bullet in resume.pending():
        if accept_clean and {flag.code for flag in bullet.flags} <= REVIEW_ONLY_FLAGS:
            bullet.final_text = bullet.rewritten_text
            bullet.approval = Approval.accepted
            bullet.history.append("used automatically (--auto): the rewrite passed every validator check")
        else:
            bullet.final_text = bullet.original_text
            bullet.approval = Approval.reverted
            bullet.history.append(_REVERT_NOTE + reason)


def settled_decisions(resume: TailoredResume) -> list[Decision]:
    """The Decision records (for changes.md) of bullets that `settle()` resolved."""
    decisions: list[Decision] = []
    for entry, _index, bullet in resume.bullets():
        reverted = bullet.approval == Approval.reverted and any(n.startswith(_REVERT_NOTE) for n in bullet.history)
        if bullet.approval != Approval.accepted and not reverted:
            continue
        decisions.append(Decision(
            entry_id=entry.source_id, original=bullet.original_text, proposed=bullet.rewritten_text,
            final=bullet.final_text, action="accepted" if bullet.approval == Approval.accepted else "reverted",
            reasons=[f.message for f in bullet.flags], fact_ids=list(bullet.fact_ids), automatic=True,
        ))
    return decisions


def _show(entry: TailoredEntry, bullet: TailoredBullet, master: MasterResume, position: str) -> None:
    source = master.entry(entry.source_id)
    where = f"{source.display_org} - {source.display_title}" if source and source.display_title else (
        source.display_org if source else entry.source_id)
    original_label = {"approved_bullet": "Original (my approved bullet)", "draft_bullet": "Original (my draft bullet)",
                      "fact": "Original (fact)", "facts": "Original (facts)"}.get(bullet.original_kind, "Original")
    table = Table(title=f"{position}  {where}", show_lines=True, title_justify="left", expand=True)
    table.add_column(original_label, ratio=1, overflow="fold")
    table.add_column("Rewritten", ratio=1, overflow="fold")
    table.add_row(bullet.original_text, bullet.rewritten_text)
    ui.console.print(table)
    for flag in bullet.flags:
        ui.console.print(f"   [yellow]flagged:[/yellow] {flag.message}")
    if bullet.rationale:
        ui.console.print(f"   [dim]why Claude changed it: {bullet.rationale}[/dim]")
    ui.console.print(f"   [dim]cited facts: {', '.join(bullet.fact_ids)}[/dim]")


def _edit(entry: TailoredEntry, bullet: TailoredBullet, ctx: ValidationContext) -> tuple[str, str] | None:
    """Let me type my own wording. Returns (text, action) or None to revert."""
    while True:
        text = textutil.plain(ui.ask("   Your wording (empty to revert): "))
        if not text:
            return None
        hard = [v for v in validate.recheck(text, entry.source_id, bullet.fact_ids, bullet.source_bullet_id, ctx)
                if v.severity == Severity.hard]
        if not hard:
            return text, "edited"
        ui.console.print("   [red]This wording is not supported by the cited facts:[/red]")
        for violation in hard:
            ui.console.print(f"     - {violation.message}")
        ui.console.print("   [dim]If it is true, add a fact for it to master_resume.yaml instead.[/dim]")
        choice = ui.ask_choice("   ", {"e": "edit again", "k": "keep anyway", "r": "revert"}, default="r")
        if choice == "k":
            return text, "override"
        if choice == "r":
            return None


def review(resume: TailoredResume, ctx: ValidationContext, *, batch: bool = False) -> list[Decision]:
    """Resolve every pending bullet. Interactive unless `batch` or stdin is not a terminal."""
    pending = resume.pending()
    if not pending:
        return []
    if batch or not ui.is_interactive():
        return revert_pending(resume, "batch mode" if batch else "not approved (no one at the keyboard)")

    ui.console.print(f"\n[bold]{len(pending)} change(s) need your approval[/bold] "
                     "(nothing is rendered until you decide; Enter = revert)")
    decisions: list[Decision] = []
    for number, (entry, _index, bullet) in enumerate(pending, 1):
        _show(entry, bullet, ctx.master, f"[{number}/{len(pending)}]")
        reasons = [f.message for f in bullet.flags]
        choice = ui.ask_choice("  ", {"a": "approve", "e": "edit", "r": "revert"}, default="r")
        action, final = "reverted", bullet.original_text
        if choice == "a":
            action, final = "approved", bullet.rewritten_text
        elif choice == "e":
            edited = _edit(entry, bullet, ctx)
            if edited is not None:
                final, action = edited
        decisions.append(Decision(entry_id=entry.source_id, original=bullet.original_text,
                                  proposed=bullet.rewritten_text, final=final, action=action,
                                  reasons=reasons, fact_ids=list(bullet.fact_ids)))
        bullet.final_text = final
        bullet.approval = Approval(action)
        if action in ("edited", "override"):
            bullet.rewritten_text = final
        bullet.history.append(f"approval: {action}")
    return decisions


# ---------------------------------------------------------------------------
# Saving approved rewrites back to the master resume
# ---------------------------------------------------------------------------


def next_bullet_id(entry) -> str:
    numbers = [int(m.group(1)) for b in entry.bullets if (m := re.search(r"_b(\d+)$", b.id))]
    return f"{entry.id}_b{(max(numbers, default=0) + 1):02d}"


def savable(decisions: list[Decision]) -> list[Decision]:
    """Approved or edited rewrites. Overrides are never saved: they are not backed by a fact."""
    return [d for d in decisions if d.action in ("approved", "edited") and not d.automatic
            and not textutil.same_text(d.final, d.original)]


def save_approved(master: MasterResume, decisions: list[Decision], path: Path | None = None) -> list[str]:
    """Write approved rewrites to master_resume.yaml as `approved: true` bullets."""
    saved: list[str] = []
    for decision in savable(decisions):
        entry = master.entry(decision.entry_id)
        if entry is None or any(b.approved and textutil.same_text(b.text, decision.final) for b in entry.bullets):
            continue
        fact_ids = [f for f in decision.fact_ids if any(fact.id == f for fact in entry.facts)]
        if not fact_ids:
            continue
        bullet = Bullet(id=next_bullet_id(entry), text=decision.final, fact_ids=fact_ids, approved=True)
        entry.bullets.append(bullet)
        decision.saved_as = bullet.id
        saved.append(bullet.id)
    if saved:
        store.save_master_resume(master, path)
    return saved


def offer_to_save(master: MasterResume, decisions: list[Decision], path: Path | None = None) -> list[str]:
    candidates = savable(decisions)
    if not candidates or not ui.is_interactive():
        return []
    if not ui.ask_yes_no(f"Save {len(candidates)} approved rewrite(s) to master_resume.yaml as approved bullets "
                         "(future runs will need fewer approvals)?", default=False):
        return []
    saved = save_approved(master, decisions, path)
    if saved:
        ui.success(f"saved {len(saved)} approved bullet(s); a backup of master_resume.yaml is in data/backups/")
    return saved


def summarize(decisions: list[Decision]) -> str:
    if not decisions:
        return "no changes needed approval"
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.action] = counts.get(decision.action, 0) + 1
    auto = " (automatically - no one at the keyboard)" if all(d.automatic for d in decisions) else ""
    return ", ".join(f"{n} {action}" for action, n in counts.items()) + auto
