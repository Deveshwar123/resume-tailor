"""Interview: the judge's ASK_USER questions -> new verified facts.

When the resume is below target and the judge thinks I may have relevant
experience that is not in the master resume, it asks. Questions are put to me
one at a time and any of them can be skipped. Each answer is turned into a
structured fact (kind, text, ownership, metrics, skills), which I then
Approve / Edit / Reject. Nothing is written without my OK.

Safeguards:
  * the proposed fact is checked against my own answer in code: a number or an
    ownership verb that I did not say is called out before I approve
  * master_resume.yaml is backed up with a timestamp before any write
  * new facts are marked `source: interview` with the date, and every bullet
    built from them goes through the approval checkpoint
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import config
from src import store, textutil, ui
from src.errors import LLMError
from src.models import Entry, Fact, Fix, FixType, MasterResume, Ownership, ProposedFact, SkillsStatus, SkillStatus

MAX_QUESTIONS = 5

INTERVIEW_SYSTEM = """You turn a student's answer to a recruiter's question into one structured, verifiable fact for their resume fact bank.

The fact becomes evidence that resume bullets are checked against, so it must contain only what the student actually said.

- text: one sentence in resume style (starts with a verb, no "I"), stating only what the answer states. Do not add outcomes, numbers, tools, scale, or responsibility that the answer does not mention. If the answer is vague, the fact stays vague.
- kind: responsibility, achievement, metric, tool_use, or scope.
- ownership: the level the answer supports - member (took part, attended), contributor (did part of the work, helped, worked on a team), lead (led or managed people or the effort), founder or owner (started it or did it entirely alone). When the answer does not make this clear, choose the lower level.
- metrics: every number in the fact, copied exactly as the student wrote it ("20+", "3 weeks"). Empty when the answer has no numbers.
- skills: tools, languages, or skills the answer explicitly names. Empty when it names none.
- parent_id: the id of the experience or project (from the list given) that this fact belongs to, or null if none of them fits.

If the answer says they have no such experience, or says nothing usable, return an empty text."""

INTERVIEW_USER = """Recruiter's question: {question}

Student's answer: {answer}

Experiences and projects in the fact bank:
{entries}"""


@dataclass
class InterviewResult:
    new_fact_ids: list[str] = field(default_factory=list)
    asked: int = 0
    skipped: int = 0
    rejected: int = 0
    backup: Path | None = None
    log: list[str] = field(default_factory=list)


def questions_from(fixes: list[Fix]) -> list[Fix]:
    return [f for f in fixes if f.type == FixType.ASK_USER][:MAX_QUESTIONS]


def next_fact_id(entry: Entry) -> str:
    numbers = [int(m.group(1)) for f in entry.facts if (m := re.search(r"_f(\d+)$", f.id))]
    return f"{entry.id}_f{(max(numbers, default=0) + 1):02d}"


def answer_warnings(proposal: ProposedFact, answer: str) -> list[str]:
    """Things in the proposed fact that my answer did not say (checked in code)."""
    warnings = []
    for atom in textutil.unsupported_atoms(proposal.text, [answer]):
        warnings.append(f'the number "{atom.raw}" is not in your answer')
    said = textutil.ownership_terms(answer)
    for lemma, tier in textutil.ownership_terms(proposal.text).items():
        if lemma not in said and tier >= config.OWNERSHIP_TIERS["lead"]:
            warnings.append(f'the fact says "{lemma.removeprefix("noun:")}" but your answer did not')
    claimed = config.OWNERSHIP_TIERS[proposal.ownership.value]
    supported = max(said.values(), default=config.OWNERSHIP_TIERS["contributor"])
    if claimed > max(supported, config.OWNERSHIP_TIERS["contributor"]):
        warnings.append(f'ownership "{proposal.ownership.value}" is higher than your answer clearly supports')
    return warnings


def _entries_listing(master: MasterResume) -> str:
    return "\n".join(f"- {e.id}: {e.display_org}" + (f" ({e.display_title})" if e.display_title else "")
                     for e in master.entries())


def propose_fact(llm, master: MasterResume, question: str, answer: str) -> ProposedFact | None:
    proposal = llm.parse(
        purpose="interview", model=config.FAST_MODEL, system=INTERVIEW_SYSTEM,
        user=INTERVIEW_USER.format(question=question, answer=answer, entries=_entries_listing(master)),
        schema=ProposedFact, cache_system=False,
    )
    proposal.text = textutil.plain(proposal.text).rstrip(".")
    if not proposal.text:
        return None
    # A metric must literally be in the fact text (the master resume schema requires it).
    proposal.metrics = [m for m in proposal.metrics if textutil.tight(m) and textutil.tight(m) in textutil.tight(proposal.text)]
    return proposal


def _pick_parent(master: MasterResume, suggested: str | None) -> Entry | None:
    entry = master.entry(suggested) if suggested else None
    if entry is not None:
        label = f"{entry.display_org}" + (f" ({entry.display_title})" if entry.display_title else "")
        if ui.ask_yes_no(f"   File this under {label}?", default=True):
            return entry
    entries = list(master.entries())
    for number, item in enumerate(entries, 1):
        ui.console.print(f"   {number:>2}. {item.display_org}" + (f" ({item.display_title})" if item.display_title else ""))
    answer = ui.ask("   Which one does this belong to? (number, or Enter to reject): ")
    if answer.isdigit() and 1 <= int(answer) <= len(entries):
        return entries[int(answer) - 1]
    return None


def _show(proposal: ProposedFact, warnings: list[str]) -> None:
    ui.console.print(f"   [bold]Proposed fact:[/bold] {proposal.text}")
    ui.console.print(f"   [dim]kind: {proposal.kind.value} | ownership: {proposal.ownership.value} | "
                     f"metrics: {proposal.metrics or 'none'} | skills: {proposal.skills or 'none'}[/dim]")
    for warning in warnings:
        ui.console.print(f"   [yellow]check:[/yellow] {warning}")


def _verify_skills(proposal: ProposedFact, status: SkillsStatus, changed: list[str]) -> None:
    """A fact may only be tagged with verified skills; ask me about any that are not."""
    kept = []
    for skill in proposal.skills:
        if status.is_verified(skill):
            kept.append(skill)
        elif ui.ask_yes_no(f'   "{skill}" is not marked verified in skills_status.yaml. Can you back it up - mark it verified?',
                           default=False):
            existing = next((name for name in status.skills if textutil.canon(name) == textutil.canon(skill)), None)
            status.skills[existing or skill] = SkillStatus.verified
            changed.append(skill)
            kept.append(skill)
    proposal.skills = kept


def run_interview(llm, master: MasterResume, status: SkillsStatus, fixes: list[Fix], *, today: date | None = None,
                  master_path: Path | None = None, status_path: Path | None = None) -> InterviewResult:
    """Ask the ASK_USER questions and add the facts I approve to the master resume."""
    today = today or date.today()
    result = InterviewResult()
    asks = questions_from(fixes)
    if not asks or not ui.is_interactive():
        return result
    ui.console.print(f"\n[bold]{len(asks)} question(s) from the recruiter's review[/bold] "
                     "(answer in a sentence or two; Enter skips a question)")
    added: list[tuple[Entry, Fact]] = []
    verified_skills: list[str] = []
    for number, fix in enumerate(asks, 1):
        result.asked += 1
        ui.console.print(f"\n[bold cyan]Q{number}.[/bold cyan] {fix.detail}"
                         + (f"  [dim](about: {fix.target})[/dim]" if fix.target else ""))
        answer = ui.ask("   > ")
        if not answer:
            result.skipped += 1
            continue
        try:
            proposal = propose_fact(llm, master, fix.detail, answer)
        except LLMError as exc:
            ui.warn(f"Could not structure that answer ({exc.message}); skipping it.")
            result.skipped += 1
            continue
        if proposal is None:
            ui.console.print("   [dim]Nothing to add from that answer.[/dim]")
            result.skipped += 1
            continue
        while True:
            _show(proposal, answer_warnings(proposal, answer))
            choice = ui.ask_choice("  ", {"a": "approve", "e": "edit", "r": "reject"}, default="r")
            if choice != "e":
                break
            edited = textutil.plain(ui.ask("   Fact text: ", proposal.text)).rstrip(".")
            if edited:
                proposal.text = edited
                proposal.metrics = [m for m in proposal.metrics if textutil.tight(m) in textutil.tight(edited)]
            level = ui.ask(f"   Ownership (member/contributor/lead/founder/owner) [{proposal.ownership.value}]: ",
                           proposal.ownership.value).lower()
            if level in Ownership._value2member_map_:
                proposal.ownership = Ownership(level)
        if choice != "a":
            result.rejected += 1
            continue
        parent = _pick_parent(master, proposal.parent_id)
        if parent is None:
            ui.console.print("   [dim]No matching entry - add the experience to master_resume.yaml by hand, then re-run.[/dim]")
            result.rejected += 1
            continue
        _verify_skills(proposal, status, verified_skills)
        fact = Fact(id=next_fact_id(parent), kind=proposal.kind, text=proposal.text, ownership=proposal.ownership,
                    metrics=proposal.metrics, skills=proposal.skills, source="interview", added=today.isoformat())
        parent.facts.append(fact)
        added.append((parent, fact))
        result.new_fact_ids.append(fact.id)
        result.log.append(f"{fact.id} ({parent.display_org}): {fact.text}")
    if added:
        # save_master_resume backs the file up (timestamped) before writing.
        result.backup = store.save_master_resume(master, master_path)
        ui.success(f"added {len(added)} fact(s) to master_resume.yaml"
                   + (f" (backup: {result.backup.name})" if result.backup else ""))
    if verified_skills:
        store.save_skills_status(status, status_path)
        ui.success(f"marked as verified in skills_status.yaml: {', '.join(verified_skills)}")
    return result
