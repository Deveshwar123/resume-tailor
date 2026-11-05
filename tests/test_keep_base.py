"""Your resume kept whole (`base_resume`): tailoring rewords it with the posting's keywords and
never cuts it (the owner's rule, 5 Nov 2026: "all I need is reword what I did, add keywords that
are in the job description... you are cutting my resume in half")."""

import pydantic
import pytest

from src import render, tailor_resume, validate
from src.models import MasterResume
from src.relevance import Relevance
from src.validate import ValidationContext
from tests.conftest import bd, ed, make_draft


@pytest.fixture
def keep_ctx(master_data, status, analysis):
    master_data["base_resume"] = ["exp_acme", "proj_tracker"]
    master_data["base_skills"] = ["Python", "Git", "Docker"]          # Docker is only "learning"
    return ValidationContext(master=MasterResume.model_validate(master_data), status=status, analysis=analysis)


def test_every_entry_and_bullet_of_your_resume_stays_and_nothing_else_is_added(keep_ctx):
    draft = make_draft(
        experiences=[ed("exp_acme", [bd("Assisted a 4-person team in developing an application in Python.",
                                         ["exp_acme_f01"], "exp_acme_b01")]),
                     ed("exp_lab", [bd("Annotated 500 images", ["exp_lab_f01"], None, "unchanged")])],
        skills=["SQL", "Python"])
    resume = validate.enforce(draft, keep_ctx).resume
    assert [e.source_id for e in resume.experiences] == ["exp_acme"]            # not on your resume: gone
    acme = resume.experiences[0]
    assert [b.source_bullet_id for b in acme.bullets] == ["exp_acme_b01", "exp_acme_b02"]
    assert acme.bullets[0].text.startswith("Assisted a 4-person team in developing an application")
    assert acme.bullets[1].text == "Wrote SQL queries that reduced report generation time by 30%."   # dropped: back
    assert [e.source_id for e in resume.projects] == ["proj_tracker"]           # left out: back, your wording
    assert resume.projects[0].bullets[0].text == "Built a budget tracking web app with JavaScript and React."
    assert resume.skills[:2] == ["SQL", "Python"] and "Git" in resume.skills and "Docker" not in resume.skills


def test_one_bullet_per_approved_bullet(keep_ctx):
    draft = make_draft(experiences=[ed("exp_acme", [
        bd("Assisted a 4-person team in developing a Python application.", ["exp_acme_f01"], "exp_acme_b01"),
        bd("Wrote SQL queries that cut report generation time by 30%.", ["exp_acme_f02"], None),   # no id: matched by facts
        bd("Tested REST API endpoints with pytest.", ["exp_acme_f03"], None),                       # not on your resume
    ])])
    acme = validate.enforce(draft, keep_ctx).resume.experiences[0]
    assert [b.source_bullet_id for b in acme.bullets] == ["exp_acme_b01", "exp_acme_b02"]
    assert "pytest" not in " ".join(b.text for b in acme.bullets)


def test_the_prompt_asks_to_keep_everything_and_shows_only_your_resume(keep_ctx, ctx):
    system = tailor_resume.build_system(keep_ctx)
    assert "Keep ALL of it" in system and "3 bullets in total" in system and "select at most" not in system
    bank = tailor_resume.fact_bank(keep_ctx)
    assert [e["source_id"] for e in bank["experiences"]] == ["exp_acme"]
    assert [p["source_id"] for p in bank["projects"]] == ["proj_tracker"]
    assert "select at most" in tailor_resume.build_system(ctx)                  # no base resume: as before


def test_the_baseline_is_your_resume_word_for_word(keep_ctx):
    draft = tailor_resume.baseline_draft(keep_ctx)
    assert [e.source_id for e in draft.experiences] == ["exp_acme"] and len(draft.experiences[0].bullets) == 2
    assert [e.source_id for e in draft.projects] == ["proj_tracker"]


def test_page_fit_goes_back_to_your_wording_before_cutting_anything(keep_ctx, analysis):
    long = "Assisted a 4-person team in developing a Python application, " + "in Python " * 12 + "."
    draft = make_draft(experiences=[ed("exp_acme", [bd(long, ["exp_acme_f01"], "exp_acme_b01")])])
    resume = validate.enforce(draft, keep_ctx).resume
    originals = {"exp_acme_b01": "Assisted a 4-person team in developing a Python application."}

    def fits_only_with_your_wording(r, layout):
        return 1 if all(b.text == originals.get(b.source_bullet_id, b.text) for _e, _i, b in r.bullets()) else 2

    result = render.fit_to_page(resume, keep_ctx.master, fits_only_with_your_wording, relevance=Relevance(analysis))
    kinds = [s.kind for s in result.steps]
    assert result.fits and kinds[-1] == "revert" and not any(k.startswith("remove") for k in kinds)
    assert resume.experiences[0].bullets[0].text == originals["exp_acme_b01"] and result.removed == []


def test_a_note_never_repeats_the_degree_line():
    degree = "Bachelor of Science in Computer Engineering, Minor in Computer Science"
    assert render._new_parts("Minor in Computer Science  |  M.S. Finance (4+1 Program)", degree) == "M.S. Finance (4+1 Program)"
    assert render._new_parts("Honors College", degree) == "Honors College"


def test_an_unknown_id_in_your_resume_list_is_an_error(master_data):
    master_data["base_resume"] = ["exp_acme", "exp_nowhere"]
    with pytest.raises(pydantic.ValidationError, match="exp_nowhere"):
        MasterResume.model_validate(master_data)
