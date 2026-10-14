#!/usr/bin/env python3
"""Resume Tailor - command-line entry point.

    python tailor.py <job_url>                  tailor a resume for one posting
    python tailor.py --text job.txt             paste fallback (posting saved as text)
    python tailor.py <job_url> --strict         original wording unless I approve rewrites
    python tailor.py <job_url> --no-research    skip company research
    python tailor.py <job_url> --no-interview   never ask me follow-up questions
    python tailor.py <job_url> --max-rounds 5 --target 9.0
    python tailor.py <job_url> --cover-letter   also draft a cover letter from the same facts
    python tailor.py --batch links.txt          run many postings without prompts
    python tailor.py --insights                 regenerate the application dashboard
    python tailor.py --find "Company"           look up archived submissions
    python tailor.py --check                    run the startup checks only

Expected failures (missing key, bad model ID, login wall, ...) print a clear
message and exit with status 1 - never a raw stack trace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src import ui
from src.errors import TailorError, UserAbort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tailor.py",
        description="Tailor a one-page, ATS-safe resume to a job posting using only verified facts.",
    )
    parser.add_argument("job_url", nargs="?", help="link to the job posting")
    parser.add_argument("--text", metavar="FILE", help="read the job description from a text file instead of a link")
    parser.add_argument("--strict", action="store_true", help="use approved wording verbatim; rewrites need my approval")
    parser.add_argument("--no-research", action="store_true", help="skip company web research")
    parser.add_argument("--no-interview", action="store_true", help="do not ask me follow-up questions")
    parser.add_argument("--max-rounds", type=int, default=config.MAX_ROUNDS, metavar="N",
                        help=f"maximum revision rounds (default {config.MAX_ROUNDS})")
    parser.add_argument("--target", type=float, default=config.TARGET_QUALITY, metavar="X",
                        help=f"Resume Quality target out of 10 (default {config.TARGET_QUALITY})")
    parser.add_argument("--cover-letter", action="store_true", help="also draft a cover letter from the same verified facts")
    parser.add_argument("--batch", metavar="FILE", help="file with one job link per line; runs without prompts")
    parser.add_argument("--insights", action="store_true", help="regenerate the application intelligence dashboard")
    parser.add_argument("--find", metavar="COMPANY", help="list archived submissions matching a company name")
    parser.add_argument("--check", action="store_true", help="run startup checks (key, models, data files) and exit")
    return parser


def _options(args: argparse.Namespace, *, batch: bool = False):
    from src.loop import RunOptions

    if args.max_rounds < 0:
        raise TailorError("--max-rounds must be 0 or more.")
    if not 0 < args.target <= 10:
        raise TailorError("--target must be between 0 and 10.")
    return RunOptions(
        strict=args.strict,
        research=not args.no_research,
        interview=not args.no_interview and not batch,
        max_rounds=args.max_rounds,
        target=args.target,
        batch=batch,
        cover_letter=args.cover_letter,
    )


def dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    from src import startup

    if args.insights:
        from src import insights

        session = startup.run_startup(need_api=False, quiet=True)
        insights.generate(session.skills_status, session.master)
        return 0

    if args.find:
        from src import archive

        archive.print_matches(args.find)
        return 0

    if args.check:
        startup.run_startup(need_api=True, test_call=True)
        ui.success("startup checks passed")
        return 0

    if args.batch:
        from src import loop

        path = Path(args.batch)
        if not path.exists():
            raise TailorError(f"Batch file not found: {path}")
        urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if not urls:
            raise TailorError(f"{path} contains no links.")
        session = startup.run_startup(need_api=True)
        return loop.run_batch(session, urls, _options(args, batch=True))

    if args.job_url or args.text:
        from src import loop

        text_path = Path(args.text) if args.text else None
        if text_path is not None and not text_path.exists():
            raise TailorError(f"Job description file not found: {text_path}")
        session = startup.run_startup(need_api=True)
        loop.run_job(session, url=args.job_url, text_path=text_path, options=_options(args))
        return 0

    parser.print_help()
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args, parser)
    except UserAbort as exc:
        ui.info(exc.message)
        return 0
    except TailorError as exc:
        ui.error(exc.message, exc.hint)
        return 1
    except KeyboardInterrupt:
        ui.warn("Interrupted - nothing further was written.")
        return 130
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("src."):
            ui.error(f"This command needs {exc.name}, which is not built yet.")
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
