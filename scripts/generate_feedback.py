#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_feedback.py
- Teacher-triggered via GitHub Actions
- Uses GPT to produce process-oriented meta-feedback
- Supports both Code (algorithm.py) and Pseudocode (PSEUDOCODE.md)
- No preset/default hints; ALL guidance comes from the model
- Added "Change summary" (commits/files/added/removed/top files)
"""

import os
import json
import subprocess
import datetime
import pathlib
import re
import textwrap
from typing import Dict, Any

# ---------------- constants & paths ----------------
STATE_DIR = pathlib.Path(".meta-feedback")
STATE_DIR.mkdir(exist_ok=True)
STATE_FILE = STATE_DIR / "state.json"

PSEUDO_FILE = pathlib.Path("PSEUDOCODE.md")
CODE_FILE = pathlib.Path("algorithm.py")
TEST_DIR = pathlib.Path("tests")


# ---------------- shell helpers ----------------
def sh(cmd: str) -> str:
    """Run a shell command and return stdout as text (raises on non-zero)."""
    return subprocess.check_output(
        cmd, shell=True, text=True, stderr=subprocess.STDOUT
    ).strip()


def get_head() -> str:
    return sh("git rev-parse HEAD")


def get_initial_commit() -> str:
    return sh("git rev-list --max-parents=0 HEAD").splitlines()[0]


def get_last_processed() -> str | None:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text()).get("last_processed")
        except Exception:
            return None
    return None


def set_last_processed(sha: str) -> None:
    STATE_FILE.write_text(json.dumps({"last_processed": sha}, indent=2), encoding="utf-8")


def git_diff(base: str, head: str) -> Dict[str, str]:
    """Collect a compact view of repo changes + context for the model."""
    name_status = sh(f"git diff --name-status {base} {head}") if base != head else ""
    shortstat = sh(f"git diff --shortstat {base} {head}") if base != head else ""
    patch = sh(f"git diff --unified=0 {base} {head}") if base != head else ""
    try:
        tree = sh("ls -R | head -n 400")
    except Exception:
        tree = ""
    try:
        logs = sh("git log -n 10 --pretty=format:'%h %ad %s' --date=short")
    except Exception:
        logs = ""
    return {
        "name_status": name_status,
        "shortstat": shortstat,
        "patch": patch,
        "tree": tree,
        "logs": logs,
    }


def now_stamp() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")


def write_feedback(md: str) -> str:
    out = f"Meta-Feedback_{now_stamp()}.md"  # timestamp
    pathlib.Path(out).write_text(md, encoding="utf-8")
    return out


# ---------------- change summary (new) ----------------
def count_commits_between(base: str, head: str) -> int:
    """Count commits from base(exclusive) to head(inclusive)."""
    if base == head:
        return 0
    try:
        return int(sh(f"git rev-list --count {base}..{head}"))
    except Exception:
        return 0


def build_change_summary(base: str, head: str) -> str:
    """
    Summarize change size between two SHAs using numstat (robust across locales).
    Returns a short multiline string for the prompt.
    """
    if base == head:
        return "No new changes."

    # Sum added/removed by file
    added_total = 0
    deleted_total = 0
    file_changes = []  # [(adds, dels, path), ...]

    try:
        numstat = sh(f"git diff --numstat {base} {head}")
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            a, d, path = parts[0], parts[1], parts[2]
            # binary changes show as '-' — skip counting lines in that case
            ia = int(a) if a.isdigit() else 0
            idel = int(d) if d.isdigit() else 0
            added_total += ia
            deleted_total += idel
            file_changes.append((ia, idel, path))
    except Exception:
        # Fallback to shortstat if numstat fails
        try:
            short = sh(f"git diff --shortstat {base} {head}")
        except Exception:
            short = ""
        # Parse e.g. "3 files changed, 42 insertions(+), 7 deletions(-)"
        import re as _re
        files_changed = int(_re.search(r"(\d+) files? changed", short).group(1)) if "changed" in short else 0
        ins = int(_re.search(r"(\d+) insertions?\(\+\)", short).group(1)) if "insertions" in short else 0
        dels = int(_re.search(r"(\d+) deletions?\(-\)", short).group(1)) if "deletions" in short else 0
        added_total, deleted_total = ins, dels
        return "\n".join([
            f"- Commits since last feedback: {count_commits_between(base, head)}",
            f"- Files changed: {files_changed}",
            f"- Lines added: {added_total}, removed: {deleted_total}",
        ])

    files_changed = len(file_changes)
    commits = count_commits_between(base, head)

    # Top-N files by churn (adds + dels), max 5
    file_changes.sort(key=lambda t: (t[0] + t[1]), reverse=True)
    top = file_changes[:5]
    top_str = ", ".join([f"{p} (+{a}/-{d})" for a, d, p in top]) if top else "(none)"

    return "\n".join([
        f"- Commits since last feedback: {commits}",
        f"- Files changed: {files_changed}",
        f"- Lines added: {added_total}, removed: {deleted_total}",
        f"- Top changed files: {top_str}",
    ])


# ---------------- minimal content heuristics (no preset hints) ----------------
def strip_comments_and_ws_py(text: str) -> str:
    # Remove Python comments and whitespace for a rough size signal
    text = re.sub(r"#.*", "", text)
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return re.sub(r"\s+", "", text)


def strip_md(text: str) -> str:
    # Remove common Markdown markup & code fences for a rough size signal
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    text = re.sub(r"[#>*`_~\-]", "", text)
    return re.sub(r"\s+", "", text)


def read_text_safe(p: pathlib.Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def detect_content_status() -> Dict[str, Any]:
    """Return simple signals about whether students actually wrote content."""
    code_text = read_text_safe(CODE_FILE)
    pseudo_text = read_text_safe(PSEUDO_FILE)

    code_core_len = len(strip_comments_and_ws_py(code_text)) if code_text else 0
    pseudo_core_len = len(strip_md(pseudo_text)) if pseudo_text else 0

    tests_exist = TEST_DIR.exists() and any(TEST_DIR.glob("test_*.py"))

    # very rough thresholds to differentiate skeleton vs. substance
    has_meaningful_code = code_core_len >= 80
    has_meaningful_pseudo = pseudo_core_len >= 120

    return {
        "code_len": code_core_len,
        "pseudo_len": pseudo_core_len,
        "has_code": has_meaningful_code,
        "has_pseudo": has_meaningful_pseudo,
        "tests_exist": tests_exist,
    }


def describe_content_status(status: Dict[str, Any]) -> str:
    """
    Summarize what exists WITHOUT giving any hardcoded hints.
    This context helps GPT decide how to respond when the repo is minimal/empty.
    """
    details = []
    if status["has_code"]:
        details.append(f"Detected non-trivial code in algorithm.py (≈{status['code_len']} chars after stripping).")
    if status["has_pseudo"]:
        details.append(f"Detected non-trivial pseudocode in PSEUDOCODE.md (≈{status['pseudo_len']} chars after stripping).")
    if not status["has_code"] and not status["has_pseudo"]:
        details.append("No substantial code or pseudocode detected.")
    if not status["tests_exist"]:
        details.append("No test files found under 'tests/'.")
    return " ".join(details) if details else "Content status unclear."


# ---------------- OpenAI client ----------------
def call_gpt(system_prompt: str, user_prompt: str) -> str:
    """
    Robust OpenAI call (defaults + retries).
    """
    import time
    from openai import OpenAI
    from openai import APIConnectionError, APIError, RateLimitError, Timeout

    # Coalesce empty strings to sensible defaults
    base_url_env = os.environ.get("OPENAI_BASE_URL")
    base_url = base_url_env.strip() if base_url_env and base_url_env.strip() else None

    model_env = os.environ.get("OPENAI_MODEL")
    model = (model_env.strip() if model_env and model_env.strip() else "gpt-4o-mini")

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=base_url,
        timeout=20,
    )

    last_err = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except (APIConnectionError, Timeout) as e:
            last_err = f"network/timeout on attempt {attempt}: {e}"
        except (RateLimitError, APIError) as e:
            last_err = f"api error on attempt {attempt}: {e}"
        time.sleep(2 * attempt)

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}")


# ---------------- prompts ----------------
SYSTEM_PROMPT = """You are a teaching assistant providing **process-oriented meta-feedback** for a university algorithms assignment.

Students may submit either **code (algorithm.py)** or **pseudocode (PSEUDOCODE.md)** — feedback must be **in English** and **easy to understand**.

Focus on **how they plan, implement, test, and reflect**, not just correctness.  
Always give **specific and actionable** suggestions.

If the repository is empty or mostly blank, provide **constructive step-by-step guidance** to help the student start:
- suggest what to write first (e.g., function outline or pseudocode skeleton),
- mention 1–2 simple example inputs to test,
- remind them to explain their algorithm idea clearly.

Keep feedback **concise, structured, and encouraging**.  
Use **Markdown** formatting with short bullet points.  
Do **not** include generic praise or unrelated filler.
"""

USER_PROMPT_TEMPLATE = """\
Generate clear, process-focused meta-feedback for the latest change ({base}..{head}).

[Recent commits]
{logs}

[Project tree (truncated)]
{tree}

[Changed files (name-status)]
{name_status}

[Change summary]
{change_summary}

[Diff (unified; may be truncated)]
```diff
{patch}
```

[Repository content summary]
{content_summary}

Use this structure:

Meta-Feedback (Process-Oriented)
Signals Observed

(What changed? Are commits small and clear?)

(Any structure/API changes? Added/removed helpers?)

Actionable Suggestions

Planning:
(Did the student describe the problem and plan the approach clearly? What to improve?)

Implementation / Pseudocode quality:
(Clarity, correctness, readability, logic flow)

Validation:
(Tests, examples, or missing edge cases)

Reflection:
(Complexity, correctness reasoning, alternative approaches)

If the submission is empty or minimal

Give simple step-by-step hints for starting a solution.

Suggest writing a function/pseudocode outline and one or two test cases.

Encourage short comments explaining each step.
"""
# ---------------- main ----------------
def main() -> None:
    head = get_head()
    last = get_last_processed() or get_initial_commit()

    if last == head:
        print("No new commits to process.")
        return

    # Collect diff/context
    d = git_diff(last, head)

    # Describe current repo contents (no preset hints)
    status = detect_content_status()
    content_summary = describe_content_status(status)

    # Change summary
    change_summary = build_change_summary(last, head)

    # Build user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        base=last[:7],
        head=head[:7],
        logs=d["logs"] or "(none)",
        tree=d["tree"] or "(none)",
        name_status=d["name_status"] or "(none)",
        change_summary=change_summary,
        patch=(d["patch"][:20000] if d["patch"] else "(none)"),
        content_summary=content_summary,
    )

    # Call GPT
    try:
        feedback_md = call_gpt(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        feedback_md = (
            "# Meta-Feedback (Process-Oriented)\n"
            f"The feedback service encountered an error: {e}\n"
            "Please retry the instructor-triggered workflow.\n"
        )

    out = write_feedback(feedback_md)
    set_last_processed(head)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()


