"""Helpers for reading the dashboard's front-end sources from Python tests.

The dashboard is a compiled React application. Its controls, its severity
styling and its health panel live in TypeScript that gets minified into
``static/assets/index.js``, so a test cannot usefully grep the served HTML for
them -- ``index.html`` is a shell with a ``<div id="root">`` in it.

That is exactly how the page came to satisfy its own contract tests with a
hidden block of decoy markup: three ``display:none`` ``<select>`` elements and a
never-called function whose only purpose was to put the strings ``stage.disabled``
and friends somewhere a regex would find them. The tests passed against nothing.

These helpers read the *sources* instead, which is where the contract actually
lives and which is committed alongside the Python. A drift between the Python
enums and the TypeScript vocabulary now fails, and it fails for the real reason.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
STATIC = Path(__file__).resolve().parents[1] / "src" / "vantage" / "dashboard" / "static"


def source(relative: str) -> str:
    """Read one front-end source file, skipping the test if it is absent.

    Skipped rather than failed when the whole ``frontend`` tree is missing: a
    source checkout without it can still run every other test, and a hard failure
    there would say "the contract drifted" when the truth is "the file is not
    here".
    """
    path = FRONTEND / relative
    if not path.is_file():
        pytest.skip(f"front-end source {relative} is not present in this checkout")
    return path.read_text(encoding="utf-8")


def built(relative: str) -> str:
    """Read one built artefact from the served static directory."""
    path = STATIC / relative
    if not path.is_file():
        pytest.skip(f"built asset {relative} is not present; run `npm run build` in frontend/")
    return path.read_text(encoding="utf-8")


@cache
def string_array(text: str, name: str) -> tuple[str, ...]:
    """Extract ``export const NAME = ['a', 'b'] as const`` from TypeScript.

    A parser rather than a substring check, so the test asserts the *set* the
    front end offers instead of merely that a word appears somewhere in the file.
    """
    match = re.search(rf"export const {name} = \[(.*?)\] as const;", text, re.S)
    assert match, f"{name} is not declared as a const array in the vocabulary"
    return tuple(re.findall(r"'([^']+)'", match.group(1)))


def object_keys(text: str, name: str) -> tuple[str, ...]:
    """Keys of ``export const NAME: ... = { a: ..., b: ... }``.

    Brace-matched rather than regex-terminated. These are written both across
    several lines and on one, and a pattern anchored to a closing newline ran
    straight on into the next declaration and returned its keys too.
    """
    start = text.index(f"export const {name}")
    open_brace = text.index("{", start)
    depth = 0
    body = ""
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                body = text[open_brace + 1 : index]
                break
    assert body, f"{name} has unbalanced braces"
    # The opening brace is put back so the first key has the same delimiter
    # before it as every other key; without it the first entry of a multi-line
    # object was silently dropped.
    return tuple(re.findall(r"[{,]\s*(\w+)\s*:", "{" + body))
