#!/usr/bin/env python
"""
Check that the first sentence of every docstring is followed by a blank line.

Allow the first sentence, the summary, to run over several lines, but must end with a full stop at the end of a line,
followed by a blank line before the rest of the docstring.

Specifically, ruff checks the rest of the layout: D400 (first paragraph ends with a full stop), with D205
(one-line summary) turned off in pyproject.toml.

Usage::

    python tools/check_docstring_summaries.py [paths ...]      # default: swiftlet tools examples

Prints each docstring whose first paragraph holds more than its first sentence, and exits with status 1 if any do.
"""

import ast
import re
import sys
from pathlib import Path

# full stops that do not end a sentence
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "et al.", "Fig.", "Eq.", "no.")
SENTENCE_END = re.compile(r"\.(\s|$)")


def first_sentence_end(paragraph):
    """
    Index in paragraph just after the full stop ending its first sentence, or None if it has no full stop.

    :param  paragraph:  the first paragraph of a docstring, with its lines joined by spaces
    :type   paragraph:  str
    """
    for match in SENTENCE_END.finditer(paragraph):
        before = paragraph[: match.start() + 1]
        if not before.endswith(ABBREVIATIONS) and not re.search(r"\d\.$", before):
            return match.start() + 1
    return None


def check_docstring(docstring):
    """
    The problem with the summary of a docstring, or None if it has none.

    :param  docstring:  the docstring, as returned by ast.get_docstring
    :type   docstring:  str
    """
    paragraph = []
    for line in docstring.strip().splitlines():
        if not line.strip():
            break
        paragraph.append(line.strip())
    text = " ".join(paragraph)
    end = first_sentence_end(text)
    if end is None:
        return None  # no full stop at all: ruff's D400 reports it
    rest = text[end:].strip()
    if rest:
        return f"the first sentence must be followed by a blank line, not by: {rest[:60]!r}"
    return None


def check_file(path):
    """
    Problems with the docstring summaries of a file, as (line, name, problem).

    :param  path:   path of the Python file
    :type   path:   pathlib.Path
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    problems = []
    for node in [tree] + [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]:
        docstring = ast.get_docstring(node)
        if docstring is None:
            continue
        problem = check_docstring(docstring)
        if problem:
            # the line of the docstring itself
            problems.append((node.body[0].lineno, getattr(node, "name", "module"), problem))
    return problems


def main(paths):
    """
    Check every Python file under paths, print the problems and return the exit status.

    :param  paths:  files or directories to check
    :type   paths:  list of str
    """
    files = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    count = 0
    for path in files:
        for line, name, problem in check_file(path):
            print(f"{path}:{line}: {name}: {problem}")
            count += 1
    print(f"{count} docstring summaries to fix" if count else "All docstring summaries pass.")
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["swiftlet", "tools"]))
