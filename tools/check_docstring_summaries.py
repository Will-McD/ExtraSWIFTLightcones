#!/usr/bin/env python
"""
Check that the first sentence of every docstring is followed by a blank line.


1. The first sentence (the summary) can run over several lines, but must end with a full stop at the end of a line,
   then a blank line before the rest of the docstring. ruff checks the rest of the layout: D400 (the first paragraph
   ends with a full stop), with D205 (a one-line summary) turned off in pyproject.toml.
2. An __init__ docstring has no summary: it starts with its :param: fields. The class docstring has the summary,
   and the docs show the __init__ parameters after it (autoclass_content = "both" in docs/conf.py).
3. The :type:, :vartype: and :rtype: lines name types the docs can link to (see TYPE_RULES): ``numpy.ndarray``, not
   ``np.ndarray``, ``bool``, not ``boolean``, and no units, which belong in the :param: description.

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

# what a :type:, :vartype: or :rtype: line must not contain, and what to write instead
TYPE_LINE = re.compile(r"^\s*:(type|vartype|rtype)\b[^:]*:(?P<type>.*)$")
TYPE_RULES = (
    (re.compile(r"\bnp\."), "write numpy. instead of np., so the type links to NumPy's docs"),
    (re.compile(r"\bboolean\b"), "write bool instead of boolean"),
    (re.compile(r"\(units of"), "give the units in the :param: description, not in the :type: line"),
)


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
    if docstring.lstrip().startswith(":"):
        return None  # no summary, only fields (an __init__ docstring)

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


def check_init_docstring(docstring):
    """
    The problem with an __init__ docstring, or None if it starts with its fields.

    :param  docstring:  the docstring, as returned by ast.get_docstring
    :type   docstring:  str
    """
    if docstring.lstrip().startswith(":"):
        return None
    return "an __init__ docstring has no summary (the class docstring has it): start with the :param: fields"


def check_type_lines(lines):
    """
    The problems with the :type:, :vartype: and :rtype: lines of a docstring.

    Returns a list of (index of the line in lines, problem).

    :param  lines:  the lines of the docstring, as in the file
    :type   lines:  list of str
    """
    problems = []
    for index, line in enumerate(lines):
        match = TYPE_LINE.match(line)
        if match is None:
            continue
        for pattern, problem in TYPE_RULES:
            if pattern.search(match.group("type")):
                problems.append((index, f"{problem}: {line.strip()!r}"))
    return problems


def check_file(path):
    """
    Problems with the docstrings of a file, as (line, name, problem).

    :param  path:   path of the Python file
    :type   path:   pathlib.Path
    """
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    problems = []
    for node in [tree] + [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]:
        docstring = ast.get_docstring(node)
        if docstring is None:
            continue
        name = getattr(node, "name", "module")
        expr = node.body[0]
        if name == "__init__":
            problem = check_init_docstring(docstring)
        else:
            problem = check_docstring(docstring)
        if problem:
            # the line of the docstring itself
            problems.append((expr.lineno, name, problem))
        for index, problem in check_type_lines(lines[expr.lineno - 1 : expr.end_lineno]):
            problems.append((expr.lineno + index, name, problem))
    return sorted(problems)


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
