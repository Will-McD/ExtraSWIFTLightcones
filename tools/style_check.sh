#!/usr/bin/env bash
#
# Run SWIFTLET's style checks (the same as .github/workflows/styles.yml) and write their output to a log file.
#
# Usage, from anywhere in the repository:
#     bash tools/style_check.sh            # check everything, log to logs/style_check_<date>_<time>.log
#     bash tools/style_check.sh --docs     # also build the docs with -W -n (slower)
#     bash tools/style_check.sh --diff     # log the full formatting diff, not just the files ruff format would change
#
# Each check is run even if an earlier one fails. The log starts with a summary of the problems each check found,
# followed by each check's full output. Needs: pip install ruff codespell interrogate flake8 flake8-rst-docstrings pydoclint
#
set -u

cd "$(git rev-parse --show-toplevel)" || exit 1
mkdir -p logs
log="logs/style_check_$(date +%Y%m%d_%H%M%S).log"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

summary=()
failed=0
n=0
docs=false
diff=false
for arg in "$@"; do
    case "$arg" in
        --docs) docs=true ;;
        --diff) diff=true ;;
        *) echo "unknown option: $arg"; exit 2 ;;
    esac
done

# run_check <name> <command...>: run a check, keep its output and record whether it passed
run_check() {
    local name="$1"; shift
    echo "running ${name}..."
    n=$((n + 1))
    "$@" > "$tmp/$(printf '%02d' $n)_${name}.out" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        summary+=("PASS  ${name}")
    else
        summary+=("FAIL  ${name}")
        failed=$((failed + 1))
    fi
}

if $diff; then
    run_check "ruff-format"  ruff format --check --diff .
else
    run_check "ruff-format"  ruff format --check --output-format=concise .
fi
# the full list of problems, then a count of each rule
run_check "ruff-check"   bash -c 'ruff check . --output-format=concise; status=$?; echo; echo "--- count per rule ---"; ruff check . --statistics; exit $status'
run_check "docstrings"   flake8 swiftlet --show-source --statistics --count
run_check "spelling"     codespell
run_check "coverage"     interrogate swiftlet
if $docs; then
    rm -rf docs/_build docs/api/generated
    run_check "docs" sphinx-build -E -W -n --keep-going -q -b html docs docs/_build/html
fi

{
    echo "SWIFTLET style check: $(date)"
    echo "branch: $(git rev-parse --abbrev-ref HEAD)   commit: $(git rev-parse --short HEAD)"
    echo
    echo "=== Summary ==="
    printf '%s\n' "${summary[@]}"
    for f in "$tmp"/*.out; do
        name="$(basename "$f" .out)"; name="${name#*_}"
        echo
        echo "=== ${name} ($(wc -l < "$f") lines) ==="
        cat "$f"
    done
} > "$log"

printf '%s\n' "${summary[@]}"
echo "log written to ${log}"
[ $failed -eq 0 ]
