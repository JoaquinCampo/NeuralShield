import sys
from difflib import unified_diff
from pathlib import Path

from loguru import logger

from neuralshield.preprocessing.pipeline import preprocess


def run_tests() -> int:
    """
    Run preprocessing pipeline against all .in/.out pairs and write diffs.

    Returns:
        Number of failing test cases (non-zero if any diff found).
    """
    base_dir = Path(__file__).parent

    result_dir = base_dir / "result"
    in_dir = base_dir / "in"
    out_dir = base_dir / "out"
    diff_dir = base_dir / "diff"

    base_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    in_files = sorted(in_dir.glob("*.in"))
    if not in_files:
        logger.warning("No .in files found in {}", in_dir)
        return 0

    total = 0
    failures = 0

    for in_file in in_files:
        total += 1
        test_name = in_file.stem

        raw_request = in_file.read_text(encoding="utf-8")
        actual = preprocess(raw_request)

        # Always write the generated processed request
        result_file = result_dir / f"{test_name}.actual"
        result_file.write_text(actual, encoding="utf-8")

        expected_file = out_dir / f"{test_name}.out"
        if not expected_file.exists():
            logger.error(
                "Missing expected file for {}: {}", in_file.name, expected_file
            )
            failures += 1
            # Skip generating diff when expected is missing
            continue

        expected = expected_file.read_text(encoding="utf-8")

        actual_lines = actual.splitlines(keepends=True)
        expected_lines = expected.splitlines(keepends=True)

        diff_lines = list(
            unified_diff(
                expected_lines,
                actual_lines,
                fromfile=str(expected_file.name),
                tofile=f"{test_name}.actual",
            )
        )

        if diff_lines:
            failures += 1
            diff_text = "".join(diff_lines)
            diff_path = diff_dir / f"{test_name}.diff"
            diff_path.write_text(diff_text, encoding="utf-8")
            logger.error("Test {} FAILED. Diff written to {}", test_name, diff_path)
        else:
            logger.info("Test {} passed", test_name)

    logger.info("Completed {} tests. {} failed.", total, failures)
    return failures


def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    exit_code = 1 if run_tests() else 0
    # Do not call sys.exit to keep script import-friendly
    if exit_code:
        logger.error("There were failing tests.")
    else:
        logger.info("All tests passed.")


if __name__ == "__main__":
    main()
