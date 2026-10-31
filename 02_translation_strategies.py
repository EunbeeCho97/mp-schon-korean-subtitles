
"""Rule-based annotation of Korean subtitle translation strategies."""

import argparse
import re
from pathlib import Path

import pandas as pd


STRATEGY_PATTERNS = {
    "A": r"(이미|벌써|지금 막|금방|예전에|한참 전에)",
    "B": r"(정말|진짜|아주|엄청|무척|굉장히|되게)",
    "C": r"(빨리|어서|얼른|당장|지금 해|해봐|해라|하라|하렴|-거라)",
    "D": r"(잖|했지|그랬지|이미 말했|말했어)",
    "E": r"(그런 셈|글쎄|뭐 그렇|아마도|그런 기분|뭐 그런|그럴 수도)"
}


def classify_strategy(text):
    """Assign the first matching translation strategy (A–F)."""

    if pd.isna(text):
        korean = ""
    else:
        korean = re.sub(r"</?s>", "", str(text)).strip()

    for strategy, pattern in STRATEGY_PATTERNS.items():
        if re.search(pattern, korean):
            return strategy

    return "F"


def main(input_path, output_dir="outputs"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xlsm"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input must be a CSV or Excel file.")

    if "Translation" not in df.columns:
        raise ValueError("Input file must contain a 'Translation' column.")

    df["strategy"] = df["Translation"].apply(classify_strategy)

    counts = (
        df["strategy"]
        .value_counts()
        .reindex(list("ABCDEF"), fill_value=0)
    )

    summary = counts.rename_axis("Strategy").reset_index(name="Count")
    summary["Percentage"] = (
        summary["Count"] / len(df) * 100
    ).round(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_excel(
        output_dir / "translation_strategy_results.xlsx",
        index=False
    )

    summary.to_csv(
        output_dir / "translation_strategy_summary.csv",
        index=False
    )

    if "einstellung" in df.columns:
        cross_tab = pd.crosstab(
            df["einstellung"],
            df["strategy"]
        ).reindex(columns=list("ABCDEF"), fill_value=0)

        cross_tab.to_csv(
            output_dir / "illocution_strategy_crosstab.csv"
        )

    print(f"Processed {len(df):,} subtitle pairs.")
    print("\nTranslation strategies:")
    print(summary.to_string(index=False))
    print(f"\nFiles saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify Korean subtitle translation strategies."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV or Excel file containing a 'Translation' column"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory for output files"
    )

    args = parser.parse_args()
    main(args.input, args.output)
