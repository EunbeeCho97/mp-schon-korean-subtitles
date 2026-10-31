
"""Rule-based classification of German sentences by illocutionary type."""

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy


def classify_sentence(text, nlp):
    """Assign an interrogative, directive, or assertive label."""

    sentence = re.sub(r"<[^>]+>", "", str(text)).strip()

    if not sentence:
        return "unclassified"

    # Interrogatives are identified by punctuation or question words.
    if "?" in sentence or re.search(
        r"\b(was|wer|wie|wo|wann|warum|ob)\b",
        sentence.lower()
    ):
        return "interrogative"

    # Directives are identified using punctuation, verb position,
    # and the absence of an explicit subject.
    doc = nlp(sentence)
    first_pos = doc[0].pos_ if len(doc) else None
    has_subject = any(token.dep_ == "nsubj" for token in doc)

    directive_cues = [
        sentence.endswith("!"),
        first_pos in {"VERB", "AUX"},
        not has_subject
    ]

    if sum(directive_cues) >= 2:
        return "directive"

    return "assertive"


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

    if "Sentence" not in df.columns:
        raise ValueError("Input file must contain a 'Sentence' column.")

    nlp = spacy.load("de_core_news_md", disable=["ner"])

    print(f"Processing {len(df):,} sentences...")

    df["illocution_type"] = [
        classify_sentence(sentence, nlp)
        for sentence in df["Sentence"]
    ]

    categories = [
        "assertive",
        "directive",
        "interrogative",
        "unclassified"
    ]

    counts = (
        df["illocution_type"]
        .value_counts()
        .reindex(categories, fill_value=0)
    )

    summary = counts.rename_axis("Illocution").reset_index(name="Count")
    summary["Proportion (%)"] = (
        summary["Count"] / len(df) * 100
    ).round(2)

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_excel(
        output_dir / "illocution_classified.xlsx",
        index=False
    )

    summary.to_csv(
        output_dir / "illocution_summary.csv",
        index=False
    )

    print("\nClassification results:")
    print(summary.to_string(index=False))
    print(f"\nFiles saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify German sentences by illocutionary type."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV or Excel file containing a 'Sentence' column"
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory for output files"
    )

    args = parser.parse_args()
    main(args.input, args.output)
