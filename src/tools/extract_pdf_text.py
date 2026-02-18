import argparse
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts all text from a PDF file using pypdf."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    reader: PdfReader = PdfReader(pdf_path)
    text: str = ""
    for page in reader.pages:
        page_text: str = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument("pdf_file", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--output_file", type=str, help="Path to save the extracted text."
    )

    args = parser.parse_args()

    pdf_path: Path = Path(args.pdf_file)
    text = extract_text_from_pdf(pdf_path)

    if args.output_file:
        output_path: Path = Path(args.output_file)
        output_path.write_text(text, encoding="utf-8")
        print(f"Extracted text saved to: {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
