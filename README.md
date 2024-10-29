
# SEO Fuzzy Match

**SEO Fuzzy Match** is a Python script that facilitates URL redirection during website migrations. It compares content from an old and updated version of a website, using Excel/CSV data from website crawlers like Screaming Frog SEO Spider, to automatically suggest which URLs in the old site correspond to new ones.

This tool is ideal for SEO specialists who need to create URL redirects efficiently during migrations.

## Features

- Uses cosine similarity to identify the most similar content between old and new URLs.
- Supports various file formats for input data (CSV and Excel).
- Allows the selection of specific columns for URL and content comparison.
- Exports the matched results into a CSV file, ready for use in URL redirection.

## Dependencies

This script requires the following Python libraries:
- `pandas` for data handling
- `scikit-learn` for TF-IDF vectorization and cosine similarity calculation
- `tkinter` (built-in) for file selection dialogs
- `warnings` (built-in) for suppressing openpyxl warnings
- `re` (built-in) for identifying with regular expressions the warning to suppress

## Installation

Install the required packages via pip:
```bash
pip install pandas scikit-learn
```

## Usage

1. Run the script:
   ```bash
   python seo-fuzzy-match.py
   ```
2. Select the old and new website data files when prompted.
3. Choose the columns for URLs and text features for both old and new datasets.
4. (Optional) Select a source URL file to limit the URLs considered in the old website.

The script will output a CSV file (`url_mapping.csv`) with the old URLs, new URLs, and a similarity score, indicating the best URL matches.

## Changelog

### Version 0.9.1
- Updated handling of warnings from `openpyxl` when reading Excel files without a default style.
- Added comments to the most important steps.

### Version 0.9.0
- Initial release with core functionality:
  - Compares old and new website content using cosine similarity.
  - Reads data from Excel and CSV files, supporting file selection via a GUI dialog.

## License

This project is licensed under the MIT License.
