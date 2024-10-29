
# SEO Fuzzy Match

**SEO Fuzzy Match** is a Python script that facilitates URL redirection during website migrations. It compares content from an old and updated version of a website, using Excel/CSV data from website crawlers like Screaming Frog SEO Spider, to automatically suggest which URLs in the new site correspond to old ones.

This tool is ideal for those tricky situations where URLs and contents have **changed between website versions**. By comparing the content of the old and new resources, SEO Fuzzy Match can help you identify the best matches and create a redirection plan.

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
3. Choose the column indices for URLs in both the old and new datasets.
4. Select the column indices for text features to compare. You can select multiple columns by separating them with commas (e.g., 1,2,3).
5. (Optional) Select a source URL text file to limit the URLs considered in the old website. The file should contain one URL per line.

The script will output a CSV file (`url_mapping.csv`) with the old URLs, new URLs, and a similarity score, indicating the best URL matches.

### Choosing the Best Features

When selecting columns for text features, consider the following:

* The columns you select should contain *unique* data that can help the tool identify specific resources. Good candidates include page titles, meta descriptions, product SKUs, or other distinctive content.
* Prioritize data that *has not changed* between the old and new websites.
* Even if a column’s content isn’t fully unique, it can still be useful in combination with other columns to identify resources effectively.

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
