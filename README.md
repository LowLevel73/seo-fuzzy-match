
# SEO Fuzzy Match

`seo-fuzzy-match.py` is a Python script that facilitates the URL redirection process during website migrations. It compares content from an old version of a website to its updated version, using text similarity measures to automatically suggest which URLs in the old website correspond to new ones.

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
- `tkinter` for file selection dialogs

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

## License

This project is licensed under the MIT License.

## Version

Current version: 0.9.0
