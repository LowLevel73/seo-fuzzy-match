__version__ = '0.9.1'
__author__ = 'Enrico Altavilla'

import re
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk, filedialog





def select_file(title, filetypes):
    '''
    Open a file dialog to select a file.

    Parameters:
    title (str): The title of the file dialog window.
    filetypes (list): A list of tuples, where each tuple contains a description and a file extension pattern.

    Returns:
    str: The path of the selected file.
    '''
    root = Tk()
    root.attributes('-topmost', True)  # Ensure the dialog is on top
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.update_idletasks()  # Ensure all tasks are processed
    root.destroy()
    return file_path


def get_user_input():
    '''
    Get user input for the required parameters.
    
    Returns:
    tuple: A tuple containing the following elements:
        pd.DataFrame: The old website data.
        pd.DataFrame: The new website data.
        str: The column name containing the URLs in the old website data.
        str: The column name containing the URLs in the new website data.
        list: The column names containing the text features in the old website data.
        list: The column names containing the text features in the new website data.
        str: The path of the source URLs file.
    '''

    old_file = select_file("Select the OLD website file", [("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
    if not old_file:
        print("Old website file must be provided. Exiting...")
        exit(1)
    
    new_file = select_file("Select the NEW website file", [("Excel files", "*.xlsx"), ("CSV files", "*.csv")])
    if not new_file:
        print("New website file must be provided. Exiting...")
        exit(1)
    
    old_data, old_sheet_name = read_file(old_file)
    new_data, new_sheet_name = read_file(new_file)
    
    if old_sheet_name:
        print(f"Using sheet '{old_sheet_name}' from {old_file}")
    if new_sheet_name:
        print(f"Using sheet '{new_sheet_name}' from {new_file}")
    
    old_url_column = select_url_column(old_data, "URL or address column in the old website file")
    new_url_column = select_url_column(new_data, "URL or address column in the new website file")
    
    old_text_columns = select_feature_columns(old_data, "text features in the old website file")
    new_text_columns = select_feature_columns(new_data, "text features in the new website file")
    
    print("Finished selecting columns. Now selecting the optional source URLs file.")
    source_urls_file = select_file("Select the source URLs text file (optional)", [("Text files", "*.txt")])
    
    return old_data, new_data, old_url_column, new_url_column, old_text_columns, new_text_columns, source_urls_file

def read_file(file_path):
    '''
    Read a file and return the data.

    Parameters:
    file_path (str): The path of the file to read.

    Returns:
    pd.DataFrame: The data read from the file.
    str: The name of the sheet if the file is an Excel file, otherwise None.
    '''

    # Suppress openpyxl warning about missing default style
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=re.escape('openpyxl.styles.stylesheet'))

        print(f"Reading file: {file_path}")
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path), None
        elif file_path.endswith('.xlsx'):
            xls = pd.ExcelFile(file_path)
            if len(xls.sheet_names) > 1:
                print("Multiple sheets found:")
                for idx, sheet in enumerate(xls.sheet_names):
                    print(f"{idx}: {sheet}")
                sheet_idx = int(input("Enter the number of the sheet to use: "))
                return pd.read_excel(xls, sheet_name=xls.sheet_names[sheet_idx]), xls.sheet_names[sheet_idx]
            else:
                return pd.read_excel(xls), xls.sheet_names[0]
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")


def select_url_column(data, description):
    '''
    Select the column containing the URLs.

    Parameters:
    data (pd.DataFrame): The data containing the columns.
    description (str): A description of the data.

    Returns:
    str: The name of the selected column.
    '''
    
    # Find the columns containing in their name a keyword that suggests they contain URLs or addresses
    url_keywords = ['url', 'address', 'source', 'destination']
    columns = [col for col in data.columns if any(keyword in col.lower() for keyword in url_keywords)]

    if not columns:
        raise ValueError(f"No columns containing 'url' or 'address' found for {description}.")
    
    print(f"Select the {description}:")
    for idx, col in enumerate(columns):
        print(f"{idx}: {col}")
    
    col_idx = int(input("Enter the number of the column: "))
    
    return columns[col_idx]


def select_feature_columns(data, description):
    '''
    Select the columns containing the text features.

    Parameters:
    data (pd.DataFrame): The data containing the columns.
    description (str): A description of the data.

    Returns:
    list: The names of the selected columns.
    '''

    print(f"Select the {description}:")
    for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")
    col_indices = input("Enter the numbers of the columns (comma-separated): ").split(',')
    return [data.columns[int(idx)] for idx in col_indices]


def read_source_urls(file_path):
    '''
    Read the source URLs from a text file.

    Parameters:
    file_path (str): The path of the text file containing the source URLs.

    Returns:
    list: The source URLs read from the file.
    '''

    if not file_path:
        return None
    print(f"Reading source URLs from: {file_path}")
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


def calculate_similarity(old_data, new_data, old_url_column, old_text_columns, new_text_columns, source_urls):
    '''
    Calculate the similarity between the text features of the old and new data.

    Parameters:
    old_data (pd.DataFrame): The old website data.
    new_data (pd.DataFrame): The new website data.
    old_url_column (str): The column name containing the URLs in the old website data.
    old_text_columns (list): The column names containing the text features in the old website data.
    new_text_columns (list): The column names containing the text features in the new website data.
    source_urls (list): The source URLs to consider.

    Returns:
    np.ndarray: The similarity matrix between the old and new data.
    '''

    # If the user only wants to consider a subset of the URLs, filter the data accordingly
    if source_urls is not None:
        old_data = old_data[old_data[old_url_column].isin(source_urls)]

    # Combine the text features into a single column    
    old_texts = old_data[old_text_columns].fillna('').agg(' '.join, axis=1)
    new_texts = new_data[new_text_columns].fillna('').agg(' '.join, axis=1)
    
    # We combine the old and new texts to create a common vocabulary for the vectorizer
    combined_texts = old_texts.tolist() + new_texts.tolist()
    
    # Create a TF-IDF vectorizer and fit it on the combined texts
    # This operation cleans the text and creates a vocabulary
    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined_texts)

    # Transform the old and new texts into TF-IDF vectors
    tfidf_matrix_old = vectorizer.transform(old_texts)
    tfidf_matrix_new = vectorizer.transform(new_texts)
    
    # Calculate the cosine similarity between the old and new texts.
    # This is OK for small datasets, but it's a waste of memory anyway.
    # For larger datasets, we should calculate the similarity in batches.
    similarity_matrix = cosine_similarity(tfidf_matrix_old, tfidf_matrix_new)
    return similarity_matrix


def find_best_matches(similarity_matrix, old_urls, new_urls):
    '''
    Find the best matches between the old and new URLs based on the similarity matrix.

    Parameters:
    similarity_matrix (np.ndarray): The similarity matrix between the old and new data.
    old_urls (list): The URLs in the old data.
    new_urls (list): The URLs in the new data.

    Returns:
    list: A list of tuples containing the best matches between the old and new URLs.
    '''

    best_matches = []
    for idx, similarities in enumerate(similarity_matrix):
        # Find the index of the new resource that has the highest similarity to the old resource
        best_match_idx = similarities.argmax()

        # Append the best match to the list
        best_matches.append((old_urls[idx], new_urls[best_match_idx], similarities[best_match_idx]))
    return best_matches


def save_matches(matches, output_file='url_mapping.csv'):
    '''
    Save the best matches to a CSV file.

    Parameters:
    matches (list): A list of tuples containing the best matches between the old and new URLs.
    output_file (str): The path of the output CSV file.
    '''

    df = pd.DataFrame(matches, columns=['Source URL', 'Destination URL', 'Similarity Score'])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():

    # Get user input
    old_data, new_data, old_url_column, new_url_column, old_text_columns, new_text_columns, source_urls_file = get_user_input()

    # If the user only wants to consider a subset of the source URLs, we read them from a textual file    
    source_urls = read_source_urls(source_urls_file)

    # Calculate the similarity between the old and new data
    # It's OK to calculate the similarity matrix in memory for small datasets
    # For larger datasets, we should calculate the similarity in batches    
    similarity_matrix = calculate_similarity(old_data, new_data, old_url_column, old_text_columns, new_text_columns, source_urls)

    # For each old resource, find the most similar new resource    
    best_matches = find_best_matches(similarity_matrix, old_data[old_url_column].tolist(), new_data[new_url_column].tolist())
    
    # Save the best matches to a CSV file
    save_matches(best_matches, 'url_mapping.csv')

if __name__ == "__main__":
    main()
