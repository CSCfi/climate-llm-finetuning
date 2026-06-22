import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    # Import necessary libraries
    import marimo as mo

    import os
    import re
    import requests
    import threading
    import time
    import json
    import pymupdf
    import random
    import shutil

    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path
    from datasets import load_from_disk
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    from urllib.parse import urljoin
    from bs4 import BeautifulSoup
    from collections import defaultdict

    # Apply paths (data download folder)
    JSON_FILE_NAME = "extracted_texts_from_xml_pdf.json"
    DATA_PATH = Path(
        "/scratch/",
        os.getenv("SLURM_JOB_ACCOUNT"),
        "data")
    JSON_FILE_FOLDER = Path(
        "/scratch",
        os.getenv("SLURM_JOB_ACCOUNT"),
        os.getenv("SLURM_JOB_USER"),
        "climate-llm-finetuning",
        "data")
    JSON_FILE_FOLDER.mkdir(parents=True, exist_ok=True)

    JSON_FILE_PATH = Path(JSON_FILE_FOLDER / JSON_FILE_NAME)

    # Where the csv file and pdf/xml files should be saved, currently can differ from actual slurm project
    CSV_FILES_FOLDER = Path(DATA_PATH / "copernicus_files" / "csv_files")
    DOWNLOAD_FOLDER = Path(DATA_PATH / "copernicus_files" / "copernicus_new")

    NUM_WORKERS = int(os.getenv("SLURM_CPUS_PER_TASK"))

    DATA_AMOUNT = 2000 # EDITABLE! To run the notebook faster, you can adjust how many PDF/XML files will be downloaded

    REMOVE_FILES = True

    # NOTE! If in Marimo notebook view and the popup doesn't have an option to install required packages via pip, you can install them by clicking the Manage packages icon on the left side of the app view and installing the required packages from there.


@app.cell
def _():
    if not os.path.exists(CSV_FILES_FOLDER):
        CSV_FILES_FOLDER.mkdir(parents=True, exist_ok=True)
        print("CSV save folder created")

    if not os.path.exists(DOWNLOAD_FOLDER):
        DOWNLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        print("File download folder created")

    print(NUM_WORKERS)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data from Copernicus

    Data is gathered from Copernicus website. Since Copernicus has restricted programmatical access to the journals, proceed straight to 3rd option described in the instructions below.

    Instructions on usage (you can scroll to certain sections with the navigator located on the right of the app view):
    - If you are starting this notebook from scratch, [**start from the beginning**](#Data-from-Copernicus).
    - If you have downloaded the article URLs and saved them in a CSV file (you have gone through the **Get the URLs of the articles** section), proceed to section [**Download the PDF and XML files**](#Download-the-PDF-and-XML-files)
    - If you have downloaded the PDF and XML files, proceed to section [**Extract text from files**](#Extract-text-from-files)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get the URLs of the articles
    """)
    return


@app.class_definition
class SublinkFetcher:
    def __init__(self, base_url, max_retries=5, delay_range=(1, 3)):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; AcademicBot/1.0; +https://example.com/bot)"}
        )
        self.max_retries = max_retries
        self.delay_range = delay_range

    def fetch_html(self, url):
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=60)
                response.raise_for_status()
                return BeautifulSoup(response.text, "html.parser")
            except (
                requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt == self.max_retries:
                    print(f"Failed to fetch {url} after {self.max_retries} retries: {e}")
                    return None
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
        return None

    def extract_sublinks(self, url, exclude_images=True):
        soup = self.fetch_html(url)
        if soup is None:
            return []
        links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]
        if exclude_images:
            links = [link for link in links if not link.lower().endswith((".png", ".jpg"))]
        return links

    def filter_links(self, links, condition_func=None):
        if condition_func is None:
            return links
        return [link for link in links if condition_func(link)]

    # --- Conditions ---
    @staticmethod
    def article_condition(link):
        return (
            (link.endswith(".copernicus.org") or link.endswith(".copernicus.org/") or link.endswith("articles/"))
            and ("www" not in link.split("//")[1])
            and ("meetings" not in link)
        )

    @staticmethod
    def article_issue_condition(link):
        return ("article" in link and "issue" in link) and ".pdf" not in link

    @staticmethod
    def last_three_number_condition(link):
        last_three_parts = link.split("/")[-4:-1]
        return all(re.match(r"^\d+$", part) for part in last_three_parts)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    Here we first get all the journals from the copernicus website.
    """)
    return


@app.function
def get_journals(base_url):
    fetcher = SublinkFetcher(base_url)
    journal_urls = fetcher.filter_links(
        fetcher.extract_sublinks(base_url),
        SublinkFetcher.article_condition
    )

    print(f"Amount of journals in Copernicus: {len(journal_urls)}")
    return journal_urls


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    Then we get all the possible articles in each journal.
    """)
    return


@app.function
def get_articles(journal_urls, base_url):
    fetcher = SublinkFetcher(base_url)
    all_download_links = []
    journal_article_counts = {}
    journal_year_counts = defaultdict(lambda: defaultdict(int))

    start = time.time()

    for link_1 in tqdm(journal_urls, desc="\nJournal URLs"):
        print(f"Processing: {link_1}")
        journal_name = link_1.split("//")[-1].split(".")[0]

        # Try both /articles/ and root depending on URL pattern
        if "articles" in link_1:
            issue_links = fetcher.filter_links(
                fetcher.extract_sublinks(link_1),
                SublinkFetcher.article_issue_condition,
            )
        else:
            issue_links = fetcher.filter_links(
                fetcher.extract_sublinks(link_1 + "/articles/"),
                SublinkFetcher.article_issue_condition,
            )

        article_count = 0
        for link_2 in tqdm(issue_links, desc="Issue pages", leave=False):
            article_links = fetcher.filter_links(
                fetcher.extract_sublinks(link_2),
                SublinkFetcher.last_three_number_condition,
            )
            filtered_urls = [
                url for url in article_links
                if url != "javascript:void(0)"
            ]

            # Extract years from article URLs
            for url in filtered_urls:
                parts = url.strip("/").split("/")
                year = parts[-1]
                if re.match(r"^\d{4}$", year):
                    journal_year_counts[journal_name][int(year)] += 1

            article_count += len(filtered_urls)
            all_download_links.extend(filtered_urls)

        journal_article_counts[journal_name] = article_count

    end = time.time()
    print(f"Took {end-start:.0f} seconds to get all sublinks")
    print(f"Total URLs collected: {len(all_download_links)}")
    return all_download_links, journal_article_counts, journal_year_counts


@app.cell
def _():
    journal_urls = get_journals(base_url="https://publications.copernicus.org/open-access_journals/journals_by_subject.html")
    return (journal_urls,)


@app.cell
def _(journal_urls):
    all_download_links, journal_article_counts, journal_year_counts = get_articles(journal_urls, base_url="https://publications.copernicus.org/open-access_journals/journals_by_subject.html")
    return all_download_links, journal_article_counts, journal_year_counts


@app.cell
def _(journal_article_counts):
    # --- Visualization 1: Total articles per journal ---
    total_articles = sum(journal_article_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(journal_article_counts.keys(), journal_article_counts.values(), color="skyblue")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Articles")
    plt.title(f"Articles per Journal\nTotal Articles: {total_articles}", fontsize=14)

    # Optional: display the total above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{int(height)}', ha='center', va='bottom', rotation=30)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(journal_year_counts):
    # --- Prepare DataFrame ---
    data_rows = []
    for (journal, year_counts) in journal_year_counts.items():
        for (_year, count) in year_counts.items():
            data_rows.append({'Journal': journal, 'Year': _year, 'Articles': count})
    df = pd.DataFrame(data_rows)

    def plot_journals(year_range, min_articles):
    # --- Interactive Plot Function ---
        plt.figure(figsize=(12, 6))
        for journal in sorted(df['Journal'].unique()):
            subset = df[(df['Journal'] == journal) & (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) & (df['Articles'] >= min_articles)]
            if not subset.empty:
                years_sorted = sorted(subset['Year'].unique())
                counts_sorted = [subset[subset['Year'] == y]['Articles'].iloc[0] for y in years_sorted]
                plt.plot(years_sorted, counts_sorted, marker='o', label=journal)
        plt.xlabel('Year')
        plt.ylabel('Number of Articles')
        plt.title('Articles per Year per Journal')
        plt.legend(ncols=3, loc='upper left', columnspacing=1.5)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


    return df, plot_journals


@app.cell
def _(df):
    year_slider = mo.ui.range_slider.from_series(df['Year'])
    year_slider
    return (year_slider,)


@app.cell
def _(df):
    min_articles_slider = mo.ui.slider.from_series(df['Articles'])
    min_articles_slider
    return (min_articles_slider,)


@app.cell
def _(min_articles_slider, plot_journals, year_slider):
    plot_journals(year_slider.value, min_articles_slider.value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Save the article links in a csv file
    ---
    Next we save the article links in a csv file for future use.
    """)
    return


@app.function
def save_article_links(CSV_FILES_FOLDER, all_download_links):
    try:
        existing_df = pd.read_csv(f"{CSV_FILES_FOLDER}/urls.csv", header=None, index_col=False)
        existing_links = set(existing_df[0])
    except:
        existing_links = []
    new_unique_links = [link for link in all_download_links if link not in existing_links]
    if new_unique_links:
        new_df = pd.DataFrame(new_unique_links)
        new_df.to_csv(f"{CSV_FILES_FOLDER}/urls.csv", mode="a", header=False, index=False)
        print("Links saved")
    else:
        print("No new links to be saved")


@app.cell
def _(all_download_links):
    save_article_links(CSV_FILES_FOLDER, all_download_links)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Download the PDF and XML files

    Now that we have all the links of the articles saved in a csv file, we can start downloading them. We will first download all the XML files and then the PDF files. Both format types are required since not every XML file that can be downloaded has the full text of the article. The raw text attainable from the XML file is better in quality than in PDF (it's easy to extract only the text and exclude tables and figures from XML file).
    """)
    return


@app.function
def download_file(url, folder_path, file_format, downloaded_data, skipped, lock):
    try:
        topic_name = url.split('//')[1].split('.')[0]
        article_name = url.split('/')[-4:-1]
        article_name = '-'.join(article_name) + f'.{file_format}'
        article_name = topic_name + '-' + article_name

        if article_name in downloaded_data:
            print(f'{article_name} already downloaded')
            return

        with requests.Session() as session:
            response = session.get(url + article_name, timeout=60)

        if response.status_code == 200:
            with open(os.path.join(folder_path, article_name), 'wb') as file:
                file.write(response.content)
        else:
            with lock:
                skipped["count"] += 1
            print(f'Failed to download {article_name}. HTTP Status Code: {response.status_code}')

    except Exception as e:
        with lock:
            skipped["count"] += 1
        print(f'Error downloading {url}: {e}')


@app.function
def ensure_folder_exists(folder_path):
    """Ensure the download folder exists or create it if it doesn't."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


@app.function
def fast_download_files(urls, folder_path, file_format, max_workers=5):
    ensure_folder_exists(folder_path)

    lock = threading.Lock()
    # session.headers.update(
    #     {"User-Agent": 'python-requests/2.32.5'}
    # )
    downloaded_data = set(os.listdir(folder_path))

    skipped = {"count": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                download_file,
                url,
                folder_path,
                file_format,
                downloaded_data,
                skipped,
                lock
            )
            for url in urls
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f'Downloading {file_format}s'):
            future.result()  # let exceptions propagate here

    return skipped["count"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Since it's possible that the notebook environment crashes, we get the download links from the csv file that was created before in section **Save the article links in a csv file**.
    """)
    return


@app.function
def download_pdf_xml_files(CSV_FILES_FOLDER):
    df_1 = pd.read_csv(f'{CSV_FILES_FOLDER}/urls.csv', header=None)
    all_download_links_1 = df_1[0].values.tolist()[:DATA_AMOUNT]
    article_file_urls = list(set(all_download_links_1))
    for file_format in ['pdf', 'xml']:
        download_folder = f'{DOWNLOAD_FOLDER}/{file_format}/'
        _skipped_amount = fast_download_files(article_file_urls, download_folder, file_format, max_workers=NUM_WORKERS)
        print(f'Managed to download {len(article_file_urls) - _skipped_amount} / {len(article_file_urls)}.')


@app.cell
def _():
    download_pdf_xml_files(CSV_FILES_FOLDER)
    return


@app.cell(hide_code=True)
def extract_text():
    mo.md(r"""
    ## Extract text from files

    Now we have all the articles downloaded, in PDF and/or XML format. There is one more thing left to do: extracting the texts from the files.

    We will go through each article and if the full text is available in the XML file, we will extract text from there since it's cleaner. If not, then we try to extract relevant text data from the PDF file.

    Some background about reading text from a PDF file:
    - Table contents and table description texts are interpreted as text when going through a page of a PDF file. There are several Python libraries that use a sort of computer vision algorithm to detect a table in a PDF so that it could be excluded from the body text itself. However, after multiple attempts we didn't manage to get these approaches working efficiently enough (slow performance and adequate table detection).
    - Also figure description texts are interpreted as text and can mess with the flow of the body text.
    """)
    return


@app.function
def find_table_from_text(text, file_path):
    """
    Return text without a table.

    Check recursively whether a given text (pdf page) has <Table x.> in it,
    indicating that there is a table in the text that could be extracted.

    Return the remaining text before and after the table when no <Table x.> exist in the text.
    """
    word_amount_checker = 8
    # Check with regex if a text has text 'Table x.' in it.
    table_match = list(re.finditer(r"(?:\n|^)(?<!\S)Table \w?\d+\.", text))

    text_to_include = ""
    if not table_match:
        # If no <Table x.> found, return the text
        return text
    # Check all the matches found in the text
    for i_match in table_match:
        text_without_tables = ""
        preserve_text = text[: i_match.span()[0]]
        cut_text = text[i_match.span()[0] :]
        possible_body_text = 0
        possible_table_text = 0
        table_started = False
        table_extracted = False
        extracted_table = ""
        extracted_table_final = ""
        lines = cut_text.split("\n")
        possible_body_text_start = []
        first_line = True
        for line in lines:
            words = re.split(" |-|–", line)
            if first_line and len(words) > 1:
                table_header_wide = True if len(words) >= 15 else False
                word_amount_checker = 9 if table_header_wide else 7
                first_line = False

            if table_extracted:
                if len(words) > 20:
                    possible_body_text_start = []
                    continue
                # Once the table has been found, start adding the lines to form the rest of the text
                text_without_tables += f"{line}\n"
                continue
            if len(words) >= word_amount_checker:
                # If the amount of words in a line is higher than or equl to threshold (indicating a possible body text)
                if len(words) > 20 and table_header_wide and not re.match(r"(?:\n|^)(?<!\S)Table \w?\d+\.", line):
                    continue
                possible_body_text_start.append(line)
                possible_body_text += 1
                if possible_body_text == 3 and table_started:
                    extracted_table_final = extracted_table.split("\n")[:-3]
                    table_extracted = True
                    continue
                elif possible_body_text > 3:
                    if table_header_wide:
                        if possible_body_text == 6 and not table_started:
                            break
                    else:
                        if possible_body_text == 10 and not table_started:
                            break
                    continue

            elif len(words) < word_amount_checker:
                # If the amount of words in a line is lower than threshold (indicating table content)
                if possible_table_text == 2:
                    table_started = True
                possible_table_text += 1
                if possible_body_text > 0:
                    possible_body_text -= 1
                    possible_body_text_start = []
            extracted_table += " ".join(words) + "\n"

        if len(extracted_table_final) == 0 and table_started:
            extracted_table_final = extracted_table.split("\n")
            table_extracted = True
        if table_extracted:
            # Recursively start another search for <Table x.>
            text_to_include += f"{preserve_text}\n" + "\n".join(possible_body_text_start) + f"\n{text_without_tables}"
            return find_table_from_text(text_to_include, file_path)
        if not table_started:
            # Recursively start another search for <Table x.>
            # This if statement is for situations where there might be <Table x.> in the text,
            # but it wasn't actually a table. Here the start of the text is cropped to exclude the wrong <Table x.>.
            return find_table_from_text(cut_text[5:], file_path)


@app.function
def find_tables(text, file_path):
    """
    Return text without tables.

    Extract table(s) from scientific journals from Copernicus.

    This only works "correctly", if the "table heading"/description is before the actual table.

    Has some flaws: in some cases, the code extracts some body text after the table.
    In few cases (where the text lines inside the table are long) the table might be cropped halfway.
    This method calls a function, that recursively checks if a given text (page of a pdf) has tables in them.

    Returns:
    - A string that is the text excluding table(s) and it's (their) contents.

    """
    any_tables = re.findall(r"(?:\n|^)(?<!\S)Table \w?\d+\.", text)
    if any_tables:
        return find_table_from_text(text, file_path)


@app.function
def find_pages_with_little_text(text):
    """
    Return empty string if page has under 10 lines of text.

    10 lines of text indicate that the page consists of only figures and their corresponding texts.

    Returns:
    - An empty string if page has under 10 lines of text.

    """
    lines = text.split("\n")
    if len(lines) <= 12:
        return True
    return False


@app.function
def regex_check_for_end(text):
    """
    Check if certain parts (Code (and) Data availability, Acknowledgements, Author contributions, References or Appendix) are in the extracted text.

    Returns:
        - The first regex match if certain part is found, else None

    """
    last_part = re.findall(
        r"(?:\n|^)Appendix ?\S*(?:\n|\:)|(?:\n|^)Acknowledgements\.(?:\s*\S*)|(?:\n|^)Author contributions\.(?:\s*\S*)|(?:\n|^)References\s*\n|(?:\n|^)\d?\s?(?:Data|Code|Code and data) availability\.?(?:\s*\S*)",
        text,
    )

    if last_part:
        last_part_match = re.finditer(
            r"(?:\n|^)Appendix ?\S*(?:\n|\:)|(?:\n|^)Acknowledgements\.(?:\s*\S*)|(?:\n|^)Author contributions\.(?:\s*\S*)|(?:\n|^)References\s*\n|(?:\n|^)\d?\s?(?:Data|Code|Code and data) availability\.?(?:\s*\S*)",
            text,
        )
        match = list(last_part_match)[0]
        return match
    return None


@app.function
def extract_pdf_text_without_headers_footers(pdf_path):
    filename = os.path.basename(pdf_path).split('.')[0]
    # Open the PDF file
    doc = pymupdf.open(pdf_path)

    # Initialize an empty string to store the extracted text
    extracted_text = ""
    abstract_found = False

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load each page

        # Get page dimensions
        width, height = page.rect.width, page.rect.height

        # Define header and footer heights
        if page_num == 0:  # First page
            header_height = 100
        else:  # Subsequent pages
            header_height = 50

        footer_height = 50

        # Define the rectangle to exclude header and footer (we want the main body)
        body_rect = pymupdf.Rect(0, header_height, width, height - footer_height)

        # Extract text within the defined rectangle (excluding header and footer)
        text = page.get_text("text", clip=body_rect)

        if abstract_found:
            if find_pages_with_little_text(text):
                continue
            text_without_tables = find_tables(text, filename)
            if text_without_tables:
                extracted_text += text_without_tables + "\n"
                end_match = regex_check_for_end(extracted_text)
                if end_match:
                    extracted_text = extracted_text[: end_match.span()[0]]
                    break
                continue
            end_match = regex_check_for_end(text)
            if end_match:
                extracted_text += text[: end_match.span()[0]]
                break
            # If last part is not found, continue adding text
            extracted_text += text + "\n"

        elif not abstract_found:
            any_abstract = re.findall(r"(?:\n|^)Abstract(\:|\.)?\s?\S*", text)
            if any_abstract:
                abstract_found = True
                abstract_match = list(re.finditer(r"(?:\n|^)Abstract(\:|\.)?\s?", text))[0]
                extracted_text += text[abstract_match.span()[1] :]
                continue

    extracted_text = re.sub("ﬂ", "fl", extracted_text)
    extracted_text = re.sub("ﬁ", "fi", extracted_text)

    # Return the dictionary with filename and text
    return {"filename": filename, "text": extracted_text}


@app.function
def extract_abstract_from_xml(xml_path):
    filename = os.path.basename(xml_path).split('.')[0]
    with open(xml_path, 'r', encoding='utf-8') as file:
        try:
            soup = BeautifulSoup(file, 'xml')
            abstract = soup.find('abstract')
            if abstract:
                abstract_text = abstract.find('p').get_text(strip=True)
                return {"filename": filename, "abstract_text": abstract_text}
            return {"filename": filename, "abstract_text": ""}
        except Exception as e:
            print(f"Error in abstract extraction from xml, {filename}: {e}")
            return {"filename": filename, "abstract_text": ""}


@app.function
def extract_body_text_from_xml(xml_path):
    filename = os.path.basename(xml_path).split('.')[0]
    with open(xml_path, 'r', encoding='utf-8') as file:
        try:
            soup = BeautifulSoup(file, 'xml')
            body = soup.find('body')
            if body:
                for tag in body.find_all(['fig', 'table-wrap']):
                    tag.decompose()
                paragraphs = body.find_all('p')
                text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
                return {"filename": filename, "text": text}
            return {"filename": filename, "text": ""}
        except Exception as e:
            print(f"Error in body text extraction from xml, {filename}: {e}")
            return {"filename": filename, "text": ""}


@app.function
def get_xml_pdf_files(folder_path) -> dict[list]:
    # Get all the files from the folders
    file_dict = defaultdict(list)

    pdf_folder = os.path.join(folder_path, "pdf")
    xml_folder = os.path.join(folder_path, "xml")

    # Add PDF files
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            name = os.path.splitext(filename)[0]
            full_path = os.path.join(pdf_folder, filename)
            file_dict[name].append(full_path)

    # Add XML files
    for filename in os.listdir(xml_folder):
        if filename.lower().endswith(".xml"):
            name = os.path.splitext(filename)[0]
            full_path = os.path.join(xml_folder, filename)
            file_dict[name].append(full_path)

    # Optional: convert to regular dict
    file_dict = dict(file_dict)

    return file_dict


@app.function
def extract_text_and_abstract(xml_pdf_item):
    # Initialize a list to store the results
    xml_path = None
    pdf_path = None

    article_name, paths = xml_pdf_item

    for path in paths:
        if path.lower().endswith(".xml"):
            xml_path = path
        elif path.lower().endswith(".pdf"):
            pdf_path = path

    if xml_path:
        abstract_from_xml = extract_abstract_from_xml(xml_path)
        text = extract_body_text_from_xml(xml_path)
        if len(text["text"]) < 1:
            if pdf_path:
                text = extract_pdf_text_without_headers_footers(pdf_path)
                return {**abstract_from_xml, **text, "text_source": "pdf"}
        return {**abstract_from_xml, **text, "text_source": "xml"}
    elif pdf_path:
        text = extract_pdf_text_without_headers_footers(pdf_path)
        return {**text, "abstract_text": "", "text_source": "pdf"}

    return {"filename": article_name, "abstract_text": "", "text": "", "text_source": None}


@app.function
def extract_text_from_files_in_folder(folder_path):
    # Initialize a list to store the results
    results = []

    files = get_xml_pdf_files(folder_path)

    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(extract_text_and_abstract, item): item for item in files.items()}
        for future in tqdm(as_completed(future_to_file), total=len(files)):
            result = future.result()
            results.append(result)

    return results


@app.function
def open_extracted_texts_json(json_file_path):
    with open(json_file_path, 'r') as file:
        return json.load(file)


@app.function
def save_texts_as_json(texts, output_file, update_existing=False):
    # Save the extracted texts as a JSON file
    if update_existing:
        i = 0
        data = open_extracted_texts_json(output_file)
        abstracts = {item["filename"].split(".")[0]: item["abstract_text"] for item in texts}
        for item in data:
            filename = item["filename"]
            if filename in abstracts:
                item["abstract"] = abstracts[filename]
                if i == 0:
                    print(item)
                    i += 1
            else:
                item["abstract"] = ""

        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    else:
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(texts, json_file, ensure_ascii=False, indent=4)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    Here we run the code itself. Extracting the text from the files doesn't take long!
    """)
    return


@app.function
def main(remove_files:bool = False):
    start = time.time()

    results = extract_text_from_files_in_folder(DOWNLOAD_FOLDER)

    save_texts_as_json(results, JSON_FILE_PATH)

    duration = time.time() - start

    print(f"Extracted texts saved to {JSON_FILE_PATH}")
    print(f"Took {duration:.2f} seconds")
    if remove_files:
        shutil.rmtree(path=DOWNLOAD_FOLDER)


@app.cell
def _():
    main(remove_files=REMOVE_FILES)
    return


@app.function
def sbatch_main():
    journal_urls = get_journals(base_url="https://publications.copernicus.org/open-access_journals/journals_by_subject.html")
    all_download_links, _, _ = get_articles(journal_urls, base_url="https://publications.copernicus.org/open-access_journals/journals_by_subject.html")
    save_article_links(CSV_FILES_FOLDER, all_download_links)
    download_pdf_xml_files(CSV_FILES_FOLDER)
    main(remove_files=REMOVE_FILES)


if __name__ == "__main__":
    app.run()
