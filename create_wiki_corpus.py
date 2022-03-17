import csv
import sys
import re
from datetime import datetime
from typing import IO, Iterable

import nltk
import wikitextparser as wtp
import mwparserfromhell as mwp
from mediawiki import MediaWiki
from mediawiki.exceptions import PageError, DisambiguationError
from ratelimit import limits, sleep_and_retry

"""
Script to pull text from randomized pages in three wikis and generate labeled sentence-level data 
suitable for classification. Uses a wrapper around the MediaWiki API.
Rate limits to avoid overwhelming MediaWiki end points on Wiki sites.  
"""

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"\
    "Chrome/98.0.4758.80 Safari/537.36"

wikis = {
    "conservapedia": MediaWiki("https://www.conservapedia.com/api.php", user_agent=user_agent),
    "metapedia": MediaWiki("https://en.metapedia.org/m/api.php"),
    "wikipedia": MediaWiki(user_agent=user_agent),  # default is wikipedia
    "powerbase": MediaWiki("https://powerbase.info/api.php")
}

MAX_CALLS_PER_MINUTE = 45
ONE_MINUTE = 60


@sleep_and_retry
@limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
def get_page_text(wiki: MediaWiki, name: str, title: str) -> str:
    """
    Extract plain text from wiki page
    :param wiki: MediaWiki object
    :param name: Name of wiki
    :param title: Page title to extract
    :return: Plain text
    """
    try:
        page = wiki.page(title)
    except (PageError, DisambiguationError) as e:
        # note error but continue with other pages
        print(f"Page {title} generates error on {name}")
        return None
    else:
        wiki_fied = page.wikitext
        if isinstance(wiki_fied, str):
            # return wtp.parse(wiki_fied).plain_text()
            return mwp.parse(wiki_fied).strip_code()
        elif isinstance(wiki_fied, dict):
            # return wtp.parse(wiki_fied["*"]).plain_text()
            return mwp.parse(wiki_fied["*"]).strip_code()
        else:
            raise RuntimeError(f"Unexpected type {type(wiki_fied)} for title {title} on {name}")


def write_to_corpus(name: str, file: IO[str], parser: Iterable) -> None:
    """
    Write raw_text to file in csv format
    :param name: Name of wiki for purpose of labelling
    :param file: File object
    :param parser: Parser that will generate text to be written
    :return: None
    """
    writer = csv.writer(file, delimiter=",", quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
    count = 0
    for sentence in parser:
        writer.writerow([
            datetime.now().strftime('%Y%m%d%H%M%S%f'),
            name,
            sentence
        ])
        count += 1
    return count


def normalize(text: str) -> str:
    """
    Perform some basic text normalization
    :param text: text to normalize
    :return: normalized text
    """
    text = text.lower()
    text = re.sub(r"category:[\w+| ]+", "", text)
    text = re.sub(r"==[\w+| ]+==", " ", text)
    text = re.sub(r"^thumb\|.*", "", text)
    text = re.sub(r"external links.*", "", text)
    text = re.sub(r"^encyclopedias .*", "", text)
    text = re.sub(r"encyclopedia britannica\: .*", "", text)
    text = re.sub(r"encyclopedia.com\: .*", "", text)
    text = re.sub(r"isbn [\d+|-]+.", "", text)
    text = re.sub(r"metapedia", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


if __name__ == '__main__':

    num_pages = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    timestamp = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}"
    corpus_file = f"./data/wiki_corpus_{timestamp}.csv"

    with open(corpus_file, "w") as output_file:
        output_file.write("id,source,sentence\n")
        for wiki_name, media_wiki in wikis.items():
            sentence_count = 0
            print(f"Processing {num_pages} from {wiki_name}")
            page_titles = media_wiki.random(num_pages)
            print(f"Page titles ({len(page_titles)}: {page_titles}")
            for page_title in page_titles:
                print(f"Processing {page_title}")
                raw_text = get_page_text(media_wiki, wiki_name, page_title)
                text = normalize(raw_text)
                if raw_text:
                    sentence_count += write_to_corpus(
                        wiki_name,
                        output_file,
                        nltk.sent_tokenize(text)
                    )
            print(f"Wrote {sentence_count} sentences from {wiki_name}")
