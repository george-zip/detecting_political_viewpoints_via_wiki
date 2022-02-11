from datetime import datetime
from typing import IO

import json
import nltk
import wikitextparser as wtp
from mediawiki import MediaWiki
from mediawiki.exceptions import PageError
from ratelimit import limits, sleep_and_retry

"""
Script to pull text from randomized pages in three wikis and generate labeled sentence-level data 
suitable for classification. Uses a wrapper around the MediaWiki API.
Rate limits to avoid overwhelming MediaWiki end points on Wiki sites.  
"""

wikis = {
    "conservapedia": MediaWiki("https://www.conservapedia.com/api.php"),
    "rational": MediaWiki("https://rationalwiki.org/w/api.php"),
    "wikipedia": MediaWiki()  # default is wikipedia
}

# some global settings: TODO: make some of these into command-line parameters
num_pages = 2
timestamp = f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}"
corpus_file = f"./data/wiki_corpus_{timestamp}.json"
max_calls_per_minute = 30
one_minute = 60


class Parser:
    """
    Parse text into lists of words
    """
    word_tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")

    def __init__(self, text):
        self.sentences = nltk.sent_tokenize(text.lower())

    def __iter__(self):
        for s in self.sentences:
            yield Parser.word_tokenizer.tokenize(s)


@sleep_and_retry
@limits(calls=max_calls_per_minute, period=one_minute)
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
    except PageError as e:
        # note error but continue with other pages
        print(f"Page {title} generates error on {name}")
        return None
    else:
        wiki_fied = page.wikitext
        if isinstance(wiki_fied, str):
            return wtp.parse(wiki_fied).plain_text()
        elif isinstance(wiki_fied, dict):
            return wtp.parse(wiki_fied["*"]).plain_text()
        else:
            raise RuntimeError(f"Unexpected type {type(wiki_fied)} for title {title} on {name}")


def write_to_corpus(name: str, file: IO[str], parser: Parser) -> None:
    """
    Write raw_text to file in json format
    :param name: Name of wiki for purpose of labelling
    :param file: File object
    :param parser: Parser that will generate text to be written
    :return: None
    """
    timestamp = str(datetime.now())
    for sentence in parser:
        json_data = {
            "timestamp": timestamp,
            "id": id(sentence),
            "source": name,
            "sentence": sentence
        }
        json.dump(json_data, file)


with open(corpus_file, "w") as output_file:
    for wiki_name, media_wiki in wikis.items():
        print(f"Processing: {wiki_name}")
        page_titles = media_wiki.random(num_pages)
        print(f"Page titles: {page_titles}")
        for page_title in page_titles:
            raw_text = get_page_text(media_wiki, wiki_name, page_title)
            if raw_text:
                print(f"Got {len(raw_text)} characters from {wiki_name}")
                write_to_corpus(wiki_name, output_file, nltk.sent_tokenize(raw_text.lower()))
