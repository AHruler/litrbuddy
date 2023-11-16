import os
import json
import argparse
import ntpath
import io
import time
from bs4 import BeautifulSoup, NavigableString
from typing import Optional, Dict, Union, Any, List
from pydantic.v1 import BaseModel
import hashlib
from io import StringIO

from review.utils.gorbid_client import GrobidClient
from review.utils.xml_to_json import convert_article_soup_to_dict
import tempfile

from langchain.docstore.document import Document

BASE_TEMP_DIR = 'temp'
BASE_OUTPUT_DIR = 'output'
BASE_LOG_DIR = 'log'
article_keys = ['title', 'authors', 'pub_date', 'abstract', 'sections', 'references', 'doi']
ref_keys = ['title', 'journal', 'year', 'authors']

  
class ParsedWithSources(BaseModel):
    title_abstracts: List[str]
    file_dict: Dict

def upload_pdf(paths, test=False, ref=True) -> ParsedWithSources:
    file_dict = {}
    # temp_dir = tempfile.TemporaryDirectory()
    for path in paths:
        if isinstance(path, str):
            with open(path, 'rb') as f:
                pdf_content = f.read()
            filename = ntpath.basename(path)
        else:
            try:
                pdf_content = path.read()
                filename = path.name
            except:
                print("Error reading file stream: ", path)
                continue
            
        # # compute hash
        pdf_hash = hashlib.sha1(pdf_content).hexdigest()
        # get Grobid -> TEI.XML
        if test and ref:
            article = process_pdf_stream(filename, pdf_hash, pdf_content, test=True, ref=True)
        elif test:
            article = process_pdf_stream(filename, pdf_hash, pdf_content, test=True)
        elif ref:
            article = process_pdf_stream(filename, pdf_hash, pdf_content, ref=True)
        else:
            article = process_pdf_stream(filename, pdf_hash, pdf_content)
        
        if 'title' in article:
            file_dict[article['title']] = article
        else:
            if isinstance(path, str):
                file_dict[path] = article

    # keep headings between first and where heading has 'CONCLUSION' or 'DISCUSSION' with or withou caps
    def get_conclusion_index(headings):
        for i, h in enumerate(headings):
            if 'CONCLUSION' in h or 'DISCUSSION' in h or 'conclusion' in h or 'discussion' in h or 'Conclusion' in h or 'Discussion' in h:
                return i
        return len(headings) - 1

    def filter_section_dict(file_dict, titles):
        headings = [[s['heading'] for s in file_dict[i]['sections']] for i in titles.values()]
        intro_to_conlusion = [headings[i][:get_conclusion_index(headings[i]) + 1] for i in range(len(headings))]
        # keep only sections with headings in intro_to_conclusion in file_dict[i]['sections']
        new_sections = [[s for s in file_dict[t]['sections'] if s['heading'] in intro_to_conlusion[i]] for i, t in enumerate(titles.values())]
        left_out_sections = [[s for s in file_dict[t]['sections'] if s['heading'] not in intro_to_conlusion[i]] for i, t in enumerate(titles.values())]
        for i, t in enumerate(titles.values()):
            file_dict[t]['sections'] = new_sections[i]
            file_dict[t]['left_out_sections'] = left_out_sections[i]
        return file_dict
    
    titles = {i: t for i, t in enumerate(list(file_dict.keys()))}
    len_abstracts = [len(file_dict[title]['abstract']) for title in titles.values()]
    # if len abstract is 0, replace with file_dict[title]['sections'][0]['text']
    for i, l in enumerate(len_abstracts):
        if l < 10:
            title = titles[i]
            try:
                file_dict[title]['abstract'] = file_dict[title]['sections'][0]['text'] if len(file_dict[title]['sections'][0]['text']) > 10 else ''
            except:
                file_dict[title]['abstract'] = 'None'
    title_abstracts = [title + '[SEP]' + file_dict[title]['abstract'] for i, title in titles.items()]
 
    return ParsedWithSources(title_abstracts=title_abstracts, file_dict=filter_section_dict(file_dict, titles))

def process_pdf_stream(input_file: Union[str, List], sha: str, input_stream: bytes, grobid_config: Optional[Dict] = None, 
                       ref=True, test=False) -> Dict:
    """
    Process PDF stream
    :param input_file:
    :param sha:
    :param input_stream:
    :return:
    """

    
    # process PDF through Grobid -> TEI.XML
    if ref:
        grobid_config = "ref"
    
    if test:
        client = GrobidClient(config=grobid_config, test=True)
    else:
        client = GrobidClient(config=grobid_config)
    
    tei_text = client.process_pdf_stream(input_file, input_stream, 'temp', "processFulltextDocument")
    
    content = "".join(tei_text)
    bs_content = BeautifulSoup(content, "lxml")
    article = convert_article_soup_to_dict(bs_content)


    return article


def process_pdf_file(
        input_file: str,
        temp_dir: str = BASE_TEMP_DIR,
        output_dir: str = BASE_OUTPUT_DIR,
        grobid_config: Optional[Dict] = None,
        ref: bool = False,
        test: bool = False
) -> str:
    """
    Process a PDF file and get JSON representation
    :param input_file:
    :param temp_dir:
    :param output_dir:
    :return:
    """
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # get paper id as the name of the file
    paper_id = '.'.join(input_file.split('/')[-1].split('.')[:-1])
    tei_file = os.path.join(temp_dir, f'{paper_id}.tei.xml')
    output_file = os.path.join(output_dir, f'{paper_id}.json')

    # check if input file exists and output file doesn't
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} doesn't exist")
    if os.path.exists(output_file):
        print(f'{output_file} already exists!')

    # process PDF through Grobid -> TEI.XML
    if ref:
        grobid_config = "ref"
    
    if test:
        client = GrobidClient(config=grobid_config, test=True)
    else:
        client = GrobidClient(config=grobid_config)
    # TODO: compute PDF hash
    # TODO: add grobid version number to output
    client.process_pdf(input_file, temp_dir, "processFulltextDocument")

    # process TEI.XML -> JSON
    assert os.path.exists(tei_file)
    paper = convert_article_soup_to_dict(tei_file)

    # write to file
    with open(output_file, 'w') as outf:
        json.dump(paper.release_json(), outf, indent=4, sort_keys=False)

    return output_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run S2ORC PDF2JSON")
    parser.add_argument("-i", "--input", default=None, help="path to the input PDF file")
    parser.add_argument("-t", "--temp", default=BASE_TEMP_DIR, help="path to the temp dir for putting tei xml files")
    parser.add_argument("-o", "--output", default=BASE_OUTPUT_DIR, help="path to the output dir for putting json files")
    parser.add_argument("-k", "--keep", action='store_true')

    args = parser.parse_args()

    input_path = args.input
    temp_path = args.temp
    output_path = args.output
    keep_temp = args.keep

    start_time = time.time()

    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    process_pdf_file(input_path, temp_path, output_path)

    runtime = round(time.time() - start_time, 3)
    print("runtime: %s seconds " % (runtime))
    print('done.')