# Scholarly - Streamlit Demo App
**Note:** The app's a work in progress
## Overview
A tool that helps  explore research papers. It uses:
* [Semantic Scholar API](https://www.semanticscholar.org/product/api) to search for papers
* Clustering to group similar papers (see ResearchCorpus.py)
* Keyword extraction + Llama to generate topic names 
* cytoscape.js for easy visualization and selection of interesting papers 
* [Streamlit](https://www.streamlit.io/) for the UI

  [Click here to test on streamlit](https://scholarly.streamlit.app/)

## Current Features
- **Article Search:** Utilize the Semantic Scholar API to search for scholarly articles.
- **Paper Recommendation:** Retrieve N recommended papers based on the selected seed paper.
- **Text Embeddings & Clustering:** Leverage ResearchCorpus to generate text embeddings, cluster articles, and assign topics using LLama.
- **Visualization with Cytoscape:** Display the articles and clusters in an interactive graph format for easy exploration.
- **Article Selection and Download:** Allow users to select articles of interest and download a list containing titles and abstracts.

## Features in Development
- **Open Access PDF Search:** [TO DO] Implement an option to search and download open-access PDFs of the articles.
- **Optional AI Summarization:** [TO DO] Incorporate an AI-powered summarization feature for articles.

## To-Do List
- [ ] **Implement 'Open Access PDF Search':** Enable users to search for and download open-access PDFs of articles directly within the app.
- [ ] **Integrate Optional AI Summarization:** Implement an optional feature for AI-generated summarization of articles.
- [ ] **Improve Clustering Visualization:** Improve the visualization of clusters in the Cytoscape graph - e.g. adjust size and distance between nodes (closer nodes should be more similar while bigger more central nodes should be more important given query used to generate the graph).


---

