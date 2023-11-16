from utils.parser import ResponseInstruct, OutputParser, _costum_json_parser
import requests
import re
import time
import openai
import json
import streamlit as st
import pandas as pd
from streamlit.logger import get_logger
from hold import *
from st_cytoscape import cytoscape
logger = get_logger(__name__)

st.set_page_config(page_title="Search Scholarly: Find helpful sources", page_icon="💝", layout="centered", initial_sidebar_state="auto")
st.sidebar.header("💝 Search Scholarly: Find helpful sources")



S2_API_KEY = st.secrets["S2_API_KEY"]
st.cache_resource(ttl=60*20)
def get_graph(df):
    MyCorpus = ResearchCorpus(df['title_abstract'].tolist(), df['seed'].tolist(), df.to_dict('records'))
    df = MyCorpus._make_df_cluster_topics()
    graphs, df = MyCorpus.make_graphs(df)
    return graphs, df

    
def find_recommendations(paper, _S2_API_KEY=S2_API_KEY):
    result_limit = st.session_state['result_limit']
    print(f"Up to {result_limit} recommendations based on: {paper['title']}")
    rsp = requests.get(f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper['paperId']}",
                       headers={'X-API-KEY': _S2_API_KEY},
                       params={'fields': 'title,url,authors,year,abstract,isOpenAccess', 'from':'all-cs'})
    rsp.raise_for_status()
    results = rsp.json()
    return results['recommendedPapers']

    
st.cache_data(ttl=60*60*24*7)
def make_df(recommendations, seed_paper):
    result_limit = st.session_state['result_limit']
    seed_paper['seed'] = True
    recommendations.append(seed_paper)
    df = pd.DataFrame(recommendations)
    
    df['seed'] = df['seed'].fillna(False)
    df['authors'] = df['authors'].apply(lambda x: ', '.join([author['name'] for author in x]))

    #replace non abstracts; make a new title_abstract column
    df = df[~df['abstract'].isna()]
    df['title_abstract'] = df['title'] + '[SEP]' + df['abstract']
    df.sort_values('seed', ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df = df.iloc[:result_limit+1]

    return df
tab1, tab2 = st.tabs(['Search for papers', 'View graph'])




if "paper_results" not in st.session_state:
    st.session_state.paper_results = False
if "current_paper" not in st.session_state:
    st.session_state.current_paper = None
if "seed_papers" not in st.session_state:
    st.session_state.seed_papers = []
if "graph" not in st.session_state:
    st.session_state.graph = False

selection = None
with tab1:
    result_limit = st.slider("Number of results to return", min_value=1, max_value=20, value=10, step=1)
    if not result_limit:
        st.stop()
        st.error("Please select a number of results to return")
    st.session_state['result_limit'] = result_limit
    with st.form('add_query'):
        query = st.text_input("Enter a query to search for papers")
        submit = st.form_submit_button('Search')

if submit:
    st.session_state.current_paper = None
    st.session_state.graph = False
    submit = False
    selection = None
    with tab1:
        with st.spinner("Searching for relevant sources... ⏳"):
            query = "+".join(query.split(' '))
            rsp = requests.get('https://api.semanticscholar.org/graph/v1/paper/search',
                        headers={'X-API-KEY': S2_API_KEY},
                        params={'query': query, 'limit': result_limit, 'fields': 'title,url,authors,year,abstract'})
            rsp.raise_for_status()
            results = rsp.json()
            total = results["total"]
            if not total:
                st.error("No matches found. Please try another query.")
                st.stop()
            else:
                st.markdown(f'Found {total} results. Showing up to {result_limit}.')
                list = ''
                papers = results['data']
                for idx, paper in enumerate(papers):
                    list += f"{idx}. [{paper['title']}]({paper['url']})\n"
                st.markdown(list)
                st.session_state.paper_results = True
                st.session_state.papers = papers
            
if st.session_state.paper_results:  
    # Select a paper to base recommendations by index number (result_limit is the max index)
    with tab1:
        selection = st.selectbox('Select a paper # to base recommendations on', (str(i) for i in range(result_limit)), index=None, placeholder='Select a paper #')

if selection is not None:
    st.session_state.current_paper = selection

if st.session_state.current_paper is not None and not st.session_state.graph:
    selection = st.session_state.current_paper
    papers = st.session_state.papers
    basis_paper = papers[int(selection)]
    recommendations = find_recommendations(basis_paper)
    st.session_state.seed_papers.append({'seed': basis_paper, 'recommendations': recommendations})
    for seed in st.session_state.seed_papers:
        recommendation = seed['recommendations']
        seed_paper = seed['seed']
        with tab1:
            with st.spinner("Generating Knowledge graph based on selection... ⏳"):
                df = make_df(recommendation, seed_paper)
            
                graph, df = get_graph(df)
                nodes = graph['nodes']
                edges = graph['edges']
                elements = nodes + edges
                topics = df['topic'].unique().tolist()
                topic_name = df['Topic'].unique().tolist()
                topic_to_dict = {topic: df[df['Topic']==topic]['topic_color'].tolist()[0] for i, topic in enumerate(topic_name)}
                topic_to_dict['Seed paper'] = df[df['is_seed']==True]['topic_color'].tolist()[0]
                size_sort = sorted(topics, key=lambda x: len(df[df['topic']==x]), reverse=True)

                ids_by_topicsorted = []
                for topic in size_sort:
                    ids_by_topicsorted.extend(df[df['topic']==topic]['ID'].tolist())
                id_of_seed = df[df['is_seed']==True]['ID'].tolist()[0]
                ids_by_topicsorted.remove(id_of_seed)
                color_topics = {topic: np.random.randint(0, 255, 3).tolist() for topic in topics}
                layout = {"name": "grid", "animationDuration": 1}
                layout["nodeRepulsion"] = 5000
                layout["alignmentConstraint"] = [{"axis": "x", "offset": 0, "left": id_of_seed}]
                layout["relativePlacementConstraint"] = [{"top": id_of_seed, "bottom": ids_by_topicsorted[0]}]
                for i in range(len(ids_by_topicsorted)-1):
                    layout["relativePlacementConstraint"] = [{"top": id_of_seed, "bottom": ids_by_topicsorted[i+1]}]
                    layout["relativePlacementConstraint"].append({"top": ids_by_topicsorted[i], "bottom": ids_by_topicsorted[i+1]})
        st.write('Graph generated! 🎉 --> Click on the "View graph" tab to see the graph.')
        st.session_state.graph = True
        st.session_state.graph_data = {'elements': elements, 'layout': layout, 'df': df, 'topic_to_dict': topic_to_dict}
if st.session_state.graph:
    elements = st.session_state.graph_data['elements']
    layout = st.session_state.graph_data['layout']
    layout["name"] = 'grid'
    df = st.session_state.graph_data['df']
    topic_to_dict = st.session_state.graph_data['topic_to_dict']
    body ='''
                <ul>
                    <span><b class="fa fa-square" style="{topic_color}">⊛</b></span><span> {topic}</span>

                </ul>
                '''
    
    stylesheet = [
        {"selector": "node", "style": {"label": "data(id)", "width": 40, "height": 40, 'background-color': 'data(faveColor)'}},
        {"selector": "edge",
            "style": {
                    
                    "mid-target-arrow-color": 'data(faveColor)',
                    "mid-target-arrow-shape": "vee",
        
                    'opacity': 0.5,
                    'z-index': 7777
                }
        },
        {"selector":"node:selected", "style": {
        "text-valign":"center",
        "text-halign":"center",
        "border-color": "white",
        "overlay-opacity": 0.1,
        "overlay-width": "300px",
        "overlay-color": "gray",
        "text-max-height": 100,
        "font-size":"28px",
        "text-wrap": "wrap",
        "text-opacity": 1,
        "text-max-width": "300px",
        "z-index": 9999
        }
        },
        {"selector":"edge:selected", "style": {
        "overlay-color": "purple",
        "line-color": "white",
        "content": "data(weight)",
        "overlay-opacity": 0.1,
        "text-opacity": 1,
        "text-color": "black",
        "font-size":"15px",
        "z-index": 9999,
        }},
    ]
    with tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header('Graph')
            selected = cytoscape(
                elements,
                stylesheet,
                height="550px",
                width="80%",
                layout=layout,
                key="graph",
            )
        with col2:
            st.header('Topics')
            legend = ''
            # print 'Seed paper' first
            legend += body.format(topic='Seed paper', topic_color="color: {};font-size:'20px'".format(topic_to_dict['Seed paper']))
            for topic in topic_to_dict.keys():
                if topic == 'Seed paper':
                    continue
                color = topic_to_dict[topic]
                legend += body.format(topic=topic, topic_color="color: {};font-size:'20px'".format(color))
            st.markdown(legend, unsafe_allow_html=True)
    with st.sidebar:
        # quey data frame with selected["nodes"] - print title with url and abstract
        try: 
            selected_ids =  selected['nodes']
            md = '### Selected articles: \n'
            for id in selected_ids:
                selddf = df.query(f'ID == {int(id)}')
                title = selddf['title'].tolist()[0]
                url = selddf['url'].tolist()[0]
                abstract = selddf['abstract'].tolist()[0]
                year = selddf['year'].fillna('').tolist()[0]
                if year != '':
                    year = int(year) 
                else: year = 'Unknown'
                md += f'[*{id}*] [**{title}** ({year})]({url})\n'
                md += f'> {abstract}\n'
                md += '***\n'
            st.markdown(md)
        except:
            pass
