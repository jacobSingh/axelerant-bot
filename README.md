## To setup:

1. Go into the data directory and extract the file.  
This contains the scrape of your intranet.  A better way to do this is using the Confluence DocLoader with Langchain, but since I don't have access, I just scraped it.

2. pip install -r requirements.txt

3. Create .streamlit/secrets.toml.  Put the following in (with your values):
OPENAI_API_KEY="" 
PINECONE_API_KEY = ""
PINECONE_ENV = "us-west1-gcp-free"

4. Run axelerant_doc_loader.py.  Note this will wipe out the index and then read pages from data/axelerant_pages.txt

## To Run:

`streamlit run axelerant_sl.py`

Should work fine locally, to deploy, use the streamlit website to copy from your fork.