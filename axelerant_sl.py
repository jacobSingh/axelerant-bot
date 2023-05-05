"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
import pinecone

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_ENV']  # next to api key in console
)

EMBEDDING = OpenAIEmbeddings()
PINECONE_INDEX_NAME = "axelerant-oa"


from langchain.chains import ConversationChain
from langchain.llms import OpenAI


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

#chain = load_chain()
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    

    docs = docsearch.similarity_search(query)
    output = docs[0].text
    
    #output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")