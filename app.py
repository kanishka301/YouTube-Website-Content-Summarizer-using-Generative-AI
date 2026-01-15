import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter



st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0
)

prompt = PromptTemplate(
    template="""
Provide a clear and concise summary of the following content
in about 300 words.

Content:
{text}
""",
    input_variables=["text"]
)

if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Loading & summarizing..."):

                # Load content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(
                            generic_url,
                            add_video_info=False,   # 🔥 IMPORTANT
                            language=["en"]
                        )
                        documents = loader.load()
                    except Exception:
                        st.error(
                            "❌ Unable to fetch transcript.\n\n"
                            "This video may not have captions enabled."
                        )
                        st.stop()


                documents = loader.load()

                # Split documents (important for long content)
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200,
                    chunk_overlap=150
                )
                docs = splitter.split_documents(documents)

                # LCEL summarization chain
                chain = prompt | llm | StrOutputParser()

                summaries = [
                    chain.invoke({"text": doc.page_content})
                    for doc in docs
                ]

                final_summary = "\n\n".join(summaries)

                st.success(final_summary)

        except Exception as e:
            st.exception(e)
