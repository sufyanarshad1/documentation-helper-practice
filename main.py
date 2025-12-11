from backend.core import run_llm
import streamlit as st

st.header("Documentation Helper")


def create_sources_string(source_urls: set[str]) -> str:
    if not source_urls:
        return "No sources available."
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


prompt = st.text_input("Prompt", placeholder="Enter your question about LangChain:")


if (
    "user_prompt_history" not in st.session_state
    and "chat_answer_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state.user_prompt_history = []
    st.session_state.chat_answer_history = []
    st.session_state.chat_history = []


if prompt:
    with st.spinner("Generating response"):
        generated_response = run_llm(query=prompt,chat_history=st.session_state["chat_history"])
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        format_response = (
            f"{generated_response['answer']}\n\n{create_sources_string(sources)}"
        )

        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answer_history.append(format_response)
        st.session_state.chat_history.append(("human", prompt))
        st.session_state.chat_history.append(("ai", generated_response['answer']))

if st.session_state["chat_answer_history"]:
    for generated_response, user_query in zip(
        st.session_state["user_prompt_history"],
        st.session_state["chat_answer_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)