import streamlit as st
import pandas as pd
import base64
import os
import asyncio

from langchain_openai import (
    ChatOpenAI,
)  # If you want to replicate the style of the existing app
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Import the EDA agent
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.plotly import plotly_from_dict

# OPTIONAL: If you need the same illusions of model selection as the original app:
MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]


def main():
    st.set_page_config(page_title="EDA Tools Agent", page_icon="📊")
    st.title("Exploratory Data Analysis (EDA) Agent")

    st.markdown("""
    This Streamlit app demonstrates how to interact with an EDA Agent that uses various tools to 
    describe datasets, visualize missing data, generate correlation funnels, and create Sweetviz reports.
    """)

    # Add an expander with example prompts
    with st.expander("Example Prompts"):
        st.write("""
        - "Describe the dataset"
        - "Visualize missing data with a sample of 100 rows"
        - "Generate a Sweetviz report"
        - "Create a correlation funnel with target='Churn'"
        """)

    # Sidebar
    st.sidebar.header("Settings")

    # For demonstration, mimic the approach of the SQL agent requiring an API key
    # (If your EDAToolsAgent doesn't need an LLM, you can skip this)
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Needed if your EDA agent uses LLM calls",
    )

    # Model selection
    model_option = st.sidebar.selectbox("Choose a model", MODEL_LIST, index=0)

    # 1) File uploader
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, Excel, etc.)", type=["csv", "xlsx", "xls", "parquet"]
    )
    if uploaded_file is not None:
        try:
            fname = uploaded_file.name.lower()
            if fname.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif fname.endswith(".xls") or fname.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif fname.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file type!")
                return
            st.session_state["data_raw"] = df

            st.write("## Dataset Preview")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("Please upload a dataset to begin.")
        st.stop()

    # 2) Create or load the EDA agent in session state
    if "eda_agent" not in st.session_state:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar to continue.")
            st.stop()
            
        llm = ChatOpenAI(model=model_option, api_key=api_key)
        st.session_state["eda_agent"] = EDAToolsAgent(model=llm)

    eda_agent = st.session_state["eda_agent"]

    # 3) Chat-like interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing conversation
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("ai").write(msg["content"])

    # 4) Provide an input for user instructions
    user_input = st.chat_input("Enter your EDA instruction or question:")
    if user_input:
        # Append user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # 5) Invoke the agent
        # If the agent supports async calls, we can do something like:
        # result = asyncio.run(eda_agent.ainvoke_agent(
        #     user_instructions=user_input,
        #     data_raw=st.session_state["data_raw"]
        # ))
        try:
            eda_agent.invoke_agent(
                user_instructions=user_input, data_raw=st.session_state["data_raw"]
            )
        except Exception as e:
            st.error(f"Agent error: {e}")
            return

        # 6) Display results
        response = (
            eda_agent.response
        )  # A dict with "messages", "internal_messages", and "eda_artifacts"

        # - The final AI message:
        ai_message = (
            response["messages"][0].content
            if response and "messages" in response
            else "No response"
        )
        st.session_state.messages.append({"role": "ai", "content": ai_message})
        st.chat_message("ai").write(ai_message)

        # - Artifacts (like images, HTML, etc.)
        artifacts = response.get("eda_artifacts", None)
        if artifacts:
            # If it's a dictionary, we loop over potential keys
            if isinstance(artifacts, dict):
                for key, val in artifacts.items():
                    # If val is base64, we can decode and show as image
                    # or if it's a plotly figure dict, convert to figure
                    if isinstance(val, str) and len(val) > 100:
                        # Attempt to interpret as base64 image
                        try:
                            img_bytes = base64.b64decode(val)
                            st.image(img_bytes, caption=f"Artifact: {key}")
                        except:
                            pass
                    if key == "report_html" and val:
                        # This is possibly the Sweetviz HTML
                        st.markdown("### Sweetviz Report")
                        # Option 1: Show an iframe with st.components.v1.html
                        st.components.v1.html(val, height=800, scrolling=True)
                        # Option 2: Save the HTML to a file and show a link
                    if key == "plotly_figure" and isinstance(val, dict):
                        try:
                            fig_obj = plotly_from_dict(val)
                            st.plotly_chart(fig_obj)
                        except:
                            st.warning("Unable to render Plotly figure.")

    # 7) Additional tips or usage instructions
    st.info(
        "Ask additional questions or try new commands. Examples: 'Show me missing data', 'Generate correlation funnel with target=Attrition'."
    )


if __name__ == "__main__":
    main()
