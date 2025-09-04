from typing import Dict, List, Optional

import json
import streamlit as st
import time
from functools import partial
from snowflake.cortex import complete
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.custom import instrument
from trulens.providers.cortex.provider import Cortex
from trulens.core import Feedback, SnowflakeFeedback, Select
from trulens.core.feedback import feedback as core_feedback
from trulens.apps.custom import TruCustomApp
from snowflake.snowpark import Session

import json
from logging import getLogger
from typing import Any

import requests
import snowflake.connector
import streamlit as st

HOST = st.secrets["connections"]["snowflake"]["host"]
DATABASE = st.secrets["connections"]["snowflake"]["database"]
SCHEMA = st.secrets["connections"]["snowflake"]["schema"]
STAGE = st.secrets["connections"]["snowflake"]["stage"]
FILE = st.secrets["connections"]["snowflake"]["semantic_context_file"]

import requests
SNOWFLAKE_ACCOUNT_URL = f"https://{HOST}"
SNOWFLAKE_PAT = st.secrets["connections"]["snowflake"]["pat"]
API_ENDPOINT = "/api/v2/cortex/analyst/message"

# Create a single session that will be reused everywhere
@st.cache_resource
def create_snowpark_session():
    """Create a single Snowpark session to be shared across the application"""
    connection_parameters = {
        "account": st.secrets["connections"]["snowflake"]["account"],
        "user": st.secrets["connections"]["snowflake"]["user"],
        "password": st.secrets["connections"]["snowflake"]["password"],
        "role": st.secrets["connections"]["snowflake"].get("role"),
        "warehouse": st.secrets["connections"]["snowflake"].get("warehouse"),
        "database": DATABASE,
        "schema": SCHEMA
    }
    
    # Remove None values
    connection_parameters = {k: v for k, v in connection_parameters.items() if v is not None}
    
    session = Session.builder.configs(connection_parameters).create()
    st.write("Created shared Snowpark session")
    return session

# Get the shared session
snowpark_session = create_snowpark_session()

@st.cache_resource
def create_tru_session():
    """Create TruSession using the shared Snowpark session"""
    conn = SnowflakeConnector(snowpark_session=snowpark_session)
    tru_session = TruSession(connector=conn)
    # Define llm for eval - use the same session
    provider = Cortex(snowpark_session, "mistral-large2")
    return tru_session, provider

# Initialize TruLens with shared session
tru_session, provider = create_tru_session()

SUMMARIZATION_LLM = st.sidebar.selectbox('Select your Summarization LLM:',(
                "mistral-large2",
                "openai-gpt-4.1",
                "llama3.3-70b",
                "llama3.1-70b",
                "llama4-maverick",
                "llama4-scout",   
                "claude-3-5-sonnet",
                "gemma-7b",
                "jamba-1.5-mini",
                "jamba-1.5-large",
                "jamba-instruct",
                "llama2-70b-chat",
                "llama3-8b",
                "llama3-70b",
                "llama3.1-8b",
                "llama3.1-405b",
                "llama3.2-1b",
                "llama3.2-3b",
                "snowflake-llama3.3-70b",
                "mistral-large",
                "mistral-large2",
                "mistral-7b",
                "mixtral-8x7b",
                "reka-core",
                "reka-flash",
                "snowflake-arctic",
                "snowflake-llama-3.1-405b"), key="model_name")

# Final Answer Relevance - How well does the final summarization answer the users initial prompt?
final_answer_relevance = (
            Feedback(provider.relevance_with_cot_reasons, name = "Final Answer Relevance")
            .on_input()
            .on_output())

# Interpretation Accuracy - How accurately is cortex analyst inpreting the users prompt?
interpretation_criteria = """Provided is a user query as input 1 and an LLM generated interpretation 
                            of the users query as input 2. Grade how strong the interpretation was and explain
                            any discrepencies in the interpretation."""

interpretation_accuracy = (
                Feedback(provider.relevance_with_cot_reasons,
                name="Interpretation Accuracy",
                criteria = interpretation_criteria)
                .on_input()
                .on(Select.RecordCalls.call_analyst_api.rets['message']['content'][0]['text']))

# SQL Relevance - How relevant is the generated SQL to the users prompt?
sql_gen_criteria = """Provided is an interpretation of a user's query as input 1 and an LLM generated SQL query 
                    designed to answer the users query as input 2. Grade how relevant the SQL code 
                    appears to be to the user's question."""

sql_relevance = (
            Feedback(provider.relevance_with_cot_reasons, 
            name = "SQL relevance",
            criteria = sql_gen_criteria)
            .on(Select.RecordCalls.call_analyst_api.rets['message']['content'][0]['text'])
            .on(Select.RecordCalls.call_analyst_api.rets['message']['content'][1]['statement']))

#Summarization Groundedness -  How well grounded in the sql results is the summarization
groundedness_configs = core_feedback.GroundednessConfigs(use_sent_tokenize=False, 
                                                         filter_trivial_statements=False)

summarization_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons,
        name="Summarization Groundedness"
    )
    .on_input()
    .on_output()
)

feedback_list = [interpretation_accuracy, sql_relevance, final_answer_relevance, summarization_groundedness]

class CortexAnalyst():
    def __init__(self):
        """Initialize with the shared session"""
        self.session = snowpark_session
    
    @instrument
    def call_analyst_api(self, prompt: str) -> dict:
        """Calls the REST API and returns the response."""
        request_body = {
            "messages": st.session_state.messages,
            "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        }

        API_HEADERS = {
            "Authorization": f"Bearer {SNOWFLAKE_PAT}",
            "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
            "Content-Type": "application/json",
        }
        
        resp = requests.post(
            f"{SNOWFLAKE_ACCOUNT_URL}{API_ENDPOINT}",
            json=request_body,
            headers=API_HEADERS,
            timeout=30000,
        )
        request_id = resp.headers.get("X-Snowflake-Request-Id")
        
        if resp.status_code < 400:
            return {**resp.json(), "request_id": request_id}
        else:
            st.session_state.messages.pop()
            raise Exception(
                f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.text}"
            )

    @instrument
    def process_api_response(self, prompt: str) -> str:
        """Processes a message and adds the response to the chat."""
        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = self.call_analyst_api(prompt=prompt)
                request_id = response["request_id"]
                content = response["message"]["content"]
                st.session_state.messages.append(
                    {**response['message'], "request_id": request_id}
                )
                final_return = self.process_sql(content=content, request_id=request_id)
                
        return final_return
        
    @instrument
    def process_sql(self,
        content: List[Dict[str, str]],
        request_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ) -> str:
        """Displays a content item for a message."""
        message_index = message_index or len(st.session_state.messages)
        sql_markdown = 'No SQL returned!'
        if request_id:
            with st.expander("Request ID", expanded=False):
                st.markdown(request_id)
        for item in content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "suggestions":
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(item["suggestions"]):
                        if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                            st.session_state.active_suggestion = suggestion
            elif item["type"] == "sql":
                sql_markdown = self.execute_sql(sql = item["statement"])

        return sql_markdown

    @instrument
    def execute_sql(self, sql: str) -> str:
        with st.expander("SQL Query", expanded=False):
            st.code(sql, language="sql")
        with st.expander("Results", expanded=True):
            with st.spinner("Running SQL..."):
                # Use the instance's session reference
                df = self.session.sql(sql).to_pandas()
                if len(df.index) > 1:
                    data_tab, line_tab, bar_tab = st.tabs(
                        ["Data", "Line Chart", "Bar Chart"]
                    )
                    data_tab.dataframe(df)
                    if len(df.columns) > 1:
                        df = df.set_index(df.columns[0])
                    with line_tab:
                        st.line_chart(df)
                    with bar_tab:
                        st.bar_chart(df)
                else:
                    st.dataframe(df)

        return df.to_markdown(index=True)

    @instrument
    def summarize_sql_results(self, prompt: str) -> str:
        sql_result = self.process_api_response(prompt)
        st.write(f"Summarizing result using {SUMMARIZATION_LLM}...")
        # Use the session explicitly for Cortex complete function
        summarized_result = complete(SUMMARIZATION_LLM, 
                                     f'''Summarize the following input prompt and corresponding SQL result 
                                     from markdown into a succint human readable summary. 
                                     Original prompt - {prompt}
                                     Sql result markdown - {sql_result}''',
                                     session=self.session)
        st.write(f"**{summarized_result}**")
        helper = self.helper_function(prompt)
        return summarized_result

    @instrument
    def helper_function(self, prompt: str) -> dict:
        helper_dict = {}
        helper_dict['interpretation'] = f"Interpret or clarify the following prompt: {prompt}"
        helper_dict['sql_gen'] = f"Create sql that would be appropriate to answer the input prompt - {prompt}"
        return helper_dict

# Instantiate class
CA = CortexAnalyst()

TRULENS_APP_NAME = "CORTEX_ANALYST_APP"
TRULENS_APP_VERSION = SUMMARIZATION_LLM

# CREATE TRULENS APP WITH CA instance
tru_app = TruCustomApp(
    CA,
    app_id= TRULENS_APP_NAME,
    app_version=TRULENS_APP_VERSION,
    feedbacks=feedback_list
)

def show_conversation_history() -> None:
    for message_index, message in enumerate(st.session_state.messages):
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
               try:
                   CA.process_sql(
                        content=message["content"],
                        request_id=message.get("request_id"),
                        message_index=message_index,
                    )
               except: 
                   st.write("No history found!")

def reset() -> None:
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.active_suggestion = None

# Initialize session state
if "messages" not in st.session_state:
    reset()

st.title(f":snowflake: Text to SQL Assistant with Snowflake Cortex :snowflake:")
st.markdown(f"Semantic Model: `{FILE}`")

with st.sidebar:
    if st.button("Reset conversation"):
        reset()

show_conversation_history()

if user_input := st.chat_input("What is your question?"):
    # Test the pipeline with TruLens
    with tru_app as recording:
        recording.record_metadata = ({"Semantic_Model_File": FILE,
                                   "Summarization_LLM": SUMMARIZATION_LLM})
        CA.summarize_sql_results(prompt=user_input)
    
if st.session_state.active_suggestion:
    CA.process_api_response(prompt=st.session_state.active_suggestion)
    st.session_state.active_suggestion = None