import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from dotenv import load_dotenv
import os
import urllib.parse
import pandas as pd
from yaml_gener import generate_yaml

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Chat with Your SQL Server Database", page_icon="🦜")
st.title("🦜Chat with Your SQL Server Database")

# Fetch credentials securely from environment variables
username = os.getenv('username')
password = os.getenv('password')
server_name = os.getenv('server_name')
db_name = os.getenv('db_name')

if not username or not password or not server_name or not db_name:
    st.error("Missing database credentials. Please check your `.env` file.")
    st.stop()

# Create ODBC connection parameters
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_name};"
    f"DATABASE={db_name};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30"
)

# Construct database connection URI
db_uri = f"mssql+pyodbc:///?odbc_connect={params}"

# Fetch OpenAI credentials
azure_endpoint = os.getenv('Azure_EndPoint')
api_key = os.getenv('API_Key')

if not azure_endpoint or not api_key:
    st.error("Missing Azure OpenAI credentials. Please check your `.env` file.")
    st.stop()

# Initialize Azure OpenAI model
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    model="gpt-4o",
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    api_version='2024-02-15-preview'
)

def create_agent(tools, llm, yaml_text):
    system_prompt = """You are an AI SQL Assistant with access to the following tools:
    {tools}

    Your task is to generate a **correct T-SQL query** based on the user input. 
    You have access to this Yaml Semantic Model for the database as well {yaml_text}
    Please analyze the yaml file based on the user input and give me correct T-SQL Query
    
    Steps: 
    - **Analyze all available tables** and their relationships.  
    - **Ensure the query is free of syntax errors.**  
    - **Use the appropriate T-SQL syntax** for the given database.  
    - **Return only the T-SQL query** without any explanation.  

     **Output format (always return JSON):**
    ```json
    {{
      "action": "Final Answer",
      "action_input": "GENERATED_TSQL_QUERY"
    }}
    ```"""

    human_prompt = """User Input:{input}

    Previous Query (if any): {agent_scratchpad}
    """

    memory = ConversationBufferMemory()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human_prompt),
        ]
    ).partial(
        tools=render_text_description_and_args(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            chat_history=lambda x: memory.chat_memory.messages,
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, memory=memory)


# Database connection setup with error handling
@st.cache_resource(ttl=7200)
def configure_db():
    try:
        engine = create_engine(db_uri)
        return SQLDatabase(engine)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

db = configure_db()

def execute_query(query):
    engine = create_engine(db_uri)
    with engine.connect() as connection:
        result = connection.execute(query)
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys())
    return df

# Initialize LangChain SQL Toolkit and Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

agent_executor = create_agent(tools, llm)

# Define Output Parser

# Initialize chat history
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with your SQL queries?"}]

# Get user input
user_query = st.chat_input(placeholder="Ask anything from the database...")

rephrase_prompt = f''' Your task to rephase the text given below and give different forms for rephrased version
text : {user_query}

output be like:
1. rephrased sentence 1
2. rephrased sentence 2
3. rephrased sentence 3
'''

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": rephrase_prompt},
]

response_rephrase = llm.invoke(messages)

extracted_response = response_rephrase.content

finalized_suggesstion = extracted_response[extracted_response.find('1.'):]

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    st.write(finalized_suggesstion)

    with st.chat_message("assistant"):
        try:
            # Execute the SQL query generated by the agent
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke({"input": user_query})

            # Parse the JSON response safely
            if isinstance(response, dict) and "output" in response:
                extracted_query = str(response["output"])
            else:
                st.error("Unexpected response format from agent.")
                extracted_query = None

            if extracted_query:
                with st.expander('SQL Query'):
                    st.write(f"```{extracted_query}```")

                # Execute SQL Query
                df = execute_query(text(extracted_query))

                # Display results in Streamlit
                if not df.empty:
                    if len(df.index) > 1:
                        data_tab, line_tab, bar_tab, area_tab = st.tabs(
                            ["Data", "Line Chart", "Bar Chart", "Area Chart"]
                        )
                        data_tab.dataframe(df)
                        if len(df.columns) > 1:
                            df = df.set_index(df.columns[0])
                        with line_tab:
                            st.line_chart(df)
                        with bar_tab:
                            st.bar_chart(df)
                        with area_tab:
                            st.area_chart(df)
                    else:
                        st.dataframe(df)
                else:
                    st.write("No results found.")

        except Exception as e:
            response = f"⚠️ Error processing query: {e}"
            st.error(response)

        # Store assistant response in chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("### How to Use")
st.sidebar.info("""
- Ask questions about your database in natural language.
- Review the generated SQL query.
- Your Results will be generated in Table format.
- Download results as CSV if needed.
- Access previous queries from the history.
""")