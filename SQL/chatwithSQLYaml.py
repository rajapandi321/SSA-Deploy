import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine, text
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
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

# Streamlit UI Configuration
st.set_page_config(page_title="SQL Explorer", layout="wide")

def create_connection_string(user, password, server_name, database):
    """
    Create a properly encoded ODBC connection string
    """
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server_name};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30"
    )
    return f"mssql+pyodbc:///?odbc_connect={params}"

def create_agent(llm, yaml_output):
    """
    Create an AI agent for generating SQL queries without SQLDatabaseToolkit
    """
    system_prompt = """
    Your task is to generate a **correct T-SQL query** based on the user input. 
    You have access to this YAML database schema:
    ```yaml
    {yaml_output}
    ```
    
    Steps: 
    - **Analyze all available tables and YAML database schema** and their relationships.  
    - **Ensure the query is free of syntax errors.**  
    - **Use the appropriate T-SQL syntax based on the YAML input** for the given database.  
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
        yaml_output=yaml_output,
    )
    
    tools =[]

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

def execute_query(query, db_uri):
    """
    Execute a SQL query and return results as a DataFrame
    """
    engine = create_engine(db_uri)
    with engine.connect() as connection:
        result = connection.execute(text(query))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=result.keys())
    return df

@st.cache_resource(ttl=7200)
def configure_db(db_uri):
    try:
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "login"

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

    # Login Page
    if st.session_state.page == "login":
        st.title("SQL Login")
        
        # User Input
        user = st.text_input("User Name", value="", placeholder="Enter SQL Server User")
        password = st.text_input("Password", value="", placeholder="Enter Password", type="password")
        server_name = st.text_input("Server or Host", value="", placeholder="Enter Server Name")
        database = st.text_input("Database", value="", placeholder="Enter Default Database")
        
        if st.button("Connect"):
            try:
                # Create connection string
                db_uri = create_connection_string(user, password, server_name, database)
                
                # Create database engine
                engine = create_engine(db_uri)
                
                # Create YAML File
                yaml_output = generate_yaml(engine)
                
                st.toast('Your YAML file is getting generated...Please wait')
                
                # Store connection details in session state
                st.session_state.db_uri = db_uri
                st.session_state.yaml_output = yaml_output
                st.session_state.page = "dashboard"
                st.rerun()
                
            except Exception as e:
                st.error(f"Connection failed: {e}")

    # Dashboard Page
    elif st.session_state.page == "dashboard":
        st.title("Chat with your SQL Database")
        st.subheader('Connected to the YAML Semantic Model File')
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.page = "login"
            st.rerun()

        # Retrieve stored connection details
        db_uri = st.session_state.get('db_uri')
        yaml_output = st.session_state.get('yaml_output')

        if not db_uri:
            st.error("No active database connection. Please log in.")
            st.stop()

        # Configure database
        db_engine = configure_db(db_uri)

        # Initialize agent without SQLDatabaseToolkit
        agent_executor = create_agent(llm, yaml_output)

        # Initialize chat history
        if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with your SQL queries?"}]

        # Get user input
        user_query = st.chat_input(placeholder="Ask anything from the database...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            # Rephrase query (optional)
            try:
                rephrase_prompt = f'''Your task is to rephrase the text given below and provide different forms
                text: {user_query}

                output format:
                1. rephrased sentence 1
                2. rephrased sentence 2
                3. rephrased sentence 3
                '''

                response_rephrase = llm.invoke([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": rephrase_prompt}
                ])

                finalized_suggestion = response_rephrase.content[response_rephrase.content.find('1.'):]
                st.write("Query Suggestions:", finalized_suggestion)
            except Exception as e:
                st.warning("Could not generate query suggestions.")

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
                        with st.expander('Generated SQL Query'):
                            st.code(extracted_query, language='sql')

                        # Execute SQL Query
                        try:
                            df = execute_query(extracted_query, db_uri)
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
                            st.info('Yaml File doesnt have the relative information based on your user query')

                except Exception as e:
                    st.error(f"Error processing query: {e}")

        # Sidebar usage instructions
        st.sidebar.markdown("### How to Use")
        st.sidebar.info("""
        - Ask questions about your database in natural language.
        - Review the generated SQL query.
        - Your Results will be generated in Table format.
        - Download results as CSV if needed.
        - Access previous queries from the history.
        """)

if __name__ == "__main__":
    main()