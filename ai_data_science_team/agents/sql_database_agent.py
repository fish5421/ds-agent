

from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

import os
import pandas as pd
import sqlalchemy as sql

from IPython.display import Markdown

from ai_data_science_team.templates import(
    node_func_execute_agent_from_sql_connection,
    node_func_human_review,
    node_func_fix_agent_code, 
    node_func_explain_agent_code, 
    create_coding_agent_graph,
    BaseAgent,
)
from ai_data_science_team.tools.parsers import SQLOutputParser  
from ai_data_science_team.tools.regex import add_comments_to_top, format_agent_name, format_recommended_steps
from ai_data_science_team.tools.metadata import get_database_metadata
from ai_data_science_team.tools.logging import log_ai_function

# Setup
AGENT_NAME = "sql_database_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")

# Class

class SQLDatabaseAgent(BaseAgent):
    """
    Creates a SQL Database Agent that can recommend SQL steps and generate SQL code to query a database. 
    The agent can:
    - Propose recommended steps to answer a user's query or instructions.
    - Generate a SQL query based on the recommended steps and user instructions.
    - Execute that SQL query against the provided database connection.
    - Return the resulting data as a dictionary, suitable for conversion to a DataFrame or other structures.
    - Log generated code and errors if enabled.

    Parameters
    ----------
    model : ChatOpenAI or langchain.llms.base.LLM
        The language model used to generate the SQL code.
    connection : sqlalchemy.engine.base.Engine or sqlalchemy.engine.base.Connection
        The SQLAlchemy connection (or engine) to the database.
    n_samples : int, optional
        Number of sample rows (per column) to retrieve when summarizing database metadata. Defaults to 10.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "sql_database.py".
    function_name : str, optional
        Name of the Python function that executes the SQL query. Defaults to "sql_database_pipeline".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of the recommended steps before generating code. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the step that generates recommended SQL steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions: str, max_retries=3, retry_count=0)
        Asynchronously runs the agent to generate and execute a SQL query based on user instructions.
    invoke_agent(user_instructions: str, max_retries=3, retry_count=0)
        Synchronously runs the agent to generate and execute a SQL query based on user instructions.
    explain_sql_steps()
        Returns an explanation of the SQL steps performed by the agent.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_data_sql()
        Retrieves the resulting data from the SQL query as a dictionary. 
        (You can convert this to a DataFrame if desired.)
    get_sql_query_code()
        Retrieves the exact SQL query generated by the agent.
    get_sql_database_function()
        Retrieves the Python function that executes the SQL query.
    get_recommended_sql_steps()
        Retrieves the recommended steps for querying the SQL database.
    get_response()
        Returns the full response dictionary from the agent.
    show()
        Displays the agent's mermaid diagram for visual inspection of the compiled graph.

    Examples
    --------
    ```python
    import sqlalchemy as sql
    from langchain_openai import ChatOpenAI
    from ai_data_science_team.agents import SQLDatabaseAgent

    # Create the engine/connection
    sql_engine = sql.create_engine("sqlite:///data/my_database.db")
    conn = sql_engine.connect()

    llm = ChatOpenAI(model="gpt-4o-mini")

    sql_database_agent = SQLDatabaseAgent(
        model=llm,
        connection=conn,
        n_samples=10,
        log=True,
        log_path="logs",
        human_in_the_loop=True
    )

    # Example usage
    sql_database_agent.invoke_agent(
        user_instructions="List all the tables in the database.",
        max_retries=3,
        retry_count=0
    )

    data_result = sql_database_agent.get_data_sql()  # dictionary of rows returned
    sql_code = sql_database_agent.get_sql_query_code()
    response = sql_database_agent.get_response()
    ```
    
    Returns
    -------
    SQLDatabaseAgent : langchain.graphs.CompiledStateGraph
        A SQL database agent implemented as a compiled state graph.
    """

    def __init__(
        self,
        model,
        connection,
        n_samples=10,
        log=False,
        log_path=None,
        file_name="sql_database.py",
        function_name="sql_database_pipeline",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False
    ):
        self._params = {
            "model": model,
            "connection": connection,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """
        Create or rebuild the compiled graph for the SQL Database Agent.
        Running this method resets the response to None.
        """
        self.response = None
        return make_sql_database_agent(**self._params)

    def update_params(self, **kwargs):
        """
        Updates the agent's parameters (e.g. connection, n_samples, log, etc.) 
        and rebuilds the compiled graph.
        """
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def ainvoke_agent(self, user_instructions: str=None, max_retries=3, retry_count=0, **kwargs):
        """
        Asynchronously runs the SQL Database Agent based on user instructions.

        Parameters
        ----------
        user_instructions : str
            Instructions for the SQL query or metadata request.
        max_retries : int, optional
            Maximum retry attempts. Defaults to 3.
        retry_count : int, optional
            Current retry count. Defaults to 0.
        kwargs : dict
            Additional keyword arguments to pass to ainvoke().

        Returns
        -------
        None
        """
        response = self._compiled_graph.ainvoke({
            "user_instructions": user_instructions,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response

    def invoke_agent(self, user_instructions: str=None, max_retries=3, retry_count=0, **kwargs):
        """
        Synchronously runs the SQL Database Agent based on user instructions.

        Parameters
        ----------
        user_instructions : str
            Instructions for the SQL query or metadata request.
        max_retries : int, optional
            Maximum retry attempts. Defaults to 3.
        retry_count : int, optional
            Current retry count. Defaults to 0.
        kwargs : dict
            Additional keyword arguments to pass to invoke().

        Returns
        -------
        None
        """
        response = self._compiled_graph.invoke({
            "user_instructions": user_instructions,
            "max_retries": max_retries,
            "retry_count": retry_count
        }, **kwargs)
        self.response = response

    def explain_sql_steps(self):
        """
        Provides an explanation of the SQL steps performed by the agent 
        if the explain step is not bypassed.

        Returns
        -------
        str or list
            An explanation of the SQL steps.
        """
        if self.response:
            return self.response.get("messages", [])
        return []

    def get_log_summary(self, markdown=False):
        """
        Retrieves a summary of the logging details if logging is enabled.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the summary in Markdown format.

        Returns
        -------
        str or None
            Log details or None if logging is not used or data is unavailable.
        """
        if self.response and self.response.get("sql_database_function_path"):
            log_details = f"Log Path: {self.response['sql_database_function_path']}"
            if markdown:
                return Markdown(log_details)
            return log_details
        return None

    def get_data_sql(self):
        """
        Retrieves the SQL query result from the agent's response.

        Returns
        -------
        dict or None
            The returned data as a dictionary of column -> list_of_values, 
            or None if no data is found.
        """
        if self.response and "data_sql" in self.response:
            return pd.DataFrame(self.response["data_sql"])
        return None

    def get_sql_query_code(self, markdown=False):
        """
        Retrieves the raw SQL query code generated by the agent (if available).
        
        Parameters
        ----------
        markdown : bool, optional
            If True, returns the code in a Markdown code block.

        Returns
        -------
        str or None
            The SQL query as a string, or None if not available.
        """
        if self.response and "sql_query_code" in self.response:
            if markdown:
                return Markdown(f"```sql\n{self.response['sql_query_code']}\n```")
            return self.response["sql_query_code"]
        return None

    def get_sql_database_function(self, markdown=False):
        """
        Retrieves the Python function code used to execute the SQL query.

        Parameters
        ----------
        markdown : bool, optional
            If True, returns the code in a Markdown code block.

        Returns
        -------
        str or None
            The function code if available, otherwise None.
        """
        if self.response and "sql_database_function" in self.response:
            code = self.response["sql_database_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_recommended_sql_steps(self, markdown=False):
        """
        Retrieves the recommended SQL steps from the agent's response.
        
        Parameters
        ----------
        markdown : bool, optional
            If True, returns the steps in Markdown format.

        Returns
        -------
        str or None
            Recommended steps or None if not available.
        """
        if self.response and "recommended_steps" in self.response:
            if markdown:
                return Markdown(self.response["recommended_steps"])
            return self.response["recommended_steps"]
        return None



# Function

def make_sql_database_agent(
    model, 
    connection, 
    n_samples = 10, 
    log=False, 
    log_path=None, 
    file_name="sql_database.py",
    function_name="sql_database_pipeline",
    overwrite = True, 
    human_in_the_loop=False, bypass_recommended_steps=False, 
    bypass_explain_code=False
):
    """
    Creates a SQL Database Agent that can recommend SQL steps and generate SQL code to query a database. 
    
    Parameters
    ----------
    model : ChatOpenAI
        The language model to use for the agent.
    connection : sqlalchemy.engine.base.Engine
        The connection to the SQL database.
    n_samples : int, optional
        The number of samples to retrieve for each column, by default 10. 
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether to log the generated code, by default False
    log_path : str, optional
        The path to the log directory, by default None
    file_name : str, optional
        The name of the file to save the generated code, by default "sql_database.py"
    overwrite : bool, optional
        Whether to overwrite the existing log file, by default True
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the feature engineering instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    
    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data cleaning agent as a state graph.
        
    Examples
    --------
    ```python
    from ai_data_science_team.agents import make_sql_database_agent
    import sqlalchemy as sql
    from langchain_openai import ChatOpenAI

    sql_engine = sql.create_engine("sqlite:///data/leads_scored.db")

    conn = sql_engine.connect()

    llm = ChatOpenAI(model="gpt-4o-mini")
    
    sql_agent = make_sql_database_agent(
        model=llm, 
        connection=conn
    )

    sql_agent

    response = sql_agent.invoke({
        "user_instructions": "List the tables in the database",
        "max_retries":3, 
        "retry_count":0
    })
    ```
    
    """
    
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection
    
    llm = model
    
    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")
    
    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_sql: dict
        all_sql_database_summary: str
        sql_query_code: str
        sql_database_function: str
        sql_database_function_path: str
        sql_database_function_file_name: str
        sql_database_function_name: str
        sql_database_error: str
        max_retries: int
        retry_count: int
    
    def recommend_sql_steps(state: GraphState):
        
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND STEPS")
        
        
        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a SQL Database Instructions Expert. Given the following information about the SQL database, 
            recommend a series of numbered steps to take to collect the data and process it according to user instructions. 
            The steps should be tailored to the SQL database characteristics and should be helpful 
            for a sql database coding agent that will write the SQL code.
            
            IMPORTANT INSTRUCTIONS:
            - Take into account the user instructions and the previously recommended steps.
            - If no user instructions are provided, just return the steps needed to understand the database.
            - Take into account the database dialect and the tables and columns in the database.
            - Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            - IMPORTANT: Pay attention to the table names and column names in the database. Make sure to use the correct table and column names in the SQL code. If a space is present in the table name or column name, make sure to account for it.
            
            
            User instructions / Question:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return the steps as a numbered point list (no code, just the steps).
            
            Consider these:
            
            1. Consider the database dialect and the tables and columns in the database.
            
            
            Avoid these:
            1. Do not include steps to save files.
            2. Do not include steps to modify existing tables, create new tables or modify the database schema.
            3. Do not include steps that alter the existing data in the database.
            4. Make sure not to include unsafe code that could cause data loss or corruption or SQL injections.
            5. Make sure to not include irrelevant steps that do not help in the SQL agent's data collection and processing. Examples include steps to create new tables, modify the schema, save files, create charts, etc.
  
            
            """,
            input_variables=["user_instructions", "recommended_steps", "all_sql_database_summary"]
        )
        
        # Create a connection if needed
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        # Get the database metadata
        all_sql_database_summary = get_database_metadata(conn, n_samples=n_samples)
        
        steps_agent = recommend_steps_prompt | llm
        
        recommended_steps = steps_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_sql_database_summary": all_sql_database_summary
        })
        
        return {
            "recommended_steps": format_recommended_steps(recommended_steps.content.strip(), heading="# Recommended SQL Database Steps:"),
            "all_sql_database_summary": all_sql_database_summary
        }
        
    def create_sql_query_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))
        print("    * CREATE SQL QUERY CODE")
        
        # Prompt to get the SQL code from the LLM
        sql_query_code_prompt = PromptTemplate(
            template="""
            You are a SQL Database Coding Expert. Given the following information about the SQL database, 
            write the SQL code to collect the data and process it according to user instructions. 
            The code should be tailored to the SQL database characteristics and should take into account user instructions, recommended steps, database and table characteristics.
            
            IMPORTANT INSTRUCTIONS:
            - Do not use a LIMIT clause unless a user specifies a limit to be returned.
            - Return SQL in ```sql ``` format.
            - Only return a single query if possible.
            - Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
            - Pay attention to the SQL dialect from the database summary metadata. Write the SQL code according to the dialect specified.
            - IMPORTANT: Pay attention to the table names and column names in the database. Make sure to use the correct table and column names in the SQL code. If a space is present in the table name or column name, make sure to account for it.
            
            
            User instructions / Question:
            {user_instructions}

            Recommended Steps:
            {recommended_steps}

            Below are summaries of the database metadata and the SQL tables:
            {all_sql_database_summary}

            Return:
            - The SQL code in ```sql ``` format to collect the data and process it according to the user instructions.
            
            Avoid these:
            - Do not include steps to save files.
            - Do not include steps to modify existing tables, create new tables or modify the database schema.
            - Make sure not to alter the existing data in the database.
            - Make sure not to include unsafe code that could cause data loss or corruption.
            
            """,
            input_variables=["user_instructions", "recommended_steps", "all_sql_database_summary"]
        )
        
        # Create a connection if needed
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        # Get the database metadata
        all_sql_database_summary = get_database_metadata(conn, n_samples=n_samples)
        
        sql_query_code_agent = sql_query_code_prompt | llm | SQLOutputParser()
        
        sql_query_code = sql_query_code_agent.invoke({
            "user_instructions": state.get("user_instructions"),
            "recommended_steps": state.get("recommended_steps"),
            "all_sql_database_summary": all_sql_database_summary
        })
        
        print("    * CREATE PYTHON FUNCTION TO RUN SQL CODE")
        
        response = f"""
def {function_name}(connection):
    import pandas as pd
    import sqlalchemy as sql
    
    # Create a connection if needed
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    sql_query = '''
    {sql_query_code}
    '''
    
    return pd.read_sql(sql_query, connection)
        """
        
        response = add_comments_to_top(response, AGENT_NAME)
            
        # For logging: store the code generated
        file_path, file_name_2 = log_ai_function(
            response=response,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite
        )
        
        return {
            "sql_query_code": sql_query_code,
            "sql_database_function": response,
            "sql_database_function_path": file_path,
            "sql_database_function_file_name": file_name_2,
            "sql_database_function_name": function_name,
            "all_sql_database_summary": all_sql_database_summary
        }
        
    # Human Review   
    
    prompt_text_human_review = "Are the following SQL agent instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    
    if not bypass_explain_code:
        def human_review(state: GraphState) -> Command[Literal["recommend_sql_steps", "explain_sql_database_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= 'explain_sql_database_code',
                no_goto="recommend_sql_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="sql_database_function",
            )
    else:
        def human_review(state: GraphState) -> Command[Literal["recommend_sql_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto= '__end__',
                no_goto="recommend_sql_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="sql_database_function", 
            )
    
    def execute_sql_database_code(state: GraphState):
        
        is_engine = isinstance(connection, sql.engine.base.Engine)
        conn = connection.connect() if is_engine else connection
        
        return node_func_execute_agent_from_sql_connection(
            state=state,
            connection=conn,
            result_key="data_sql",
            error_key="sql_database_error",
            code_snippet_key="sql_database_function",
            agent_function_name=state.get("sql_database_function_name"),
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during executing the sql database pipeline: "
        )
    
    def fix_sql_database_code(state: GraphState):
        prompt = """
        You are a SQL Database Agent code fixer. Your job is to create a {function_name}(connection) function that can be run on a sql connection. The function is currently broken and needs to be fixed.
        
        Make sure to only return the function definition for {function_name}().
        
        Return Python code in ```python``` format with a single function definition, {function_name}(connection), that includes all imports inside the function. The connection object is a SQLAlchemy connection object. Don't specify the class of the connection object, just use it as an argument to the function.
        
        This is the broken code (please fix): 
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="sql_database_function",
            error_key="sql_database_error",
            llm=llm,  
            prompt_template=prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("sql_database_function_path", None),
            function_name=state.get("sql_database_function_name"),
        )
        
    def explain_sql_database_code(state: GraphState):
        return node_func_explain_agent_code(
            state=state,
            code_snippet_key="sql_database_function",
            result_key="messages",
            error_key="sql_database_error",
            llm=llm,
            role=AGENT_NAME,
            explanation_prompt_template="""
            Explain the SQL steps that the SQL Database agent performed in this function. 
            Keep the summary succinct and to the point.\n\n# SQL Database Agent:\n\n{code}
            """,
            success_prefix="# SQL Database Agent:\n\n",  
            error_message="The SQL Database Agent encountered an error during SQL Query Analysis. No SQL function explanation is returned."
        )
        
    # Create the graph
    node_functions = {
        "recommend_sql_steps": recommend_sql_steps,
        "human_review": human_review,
        "create_sql_query_code": create_sql_query_code,
        "execute_sql_database_code": execute_sql_database_code,
        "fix_sql_database_code": fix_sql_database_code,
        "explain_sql_database_code": explain_sql_database_code
    }
    
    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_sql_steps",
        create_code_node_name="create_sql_query_code",
        execute_code_node_name="execute_sql_database_code",
        fix_code_node_name="fix_sql_database_code",
        explain_code_node_name="explain_sql_database_code",
        error_key="sql_database_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=MemorySaver() if human_in_the_loop else None,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
    )
        
    return app
        
             
            
        
        
        
        
    