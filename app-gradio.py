import gradio as gr
import pandas as pd
import logging
import httpx
import langchain

from argparse import ArgumentParser
from pydantic import BaseModel
from typing import Generator

# Enable LangChain debug mode for detailed logs
langchain.debug = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION ---

class AppSettings(BaseModel):
    """Manages application settings using Pydantic for validation."""
    IMPALA_HOST: str = 'coordinator-ares-impala-vw.apps.cdppvc.ares.olympus.cloudera.com'
    IMPALA_PORT: int = 443
    USERNAME: str = 'dennislee'
    # WARNING: Hardcoding passwords is not recommended in production.
    # Use environment variables or a secrets management tool instead.
    PASSWORD: str = 'YEEI0oy8BFnSRpwm'
    HTTP_PATH: str = '/cliservice'
    DATABASE: str = 'dlee_telco'
    LLM_ENDPOINT_URL: str = 'https://fastapi-llm.ml-9df5bc51-1da.apps.cdppvc.ares.olympus.cloudera.com/v1'

settings = AppSettings()

# --- 2. INITIALIZATION ---

def _get_args():
    """Parses command-line arguments for the application."""
    parser = ArgumentParser(description="Conversational AI Database Analyst Demo")
    parser.add_argument("--share", action="store_true", default=False, help="Enable Gradio sharing.")
    parser.add_argument("--inbrowser", action="store_true", default=False, help="Launch Gradio in a new browser tab.")
    parser.add_argument("--server-port", type=int, default=8000, help="Port to run the Gradio server on.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Hostname to run the Gradio server on.")
    return parser.parse_args()

def initialize_llm() -> ChatOpenAI:
    """
    Initializes the connection to the Large Language Model via a custom endpoint.
    
    Returns:
        An instance of ChatOpenAI configured for the local LLM.
    """
    logging.info(f"Connecting to LLM at endpoint: {settings.LLM_ENDPOINT_URL}")
    try:
        llm = ChatOpenAI(
            model='huggingface/dummy',
            temperature=0.7,
            base_url=settings.LLM_ENDPOINT_URL,
            api_key="dummy",
            max_tokens=4096,
            request_timeout=120,
            http_client=httpx.Client(verify=False), # Disable SSL verification for custom certs
            model_kwargs={
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "seed": 42,
            }
        )
        logging.info("LLM connected successfully (SSL verification disabled).")
        return llm
    except Exception as e:
        logging.error("Failed to connect to LLM", exc_info=True)
        raise

def initialize_database() -> SQLDatabase:
    """
    Initializes the LangChain SQLDatabase connection to Apache Impala.

    Returns:
        An instance of SQLDatabase connected to the specified database.
    """
    logging.info("Initializing database connection...")
    db_uri = (
        f"impala://{settings.USERNAME}:{settings.PASSWORD}@{settings.IMPALA_HOST}:{settings.IMPALA_PORT}/{settings.DATABASE}?"
        f"auth_mechanism=LDAP&use_ssl=true&use_http_transport=true&http_path={settings.HTTP_PATH}"
    )
    db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
    logging.info(f"Database connection initialized. Discovered tables: {db.get_usable_table_names()}")
    return db

# --- 3. GRADIO UI AND CORE LOGIC ---

def build_chatbot_ui(llm: ChatOpenAI, db: SQLDatabase) -> gr.Blocks:
    """
    Builds the Gradio user interface for the chatbot.

    Args:
        llm: The initialized ChatOpenAI language model instance.
        db: The initialized SQLDatabase instance.

    Returns:
        A Gradio Blocks object representing the UI.
    """
    with gr.Blocks(theme=gr.themes.Soft(), css=".control-height { height: 500px; overflow: auto; }") as demo:
        task_history = gr.State([])
        gr.Image(value="/home/cdsw/clouderalogo.png", height=40, show_label=False, show_download_button=False, interactive=False, container=False)
        gr.Markdown(
            f"""<div style="text-align: center;"><h1>Local LLM 💬 with Iceberg 💎 Database</h1><p>Ask any question about the <strong>{settings.DATABASE}</strong> database.</p><p>This chatbot supports Iceberg Time Travel by using <strong>system time</strong> in your question.</p></div>"""
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height", type="messages")
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="e.g., List prepaid plan that customers can buy or How many customers existed as of system time 2025-08-10?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    regenerate_btn = gr.Button("Regenerate")
                    clear_btn = gr.Button("Clear History")
        
        def predict(query: str, history: list) -> Generator[tuple[list, list], None, None]:
            """
            The core logic function that processes a user query and yields updates to the UI.
            This is a generator function to stream responses.
            """
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Thinking..."})
            yield history, history

            # --- Step 1: Generate SQL Query with Iceberg Time Travel Context ---
            try:
                # CORRECTED PROMPT: Uses {input} and {top_k} as required by the chain.
                sql_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an expert SQL assistant. Given a user question, create a syntactically correct SQL query to run against an Impala database.\n"
                     "**IMPORTANT**: This database supports Apache Iceberg time travel. If the user's question includes the keywords 'system time' or asks for data 'as of' a specific past date, you MUST use the `FOR SYSTEM_TIME AS OF 'YYYY-MM-DD HH:MI:SS'` syntax in your query. "
                     "You must never use data that does not exist yet. The current system time is 2025-08-17 18:08:05.\n"
                     "Here is the relevant table schema: {table_info}\n"
                     "Limit the number of results to {top_k}."),
                    ("user", "{input}") 
                ])

                sql_query_chain = create_sql_query_chain(llm, db, prompt=sql_generation_prompt)
                
                # The key remains "question" here, as the chain internally maps it to the "input" variable.
                raw_llm_output = sql_query_chain.invoke({"question": query})

                # Clean up LLM output to isolate the SQL query
                generated_sql = raw_llm_output
                if "```sql" in generated_sql:
                    generated_sql = generated_sql.split("```sql")[1].split("```")[0]
                if "SQLQuery:" in generated_sql:
                    generated_sql = generated_sql.split("SQLQuery:")[-1]
                if "SQLResult:" in generated_sql:
                    generated_sql = generated_sql.split("SQLResult:")[0]
                generated_sql = generated_sql.strip()
                
                logging.info(f"\n--- Generated SQL ---\n{generated_sql}\n--------------------")

            except Exception as e:
                logging.error(f"Error during SQL generation: {e}", exc_info=True)
                error_message = f"❌ **Error during SQL generation:**\n\nI encountered an issue creating the SQL query. Please check the terminal logs for details."
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return
            
            history[-1] = {"role": "assistant", "content": f"🏃 Running query...\n```sql\n{generated_sql}\n```"}
            yield history, history

            # --- Step 2: Execute the SQL Query ---
            try:
                execute_query_tool = QuerySQLDataBaseTool(db=db)
                db_results = execute_query_tool.invoke(generated_sql)
                logging.info(f"\n--- Results from DB ---\n{db_results}\n--------------------")

                if not db_results or db_results.strip() == "[]":
                    logging.info("Database returned an empty result set.")
                    final_answer = "The query ran successfully but found no matching records in the database."
                    history[-1] = {"role": "assistant", "content": final_answer}
                    yield history, history
                    return

            except Exception as e:
                logging.error(f"Error during database query: {e}", exc_info=True)
                error_message = f"❌ **Error during database query:**\n\nCheck terminal logs.\n\n**Details:**\n```\n{e}\n```"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return

            history[-1] = {"role": "assistant", "content": "✍️ Summarizing the results..."}
            yield history, history
            
            # --- Step 3: Summarize Results for the User ---
            try:
                summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Your task is to explain database query results to a user in a clear, conversational way.\n\n"
               "**IMPORTANT:** The data from the database is often a list of tuples, like `[(value,)]` or `[(3,)]`. "
               "You MUST extract the value from *inside* the tuple to form the answer. For example, if the result is `[(3,)]`, the answer is 3.\n\n"
               "**Provide a direct, factual answer and do not ask any follow-up questions.**"), # <-- ADD THIS INSTRUCTION
    ("user", "My original question was: \"{question}\"\n\nHere is the data from the database:\n```{result}```")
                ])
                
                summarize_chain = summarization_prompt | llm | StrOutputParser()
                final_summary = summarize_chain.invoke({"question": query, "result": db_results})
                
                if not final_summary:
                    final_summary = "The model returned an empty response. This might be due to content filtering or a prompt issue."

                history[-1] = {"role": "assistant", "content": final_summary}
                yield history, history

            except Exception as e:
                logging.error(f"Error during summarization: {e}", exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error during summarization. Please check the terminal logs for the root cause."}
                yield history, history

        def regenerate(history: list) -> Generator[tuple[list, list], None, None]:
            """Regenerates the last response."""
            if len(history) >= 2:
                history.pop(); history.pop()
                if history:
                    last_query = history[-1]['content']
                    yield from predict(last_query, history)
            yield history, history

        def clear_history() -> tuple[list, list]:
            """Clears the chat history."""
            return [], []

        def reset_user_input():
            """Resets the user input box."""
            return gr.update(value="")

        # --- Event Listeners ---
        submit_btn.click(predict, [query_box, chatbot], [chatbot, task_history]).then(reset_user_input, [], [query_box])
        regenerate_btn.click(regenerate, [task_history], [chatbot, task_history])
        clear_btn.click(clear_history, [], [chatbot, task_history])
        
        return demo

# --- 4. MAIN EXECUTION BLOCK ---

def main():
    """Main function to initialize components and launch the demo."""
    args = _get_args()
    llm = initialize_llm()
    db = initialize_database()
    demo = build_chatbot_ui(llm, db)
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )

if __name__ == "__main__":
    main()