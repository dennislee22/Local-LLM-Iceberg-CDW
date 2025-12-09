import gradio as gr
import pandas as pd
import logging
import httpx
import langchain
import ast
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from argparse import ArgumentParser
from pydantic import BaseModel
from typing import Generator
import impala.dbapi

langchain.debug = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AppSettings(BaseModel):
    """Manages application settings using Pydantic for validation."""
    IMPALA_HOST: str = 'coordinator-blah-impala.site'
    IMPALA_PORT: int = 443
    USERNAME: str = 'blah'
    PASSWORD: str = 'blah'
    HTTP_PATH: str = 'cliservice'
    DATABASE: str = 'mbb_product_catalogue'
    LLM_ENDPOINT_URL: str = 'https://fastapi-llm.blah.com/v1'

settings = AppSettings()

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
            http_client=httpx.Client(verify=False),
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
    logging.info("Connecting to Impala...")

    connect_args = {
        "user": settings.USERNAME,
        "password": settings.PASSWORD,
        "auth_mechanism": 'LDAP',
        "use_ssl": True,
        "use_http_transport": True,
        "http_path": settings.HTTP_PATH,
    }

    engine = create_engine(
        f"impala://{settings.IMPALA_HOST}:{settings.IMPALA_PORT}/{settings.DATABASE}",
        creator=lambda: impala.dbapi.connect(
            host=settings.IMPALA_HOST,
            port=settings.IMPALA_PORT,
            database=settings.DATABASE,
            **connect_args
        )
    )

    tables = ["dim_product_catalogue", "dim_product_category", "dim_product_price"]

    db = SQLDatabase(engine=engine, include_tables=tables, sample_rows_in_table_info=10)
    logging.info(f"Tables: {db.get_usable_table_names()}")
    return db

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
        gr.Image(value="/home/cdsw/mbb-cldr.png", height=60, show_label=False, show_download_button=False, interactive=False, container=False)
        gr.Markdown(
            f"""<div style="text-align: center;"><h1>Local LLM 💬 with Iceberg 💎 Database</h1><p>This chatbot supports Iceberg 🕒 Travel by using <strong>system time</strong> keyword in your question.</p></div>"""
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height", type="messages")
                # NEW: Add the Plot component, which will be hidden by default
                output_plot = gr.Plot(label="Chart", visible=False)
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="e.g., Compare total revenue between prepaid and postpaid plans. How many customers exist in the table based on system time 2025-08-18 03:34:59?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear History")
        
        def predict(query: str, history: list) -> Generator[tuple[list, list, gr.Plot], None, None]:
            """
            The core logic function that processes a user query and yields updates to the UI.
            This is a generator function to stream responses and charts.
            """
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Thinking..."})
            # Initially, ensure the plot is hidden
            yield history, history, gr.update(visible=False)

            # --- Step 1: Generate SQL Query with Iceberg Time Travel Context ---
            generated_sql = ""
            try:
                sql_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an expert SQL assistant. Your sole purpose is to generate a single, syntactically correct SQL query for an Impala database based on a user's question.\n\n"
                     "**CRITICAL RULES:**\n"
                     "1. **STUDY SCHEMA:** You MUST study the schema of each table, join the table if necessary to produce the correct SQL query.\n"
                     "2. **SINGLE QUERY ONLY:** You MUST generate only one single SQL query. Do not output multiple queries, comments, or any explanatory text. The entire output must be only the SQL statement.\n"
                     "3. **ICEBERG TIME TRAVEL:** If the user's question includes 'system time', you MUST use the `FOR SYSTEM_TIME AS OF 'YYYY-MM-DD HH:MI:SS'` syntax on every table in the query. Otherwise, do not use this syntax.\n\n"
                     "Here is the relevant table schema: {table_info}\n"
                     "Limit the number of results to {top_k}."
                    ),
                    ("user", "{input}") 
                ])

                sql_query_chain = create_sql_query_chain(llm, db, prompt=sql_generation_prompt)
                raw_llm_output = sql_query_chain.invoke({"question": query})
                
                # Clean up the generated SQL
                generated_sql = raw_llm_output.strip()
                if "```sql" in generated_sql:
                    generated_sql = generated_sql.split("```sql")[1].split("```")[0]
                if "SQLQuery:" in generated_sql:
                    generated_sql = generated_sql.split("SQLQuery:")[-1]
                if ";" in generated_sql:
                    generated_sql = generated_sql.split(";")[0].strip() + ";"
                
                logging.info(f"\n--- single SQL ---\n{generated_sql}\n--------------------")

            except Exception as e:
                logging.error(f"Error during SQL generation: {e}", exc_info=True)
                error_message = f"❌ **Error during SQL generation:**\n\nI encountered an issue creating the SQL query. Please check the terminal logs for details."
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history, gr.update(visible=False)
                return
            
            history[-1] = {"role": "assistant", "content": f"🏃 Running query...\n```sql\n{generated_sql}\n```"}
            yield history, history, gr.update(visible=False)

            # --- Step 2: Execute the SQL Query ---
            db_results = ""
            try:
                execute_query_tool = QuerySQLDataBaseTool(db=db)
                db_results = execute_query_tool.invoke(generated_sql)
                logging.info(f"\n--- Results from DB ---\n{db_results}\n--------------------")

                if not db_results or db_results.strip() == "[]":
                    logging.info("Database returned an empty result set.")
                    final_answer = "The query ran successfully but found no matching records in the database."
                    history[-1] = {"role": "assistant", "content": final_answer}
                    yield history, history, gr.update(visible=False)
                    return

            except Exception as e:
                logging.error(f"Error during database query: {e}", exc_info=True)
                error_message = f"❌ **Error during database query:**\n\nThe database rejected the following query:\n```sql\n{generated_sql}\n```\n**Error Details:**\n`{e}`"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history, gr.update(visible=False)
                return

            # --- Step 2.5: Decide if data is plottable and generate a chart ---
            plot_figure = None
            try:
                # This prompt asks the LLM to act as a data analyst and decide if a chart is suitable.
                # It must respond in a structured JSON format.
                plotting_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an analytical assistant. Your task is to determine if a database result can be visualized as a bar or pie chart. "
                     "The data will typically compare categories (e.g., 'prepaid' vs 'postpaid' counts). "
                     "Respond with ONLY a JSON object with the keys: 'plottable' (boolean), 'chart_type' ('bar' or 'pie' or 'none'), 'title' (string), 'x_label' (string, for the category), and 'y_label' (string, for the value). "
                     "Example for a comparison: {{\"plottable\": true, \"chart_type\": \"bar\", \"title\": \"Comparison of Plan Types\", \"x_label\": \"Plan Type\", \"y_label\": \"Count\"}}. "
                     "If the data is just a single value (e.g., '[(3,)]') or a list of names, it is not plottable."
                    ),
                    ("user", 
                     "Based on my original question and the data below, can this be plotted?\n"
                     "Original question: \"{question}\"\n"
                     "Database result: ```{result}```"
                    )
                ])
                
                plotting_chain = plotting_prompt | llm | StrOutputParser()
                decision_str = plotting_chain.invoke({"question": query, "result": db_results})
                logging.info(f"\n--- Plotting Decision ---\n{decision_str}\n--------------------")
                
                # Sanitize the LLM output string to be Python-compatible
                safe_decision_str = decision_str.replace('true', 'True').replace('false', 'False')
                
                decision = ast.literal_eval(safe_decision_str)
                if decision.get("plottable"):
                    # If the LLM says we can plot, proceed to generate the chart.
                    data = ast.literal_eval(db_results)
                    
                    # --- FIX: Handle cases where the SQL query returns more than two columns ---
                    # We will assume the first column is the category and the second is the value for plotting.
                    if data and isinstance(data[0], tuple) and len(data[0]) > 2:
                        logging.warning(f"Query returned {len(data[0])} columns; using the first two for the chart.")
                        plot_data = [(row[0], row[1]) for row in data]
                    else:
                        plot_data = data
                    
                    df = pd.DataFrame(plot_data, columns=[decision.get("x_label", "Category"), decision.get("y_label", "Value")])
                    
                    fig, ax = plt.subplots()
                    chart_type = decision.get("chart_type")
                    title = decision.get("title")
                    x_label = decision.get("x_label")
                    y_label = decision.get("y_label")

                    if chart_type == 'bar':
                        # Use Matplotlib's bar function for explicit control
                        ax.bar(df[x_label], df[y_label])
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(title)
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    elif chart_type == 'pie':
                        df[y_label] = pd.to_numeric(df[y_label])
                        ax.pie(df[y_label], labels=df[x_label], autopct='%1.1f%%')
                        ax.set_title(title)
                        
                    plt.tight_layout()
                    plot_figure = fig
                    plt.close(fig) # Close the plot to prevent it from displaying in the console

            except Exception as plot_e:
                logging.error(f"Could not generate plot. Reason: {plot_e}", exc_info=True)
                plot_figure = None

            history[-1] = {"role": "assistant", "content": "✍️ Summarizing the results..."}
            yield history, history, gr.update(value=plot_figure, visible=plot_figure is not None)
            
            # --- Step 3: Summarize Results for the User ---
            try:
                summarization_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Your task is to explain database query results to a user in a clear, conversational way.\n\n"
                     "**IMPORTANT:** The data from the database is often a list of tuples, like `[(value,)]` or `[(3,)]`. "
                     "You MUST extract the value from *inside* the tuple to form the answer. For example, if the result is `[(3,)]`, the answer is 3.\n\n"
                     "**Provide a direct, factual answer and do not ask any follow-up questions. Do not give false information especially when SQL queries did not work due to errors. Must answer in English.**"),
                    ("user", "My original question was: \"{question}\"\n\nHere is the data from the database:\n```{result}```")
                ])
                
                summarize_chain = summarization_prompt | llm | StrOutputParser()
                final_summary = summarize_chain.invoke({"question": query, "result": db_results})
                
                if not final_summary:
                    final_summary = "The model returned an empty response. This might be due to content filtering or a prompt issue."

                history[-1] = {"role": "assistant", "content": final_summary}
                # Final yield with the text summary and the generated plot
                yield history, history, gr.update(value=plot_figure, visible=plot_figure is not None)

            except Exception as e:
                logging.error(f"Error during summarization: {e}", exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error during summarization. Please check the terminal logs for the root cause."}
                yield history, history, gr.update(value=plot_figure, visible=plot_figure is not None)

        def clear_history() -> tuple[list, list, gr.Plot]:
            """Clears the chat history and the plot."""
            return [], [], gr.update(value=None, visible=False)

        def reset_user_input():
            """Resets the user input box."""
            return gr.update(value="")

        submit_btn.click(
            predict, 
            [query_box, chatbot], 
            [chatbot, task_history, output_plot]
        ).then(reset_user_input, [], [query_box])
        
        clear_btn.click(
            clear_history, 
            [], 
            [chatbot, task_history, output_plot]
        )
        
        return demo

def main():
    """Main function to initialize and launch the application."""
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
