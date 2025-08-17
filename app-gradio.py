import gradio as gr
import pandas as pd
from argparse import ArgumentParser
from pydantic import BaseModel
import logging
import langchain
import httpx

langchain.debug = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AppSettings(BaseModel):
    IMPALA_HOST: str = 'coordinator-ares-impala-vw.apps.cdppvc.ares.olympus.cloudera.com'
    IMPALA_PORT: int = 443
    USERNAME: str = 'dennislee'
    PASSWORD: str = 'YEEI0oy8BFnSRpwm'
    HTTP_PATH: str = '/cliservice'
    DATABASE: str = 'dlee_telco'
    LLM_ENDPOINT_URL: str = 'https://fastapi-llm.ml-9df5bc51-1da.apps.cdppvc.ares.olympus.cloudera.com/v1'

settings = AppSettings()

def _get_args():
    parser = ArgumentParser(description="Conversational AI Database Analyst Demo")
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--inbrowser", action="store_true", default=False)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    return parser.parse_args()

def initialize_llm():
    """Initializes the LLM using LangChain's ChatOpenAI class."""
    print(f"Connecting to LLM at endpoint: {settings.LLM_ENDPOINT_URL}")
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
        print("LLM connected successfully via LangChain (SSL verification disabled).")
        return llm
    except Exception as e:
        logging.error("Failed to connect to LLM", exc_info=True)
        raise

def initialize_database():
    """Initializes the LangChain SQLDatabase connection."""
    print("Initializing database connection...")
    db_uri = (
        f"impala://{settings.USERNAME}:{settings.PASSWORD}@{settings.IMPALA_HOST}:{settings.IMPALA_PORT}/{settings.DATABASE}?"
        f"auth_mechanism=LDAP&use_ssl=true&use_http_transport=true&http_path={settings.HTTP_PATH}"
    )
    db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
    print("Database connection initialized.")
    print(f"Discovered tables: {db.get_usable_table_names()}")
    return db

def build_chatbot_ui(llm, db):
    with gr.Blocks(theme=gr.themes.Soft(), css=".control-height { height: 500px; overflow: auto; }") as demo:
        task_history = gr.State([])
        gr.Image(value="/home/cdsw/clouderalogo.png", height=40, show_label=False, show_download_button=False, interactive=False, container=False)
        gr.Markdown(
            f"""<div style="text-align: center;"><h1>Local LLM 💬 with Iceberg 💎 Database</h1><p>Ask any question about {settings.DATABASE} database</p></div>"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height", type="messages")
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="e.g., How many call centers are in the 'Mid-West' division?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    regenerate_btn = gr.Button("Regenerate")
                    clear_btn = gr.Button("Clear History")
        
        def predict(query, history):
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Thinking..."})
            yield history, history

            try:
                sql_query_chain = create_sql_query_chain(llm, db)
                raw_llm_output = sql_query_chain.invoke({"question": query})

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
                logging.error(f"An error occurred during SQL generation: {e}", exc_info=True)
                error_message = f"❌ **Error during SQL generation:**\n\nI encountered an issue. Please check the terminal logs."
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return
            
            history[-1] = {"role": "assistant", "content": f"🏃 Running query...\n```sql\n{generated_sql}\n```"}
            yield history, history

            try:
                logging.info(f"\n--- Executing SQL on Remote DB ---\n{generated_sql}\n--------------------")
                execute_query_tool = QuerySQLDataBaseTool(db=db)
                db_results = execute_query_tool.invoke(generated_sql)
                logging.info(f"\n--- Results from DB ---\n{db_results}\n--------------------")

                if not db_results or db_results.strip() == "[]":
                    logging.info("Database returned an empty result set. Halting before summarization.")
                    final_answer = "The query ran successfully but found no matching records in the database."
                    history[-1] = {"role": "assistant", "content": final_answer}
                    yield history, history
                    return

            except Exception as e:
                logging.error(f"An error occurred during database query: {e}", exc_info=True)
                error_message = f"❌ **Error during database query:**\n\nCheck terminal logs.\n\n**Details:**\n```\n{e}\n```"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return

            history[-1] = {"role": "assistant", "content": "✍️ Summarizing the results..."}
            yield history, history
            
            summarization_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Your task is to explain database query results to a user in a clear, conversational way.\n\n"
               "**IMPORTANT:** The data from the database is returned as a list of tuples, like `[(value,)]` or `[(3,)]`. "
               "You MUST extract the value from *inside* the tuple to form the answer. Do not count the number of tuples. For example, if the result is `[(3,)]`, the answer is 3."),
    ("user", "My original question was: \"{question}\"\n\nHere is the data from the database:\n```{result}```")
            ])
            
            summarize_chain = summarization_prompt | llm | StrOutputParser()
            
            try:
                final_summary = summarize_chain.invoke({"question": query, "result": db_results})
                
                if not final_summary:
                    final_summary = "The model returned an empty response. This might be due to content filtering or a prompt issue."

                history[-1] = {"role": "assistant", "content": final_summary}
                yield history, history

            except Exception as e:
                logging.error(f"Error during summarization with .invoke(): {e}", exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error during summarization. Check the terminal logs for the root cause."}
                yield history, history

        def regenerate(history):
            if len(history) >= 2:
                history.pop(); history.pop()
                last_query = history[-1]['content'] if history else ""
                if last_query:
                    yield from predict(last_query, history)
            yield history, history

        def clear_history(): return [], []
        def reset_user_input(): return gr.update(value="")

        submit_btn.click(predict, [query_box, chatbot], [chatbot, task_history]).then(reset_user_input, [], [query_box])
        regenerate_btn.click(regenerate, [task_history], [chatbot, task_history])
        clear_btn.click(clear_history, [], [chatbot, task_history])
        
    return demo

def main():
    """Main function to initialize components and launch the demo."""
    args = _get_args()
    llm = initialize_llm()
    db = initialize_database()
    demo = build_chatbot_ui(llm, db)
    demo.queue().launch(
        share=args.share, inbrowser=args.inbrowser, server_port=args.server_port, server_name=args.server_name
    )

if __name__ == "__main__":
    main()