# Self-hosted LLM and Gradio runs inside same pod on CAI Public Cloud
import torch
import os
import argparse
import time
import uvicorn
import gradio as gr
import logging
import httpx
from sqlalchemy import create_engine, text 
import impala.dbapi
from pydantic import BaseModel, Field
from typing import List, Optional, Generator, Tuple
from contextlib import asynccontextmanager

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

model = None
tokenizer = None
CHECKPOINT_PATH = ""
os.environ["TRUST_REMOTE_CODE"] = "true"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppSettings(BaseModel):
    IMPALA_HOST: str = 'coordinator-impala.site'
    IMPALA_PORT: int = 443
    USERNAME: str = 'dennislee'
    PASSWORD: str = 'blah'
    HTTP_PATH: str = 'cliservice'
    DATABASE: str = 'mbb_product_catalogue'
    LLM_ENDPOINT_URL: str = 'local_llm_inference'

settings = AppSettings()

def get_args():
    parser = argparse.ArgumentParser(description="Run a Gradio Chatbot with a local LLM and SQL capabilities.")
    parser.add_argument(
        "-c", "--checkpoint-path",
        type=str,
        default="",
        required=True,
        help="Hugging Face model checkpoint path (local or on Hub)"
    )
    parser.add_argument("--share", action="store_true", default=False, help="Enable Gradio sharing.")
    parser.add_argument("--inbrowser", action="store_true", default=False, help="Launch Gradio in browser.")
    parser.add_argument("--server-port", type=int, default=8090, help="Port to run the Gradio server on.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Hostname for Gradio server.")
    return parser.parse_args()

def load_model_and_tokenizer(checkpoint_path: str):
    global model, tokenizer
    if not os.path.isdir(checkpoint_path):
        print(f"📥 Model not found locally at '{checkpoint_path}', attempting to download from Hub...")

    print(f"🚀 Loading model and tokenizer from {checkpoint_path}...")

    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device_map}")

    try:
        local_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

        local_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        ).eval()

        print("✅ Model and tokenizer loaded successfully.")

        model = local_model
        tokenizer = local_tokenizer
    except Exception as e:
        logging.critical(f"Failed to load model/tokenizer: {e}")
        raise e

def local_llm_inference(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    global model, tokenizer

    if model is None or tokenizer is None:
        raise RuntimeError("LLM model and tokenizer are not loaded.")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    outputs = pipe(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature if temperature > 0 else 0.01,
        top_p=0.95,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    generated_text = outputs[0]['generated_text']

    # CRITICAL: Strip the prompt from the generated text
    response_text = generated_text.split(prompt)[-1].strip()

    return response_text

def local_llm_wrapper(input_data) -> str:
    """
    Wrapper to handle both dictionary inputs (from RunnablePassthrough) 
    and direct ChatPromptValue inputs (from simplified chains).
    """
    # Check if the input is a dictionary (from Step 1 chain)
    if isinstance(input_data, dict):
        prompt_value = input_data["prompt"]
    # Otherwise, assume it is the ChatPromptValue object itself (from Steps 2.5/3 chains)
    else:
        prompt_value = input_data

    # Convert the ChatPromptValue object to a string before passing to the pipeline
    prompt = prompt_value.to_string()
    return local_llm_inference(prompt)

def initialize_llm():
    global CHECKPOINT_PATH

    logging.info(f"Loading local LLM from checkpoint: {CHECKPOINT_PATH}")
    try:
        load_model_and_tokenizer(CHECKPOINT_PATH)
        logging.info("Local LLM model loaded successfully.")

        return True

    except Exception as e:
        logging.error("Failed to load local LLM", exc_info=True)
        raise e

def test_llm_response():
    """Tests the LLM's ability to respond to a simple language question."""
    logging.info("\n--- LLM Sanity Test Starting ---")
    test_prompt = "What is the capital of France?"
    
    try:
        # Run inference using the defined local_llm_inference function
        response = local_llm_inference(test_prompt, max_tokens=50, temperature=0.1)
        
        logging.info(f"❓ Test Prompt: {test_prompt}")
        logging.info(f"✅ LLM Response: {response.split('.')[0]}...") # Log a snippet
        logging.info("--- LLM Sanity Test Successful ---\n")
        return True
    except Exception as e:
        logging.critical(f"❌ LLM Sanity Test FAILED: {e}")
        raise e # Halt startup if the LLM cannot run a simple test

def format_query_output_as_table(results: List[Tuple], query_type: str) -> str:
    if not results:
        return ""
    
    # Impala DESCRIBE output headers are usually 'name', 'type', 'comment'
    headers = ['name', 'type', 'comment']
    
    # Calculate column widths for padding
    widths = [len(h) for h in headers]
    for row in results:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Build header row
    header_row = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    
    # Build separator row
    separator_row = "|:" + "-".ljust(widths[0], '-') + "-|:" + \
                    "-".ljust(widths[1], '-') + "-|:" + \
                    "-".ljust(widths[2], '-') + "-|"

    # Build data rows (left justified for 'name', 'type', 'comment')
    data_rows = []
    for row in results:
        data_rows.append(
            "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        )
        
    return f"\n{header_row}\n{separator_row}\n" + "\n".join(data_rows)

def initialize_database() -> SQLDatabase:
    logging.info("Attempting to connect to Impala...")

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

    tables_to_check = ["dim_product_catalogue", "dim_product_category", "dim_product_price"]
    
    # --- Database Connection Test with Schema Logging ---
    try:
        # Check connection status
        with engine.connect() as connection:
            # 1. Run a simple query (SELECT 1) to confirm successful connection
            connection.execute(text("SELECT 1"))
            logging.info("✅ Successfully connected to Impala.") 
            
            # 2. Iterate and describe each table
            for i, table_name in enumerate(tables_to_check):
                full_table_name = f"{settings.DATABASE}.{table_name}"
                test_query = f"DESCRIBE {full_table_name}"
                
                # Execute DESCRIBE
                result = connection.execute(text(test_query))
                raw_results = result.fetchall()
                result.close()
                
                # Log the output in the requested table format
                table_output = format_query_output_as_table(raw_results, test_query)
                logging.info(f"\n--- Executing Query {i+1}/{len(tables_to_check)}: '{test_query};' ---\n{table_output}")

    except Exception as e:
        # This catch is critical and will halt the app if the DB/schema check fails
        logging.critical(f"❌ Database connection or schema check failed: {e}")
        raise e
    # --- End Database Connection Test ---

    db = SQLDatabase(engine=engine, include_tables=tables_to_check, sample_rows_in_table_info=10)
    logging.info(f"Tables used by LangChain: {db.get_usable_table_names()}")
    return db

def build_chatbot_ui(llm, db: SQLDatabase) -> gr.Blocks:
    # Removed plot related imports from global scope, only local variables used here.
    with gr.Blocks(theme=gr.themes.Soft(), css=".control-height { height: 500px; overflow: auto; }") as demo:
        task_history = gr.State([])
        # Assuming the image path is available
        gr.Image(value="/home/cdsw/mbb-cldr.png", height=60, show_label=False, show_download_button=False, interactive=False, container=False)
        gr.Markdown(
            f"""<div style="text-align: center;"><h1>Local LLM 💬 with Iceberg 💎 Database</h1></div>"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height", type="messages")
                # output_plot component removed from the UI
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="List all products")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear History")

        # Function signature adjusted to remove gr.Plot
        def predict(query: str, history: list) -> Generator[Tuple[list, list], None, None]:
            """
            The core logic function that processes a user query and yields updates to the UI.
            This is a generator function to stream responses.
            """
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Thinking..."})
            
            # Adjusted yield to remove gr.Plot
            yield history, history

            # --- Step 1: Generate SQL Query (Robust Chain Restored) ---
            generated_sql = ""
            try:
                sql_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an expert SQL assistant. Your sole purpose is to generate a single, syntactically correct SQL query for an Impala database based on a user's question.\n\n"
                     "**CRITICAL RULES:**\n"
                     "1. **STUDY SCHEMA:** You MUST study the schema of each table, join the table if necessary to produce the correct SQL query.\n"
                     "2. **SINGLE QUERY ONLY:** You MUST generate only one single SQL query. Do not output multiple queries, comments, or any explanatory text. The entire output must be only the SQL statement.\n"
                     "Here is the relevant table schema: {table_info}\n"
                     "Limit the number of results to {top_k}."
                    ),
                    ("user", "{input}") 
                ])

                table_info = db.get_table_info()
                top_k = 50

                # Using RunnablePassthrough and local_llm_wrapper (required for local LLM)
                sql_query_chain = (
                    RunnablePassthrough.assign(
                        prompt=sql_generation_prompt
                    )
                    | RunnableLambda(local_llm_wrapper)
                    | StrOutputParser()
                )
                
                raw_llm_output = sql_query_chain.invoke({
                    "input": query, 
                    "table_info": table_info, 
                    "top_k": top_k
                })
                
                # --- START ROBUST SQL PARSING (Combined Cleanup and Safety Check) ---
                generated_sql = raw_llm_output.strip()
                
                # 1. Attempt to strip common LLM markdown wrappers first
                temp_sql = generated_sql
                if "```sql" in temp_sql:
                    temp_sql = temp_sql.split("```sql")[-1].split("```")[0].strip()
                elif "```" in temp_sql:
                    temp_list = temp_sql.split("```")
                    temp_sql = temp_list[-2].strip() if len(temp_list) > 1 else temp_sql
                elif "SQLQuery:" in temp_sql:
                    temp_sql = temp_sql.split("SQLQuery:")[-1].strip()
                
                if temp_sql:
                    generated_sql = temp_sql

                # 2. Aggressively find and keep only the SQL part by searching for the first known command
                upper_sql = generated_sql.upper()
                start_keywords = ["SELECT", "WITH", "DESCRIBE", "SHOW", "INSERT", "DELETE", "UPDATE"]
                sql_start_index = -1
                
                for keyword in start_keywords:
                    index = upper_sql.find(keyword)
                    if index != -1 and (sql_start_index == -1 or index < sql_start_index):
                        sql_start_index = index
                
                if sql_start_index != -1:
                    generated_sql = generated_sql[sql_start_index:].strip()

                # 3. Final cleanup and semicolon check
                generated_sql = generated_sql.strip()
                
                if generated_sql and generated_sql.find(';') != -1:
                    generated_sql = generated_sql.split(';')[0].strip() + ";"
                
                if generated_sql and (generated_sql.upper().startswith("SELECT") or generated_sql.upper().startswith("WITH")) and not generated_sql.endswith(';'):
                    generated_sql = generated_sql + ";"
                
                # CRITICAL SAFETY CHECK: Ensure output is not empty
                if not generated_sql.strip():
                    logging.critical(f"❌ LLM failed to produce ANY valid SQL statement.")
                    logging.critical(f"   Raw LLM Output (Unprocessed): '{raw_llm_output.strip()}'")
                    raise ValueError("LLM generated an empty or non-extractable SQL statement.")
                # --- END ROBUST SQL PARSING ---
                
                logging.info(f"\n--- single SQL ---\n{generated_sql}\n--------------------")

            except Exception as e:
                logging.error(f"Error during SQL generation: {e}", exc_info=True)
                error_message = f"❌ **Error during SQL generation:**\n\nI encountered an issue creating the SQL query. Please check the terminal logs for details.\n\n**Root Cause:** The LLM failed to generate a valid, extractable SQL statement."
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return
            
            history[-1] = {"role": "assistant", "content": f"🏃 Running query...\n```sql\n{generated_sql}\n```"}
            yield history, history

            # --- Step 2: Execute the SQL Query ---
            db_results = ""
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
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
                error_message = f"❌ **Error during database query:**\n\nThe database rejected the following query:\n```sql\n{generated_sql}\n```\n**Error Details:**\n`{e}`"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return

            # --- Step 3: Summarize Results for the User ---
            history[-1] = {"role": "assistant", "content": "✍️ Summarizing the results..."}
            yield history, history
            
            try:
                summarization_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Your task is to explain database query results to a user in a clear, conversational way.\n\n"
                     "**IMPORTANT:** The data from the database is often a list of tuples, like `[(value,)]` or `[(3,)]`. "
                     "You MUST extract the value from *inside* the tuple to form the answer. For example, if the result is `[(3,)]`, the answer is 3.\n\n"
                     "**Provide a direct, factual answer and do not ask any follow-up questions. Do not give false information especially when SQL queries did not work due to errors. Must answer in English.**"),
                    ("user", "My original question was: \"{question}\"\n\nHere is the data from the database:\n```{result}```")
                ])
                
                summarize_chain = summarization_prompt | RunnableLambda(local_llm_wrapper) | StrOutputParser()
                final_summary = summarize_chain.invoke({"question": query, "result": db_results})
                
                if not final_summary:
                    final_summary = "The model returned an empty response. This might be due to content filtering or a prompt issue."

                history[-1] = {"role": "assistant", "content": final_summary}
                yield history, history

            except Exception as e:
                logging.error(f"Error during summarization: {e}", exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error during summarization. Please check the terminal logs for the root cause."}
                yield history, history

        # Function signature adjusted to remove gr.Plot
        def clear_history() -> Tuple[list, list]:
            """Clears the chat history."""
            return [], []

        def reset_user_input():
            """Resets the user input box."""
            return gr.update(value="")

        submit_btn.click(
            predict, 
            [query_box, chatbot], 
            [chatbot, task_history]
        ).then(reset_user_input, [], [query_box])
        
        clear_btn.click(
            clear_history, 
            [], 
            [chatbot, task_history]
        )
        
        return demo

def main():
    global CHECKPOINT_PATH
    args = get_args()

    CHECKPOINT_PATH = args.checkpoint_path

    # 1. Initialize LLM (and load model/tokenizer)
    llm = initialize_llm()
    
    # 2. Test LLM response before proceeding
    test_llm_response()

    # 3. Initialize Database (and run schema checks)
    db = initialize_database()
    
    # 4. Build Gradio UI
    demo = build_chatbot_ui(llm, db)
    
    # 5. Launch Application
    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name
    )

if __name__ == "__main__":
    main()
