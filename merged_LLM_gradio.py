import torch
import os
import argparse
import time
import uvicorn
import gradio as gr
import pandas as pd
import logging
import httpx
import ast
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
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
from langchain_core.runnables import RunnablePassthrough 

model = None
tokenizer = None
CHECKPOINT_PATH = ""
os.environ["TRUST_REMOTE_CODE"] = "true"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppSettings(BaseModel):
    IMPALA_HOST: str = 'coordinator-maybank-impala.dw-maybank1-cdp-env.xfaz-gdb4.cloudera.site'
    IMPALA_PORT: int = 443
    USERNAME: str = 'dennislee'
    PASSWORD: str = 'Cloudera@123'
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
    
    response_text = generated_text.split(prompt)[-1].strip()
    
    return response_text

def initialize_llm() -> ChatOpenAI:
    global CHECKPOINT_PATH
    
    logging.info(f"Loading local LLM from checkpoint: {CHECKPOINT_PATH}")
    try:
        load_model_and_tokenizer(CHECKPOINT_PATH)
        logging.info("Local LLM model loaded successfully.")
        
        return ChatOpenAI(
            model='local-hf-model', 
            temperature=0.0, 
            base_url='http://dummy.url',
            api_key='dummy_key'
        )
        
    except Exception as e:
        logging.error("Failed to load local LLM", exc_info=True)
        raise e

def initialize_database() -> SQLDatabase:
    logging.info("Initializing Impala database connection...")
    
    connect_args = {
        "user": settings.USERNAME,
        "password": settings.PASSWORD,
        "auth_mechanism": 'LDAP',
        "use_ssl": True,
        "use_http_transport": True,
        "http_path": settings.HTTP_PATH,
        "ca_cert": None
    }

    base_uri = (
        f"impala://{settings.IMPALA_HOST}:{settings.IMPALA_PORT}/{settings.DATABASE}"
    )

    engine = create_engine(
        base_uri,
        creator=lambda: impala.dbapi.connect(
            host=settings.IMPALA_HOST,
            port=settings.IMPALA_PORT,
            database=settings.DATABASE,
            **connect_args
        )
    )

    tables = ["dim_product_catalogue", "dim_product_category", "dim_product_price"]
    db = SQLDatabase(engine=engine, include_tables=tables, sample_rows_in_table_info=10)
    
    logging.info(f"Database connection initialized. Discovered tables: {db.get_usable_table_names()}")
    return db

def build_chatbot_ui(llm: ChatOpenAI, db: SQLDatabase) -> Tuple[gr.Blocks, str]:
    
    custom_css = ".control-height { height: 500px; overflow: auto; }"

    with gr.Blocks() as demo:
        task_history = gr.State([])
        
        gr.Image(value="/home/cdsw/clouderalogo.png", height=40, show_label=False, interactive=False, container=False)
        
        # MODIFIED: Removed Iceberg Time Travel sentence
        gr.Markdown(
            f"""<div style="text-align: center;"><h1>Local LLM 💬 with Iceberg 💎 Database</h1><p>Ask any question about the <strong>{settings.DATABASE}</strong> ⛁.</p></div>"""
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="Chatbox", elem_classes="control-height")
                # MODIFIED: Updated placeholder question
                query_box = gr.Textbox(lines=3, label="Your Question", placeholder="e.g., Compare total revenue between prepaid and postpaid plans. How many customers exist in the table?")
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear History")
        
        def predict(query: str, history: list) -> Generator[Tuple[list, list], None, None]:
            
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "🤔 Thinking..."})
            yield history, history

            generated_sql = ""
            try:
                # MODIFIED: Removed the CRITICAL RULE for ICEBERG TIME TRAVEL
                sql_generation_prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "You are an expert SQL assistant. Your sole purpose is to generate a single, syntactically correct SQL query for an Impala database based on a user's question.\n\n"
                     "**CRITICAL RULES:**\n"
                     "1. **STUDY SCHEMA:** You MUST study the schema of each table, join the table if necessary to produce the correct SQL query.\n"
                     "2. **SINGLE QUERY ONLY:** You MUST generate only one single SQL query. Do not output multiple queries, comments, or any explanatory text. The entire output must be only the SQL statement.\n\n"
                     "Here is the relevant table schema: {table_info}\n"
                     "Limit the number of results to {top_k}."
                    ),
                    ("user", "{input}") 
                ])

                # Manually render SQL prompt to bypass LangChain internal issues
                table_info = db.get_table_info()
                prompt_template = sql_generation_prompt
                
                raw_prompt_messages = prompt_template.invoke({
                    "table_info": table_info,
                    "top_k": 5, 
                    "input": query
                })
                
                # Robustly extract prompt string content
                raw_prompt_parts = []
                for msg in raw_prompt_messages:
                    if hasattr(msg, 'content'):
                        raw_prompt_parts.append(msg.content)
                    elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
                        raw_prompt_parts.append(str(msg[1]))
                    else:
                        raw_prompt_parts.append(str(msg))
                        
                raw_prompt = "\n".join(raw_prompt_parts)
                
                raw_llm_output = local_llm_inference(raw_prompt, max_tokens=1024, temperature=0.2)
                
                generated_sql = raw_llm_output.strip()
                
                # --- ENHANCED FIX FOR MALFORMED SQL ---
                
                # 1. Strip triple quotes and variable assignment first
                generated_sql = generated_sql.replace('"""', '').strip()
                if generated_sql.lower().startswith('sql_query ='):
                    generated_sql = generated_sql.split('=', 1)[-1].strip().strip('\'"').strip()
                
                # 2. Strip markdown blocks
                if "```sql" in generated_sql:
                    generated_sql = generated_sql.split("```sql")[1].split("```")[0].strip()
                
                # 3. Strip SQLQuery: label
                if "SQLQuery:" in generated_sql:
                    generated_sql = generated_sql.split("SQLQuery:")[-1].strip()
                
                # 4. Strip single/double quotes, and final strip of leading/trailing whitespace
                generated_sql = generated_sql.strip('\'" \n')
                
                # 5. Ensure it ends with a semicolon and only contains a single statement
                if generated_sql and not generated_sql.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                     raise ValueError("LLM generated output that is not a valid SQL command.")
                     
                if generated_sql and not generated_sql.endswith(';'):
                    # If semicolon exists internally, take only the first statement
                    generated_sql = generated_sql.split(";")[0].strip() + ";"
                elif ";" in generated_sql:
                    # If semicolon exists internally, take only the first statement
                    generated_sql = generated_sql.split(";")[0].strip() + ";"
                # --- END ENHANCED FIX ---
                
                logging.info(f"\n--- single SQL ---\nsql_query = \"\"\"\n{generated_sql}\n\"\"\"\n--------------------")

            except Exception as e:
                logging.error(f"Error during SQL generation: {e}", exc_info=True)
                error_message = f"❌ **Error during SQL generation:**\n\nI encountered an issue creating the SQL query. Please check the terminal logs for details."
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return
            
            history[-1] = {"role": "assistant", "content": f"🏃 Running query...\n```sql\n{generated_sql}\n```"}
            yield history, history

            db_results = ""
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
                error_message = f"❌ **Error during database query:**\n\nThe database rejected the following query:\n```sql\n{generated_sql}\n```\n**Error Details:**\n`{e}`"
                history[-1] = {"role": "assistant", "content": error_message}
                yield history, history
                return

            history[-1] = {"role": "assistant", "content": "✍️ Summarizing the results..."}
            yield history, history
            
            try:
                summarization_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful AI assistant. Your task is to explain database query results to a user in a clear, conversational way.\n\n"
                    "**CRITICAL RULE:** The raw data from the database is for your internal calculation and reference only. **You MUST NOT mention, quote, or include the raw result format** (e.g., `[(14,)]`, `[(value,)]`, or `[{result}]` block) in your final response to the user.\n\n"
                    "**IMPORTANT:** The data from the database is often a list of tuples, like `[(value,)]` or `[(3,)]`. You MUST extract the value from *inside* the tuple to form the answer. For example, if the result is `[(3,)]`, the answer is 3.\n\n"
                    "**Provide a direct, factual answer and do not ask any follow-up questions. Do not give false information especially when SQL queries did not work due to errors. Must answer in English.**"),
                    ("user", "My original question was: \"{question}\"\n\nHere is the data from the database:\n```{result}```")
                ])
                
                # Manually invoke and extract summary prompt content
                summary_prompt_template = summarization_prompt

                raw_summary_messages = summary_prompt_template.invoke({
                    "question": query,
                    "result": db_results
                })

                summary_prompt_parts = []
                for msg in raw_summary_messages:
                    if hasattr(msg, 'content'):
                        summary_prompt_parts.append(msg.content)
                    elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
                        summary_prompt_parts.append(str(msg[1]))
                    else:
                        summary_prompt_parts.append(str(msg))
                        
                summary_prompt = "\n".join(summary_prompt_parts)

                final_summary = local_llm_inference(summary_prompt, max_tokens=1024, temperature=0.7)
                
                if not final_summary:
                    final_summary = "The model returned an empty response. This might be due to content filtering or a prompt issue."

                history[-1] = {"role": "assistant", "content": final_summary}
                yield history, history

            except Exception as e:
                logging.error(f"Error during summarization: {e}", exc_info=True)
                history[-1] = {"role": "assistant", "content": f"❌ Error during summarization. Please check the terminal logs for the root cause."}
                yield history, history

        def clear_history() -> Tuple[list, list]:
            return [], []

        def reset_user_input():
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
        
        return demo, custom_css

def main():
    global CHECKPOINT_PATH
    args = get_args()
    
    CHECKPOINT_PATH = args.checkpoint_path

    server_port = args.server_port
    server_name = args.server_name

    print(f"Starting Gradio application with LLM checkpoint: {CHECKPOINT_PATH} on {server_name}:{server_port}")
    
    try:
        llm = initialize_llm()
        db = initialize_database()
        
        demo, custom_css = build_chatbot_ui(llm, db)
        
        demo.queue().launch(
            share=args.share,
            inbrowser=args.inbrowser,
            server_port=server_port,
            server_name=server_name,
            css=custom_css 
        )
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")

if __name__ == "__main__":
    main()
