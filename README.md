# Local LLM -> Iceberg 💎 Database

![LLM-iceberg](https://github.com/user-attachments/assets/810fe476-1aed-4bb1-adb1-02b237d80457)

LLM can act as powerful translators, turning natural language questions into executable database queries. This blog post shows fully functional chatbot that allows users to ask questions in plain English, which are then translated into SQL, run against an Impala with Iceberg format database, and summarized back into a conversational response. 

## How it Works?
- The Gradio chatbot [app-gradio.py](app-gradio.py) utilizes the LangChain framework whereby the `SQLDatabase` class from `langchain_community.utilities` is the component responsible for connecting to the database and providing its schema. Under the hood, `SQLDatabase` acts as a wrapper around the database connection. `db = SQLDatabase.from_uri` establishes a connection and includes methods to inspect the database's structure. `SQLDatabase` connects to the database and can pull the schema (table names, column details, data types, and sample rows).
- `create_sql_query_chain` function automatically calls the appropriate methods on the `db` object to get the schema information. It then formats this schema information and inserts it into the prompt that is sent to the LLM (ChatOpenAI) which is powered by locally hosted model, giving the model the necessary context to write a correct SQL query based on the user's question.
- Finally, `QuerySQLDataBaseTool` takes this generated SQL string and runs it to retrieve the actual data from the database.
- The chatbot connects to a locally hosted model served via a FastAPI endpoint. The model is loaded into the memory of an NVIDIA A100 80GB GPU for inference, but can also run on a CPU.


## Platform Requirement
☑️ Python 3.11

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

☑️ Cloudera Data Warehouse (CDW) 1.5.x (with Iceberg)

## How to Setup?

1. Create a new CAI project.
   
2. Install python libraries.
  ```
  pip install pydantic httpx uvicorn fastapi torch transformers ipywidgets pandas SQLAlchemy langchain_openai langchain_community langchain langchain_core
  ```

3. Download the foundational LLM into the project of the CAI/CML platform using either `git clone` or `wget`. `Qwen2.5-7B-Instruct` model is selected due to its Text-to-SQL and multilingual capabilities. 
  ```
  git lfs clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
  ```

4. To serve the local LLM, create an Application `fastapi-llm` using this [run-fastapi.py](run-fastapi.py) script. This script will run [fastapi-llm.py](fastapi-llm.py) with specific parameters. Upon successful run, take not of the Application URL.
<img width="700" height="687" alt="image" src="https://github.com/user-attachments/assets/365712a1-fee1-4e88-91ae-b7802c99fb2c" />

5. Next, ensure that Impala Virtual Warehouse in CDW is up and running. Copy the JDBC URL (LDAP) details.
<img width="700" height="669" alt="image" src="https://github.com/user-attachments/assets/d9b24ea3-9c00-4698-82b1-ccc4e0625e64" />

6. Navigate to CAI, modify the following settings in [app-gradio.py](app-gradio.py) based on the captured JDBC URL details and Application URL `fastapi-llm`.

```
class AppSettings(BaseModel):
    IMPALA_HOST: str = 'IMPALA_URL'
    IMPALA_PORT: int = 443
    USERNAME: str = 'LDAPUSER'
    PASSWORD: str = 'LDAPPASSWORD'
    HTTP_PATH: str = '/cliservice'
    DATABASE: str = 'DATABASENAME'
    LLM_ENDPOINT_URL: str = 'https://fastapi-llm.ml-9df5bc51-1da.apps.cdppvc.ares.olympus.cloudera.com/v1'
```

7. To host the frontend Gradio chabot, create an Application `ask-iceberg` using this [run-gradio.py](run-gradio.py) script. This script will run [app-gradio.py](app-gradio.py) with specific parameters. This script doesn't require GPU to run.
<img width="800" height="323" alt="image" src="https://github.com/user-attachments/assets/112f4c47-4a35-40b0-baba-b87a38f977c3" />

## Test the Chatbot 🚀

1. Insert some data into the Iceberg tables using [create_iceberg.py](create_iceberg.py).

2. Open the `ask-iceberg` URL and ask any question pertaining to the Iceberg tables. In response to the question, the chatbot will work with LLM and LangChain SQL tools to run the associated SQL query as described in the `gradio.log` below.

```
2025-08-17 04:24:06,670 - INFO - 
--- Executing SQL on Remote DB ---
SELECT p.plan_name, p.monthly_fee, p.data_allowance_gb, p.voice_minutes, p.sms_allowance
FROM plans p
WHERE p.plan_type = 'Prepaid';
--------------------
2025-08-17 04:24:06,681 - INFO - Using database dlee_telco as default
2025-08-17 04:24:06,892 - INFO - 
--- Results from DB ---
[('Basic Prepaid', 10.0, 5, 100, 50), ('Standard Prepaid', 20.0, 15, 300, 100), ('Data Hog Prepaid', 35.0, 50, 50, 50)]
--------------------
2025-08-17 04:24:10,742 - INFO - HTTP Request: POST https://fastapi-llm.ml-9df5bc51-1da.apps.cdppvc.ares.olympus.cloudera.com/v1/chat/completions "HTTP/1.1 200 OK"
```

3. In the `Impala Coordinator Web UI`, verify that the chatbot has triggered SQL query remotely to CDW successfully.

<img width="700" height="598" alt="image" src="https://github.com/user-attachments/assets/1eb138d1-677a-43cf-8d69-3c201b3863e4" />

<img width="700" height="188" alt="image" src="https://github.com/user-attachments/assets/fee40e21-207b-48d1-9b7f-4e0c34fc3ffd" />

4. To test Iceberg Time Travel feature, insert another batch of data (50 rows of customers) using [append_iceberg.py](append_iceberg.py) script.

5. Using Hue, verify that the customers table, for instance, has 2 batches of data in its history table.

<img width="700" height="573" alt="image" src="https://github.com/user-attachments/assets/26f8eeec-e9b9-48b9-8420-d6b0d738c771" />

6. Ask the chatbot about the customers with specific `system time`. Note that the chatbot has returned correct response based on the system time when data was inserted into the table.

```
how many customers exist in the table based on SYSTEM TIME 2025-08-18 03:34:59?
```

7. Ask the chatbot about the customers without specific `system time`, `how many customers registered since 2025-08-18?`. Note that the chatbot has returned correct response based on the registration time as per the column date.

<img width="700" height="525" alt="image" src="https://github.com/user-attachments/assets/92f75569-9107-460b-8266-f90d2daf31e5" />

Log:
```
--- Generated SQL ---
SELECT COUNT(*) 
FROM customers 
WHERE registration_date >= '2025-08-18';
--------------------
```
 


