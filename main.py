import streamlit as st
import pandas as pd
import json
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig, SessionExecuteRequest

sandboxID = None

st.title("AutoML Agent")
st.markdown("Upload a CSV file to get started.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
st.info("⬆️ Upload a CSV file to get started")

def summarize_dataframe(dataframe: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset for LLM context.

    This function creates a detailed text summary that includes:
    - Column data types and schema information
    - Missing value counts per data completeness
    - Cardinality (unique value counts) for each column
    - Statistical summaries for numerical columns
    - Sample data rows in CSV format

    Args:
        dataframe (pd.DataFrame): The input DataFrame to summarize

    Returns:
        A formatted string containing the dataset summary
    """
    try:
        buffer = io.StringIO()

        sample_rows = min(30, len(dataframe))

        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_data_csv = buffer.getvalue()

        dtypes = dataframe.dtypes.astype(str).to_dict()

        non_null_counts = dataframe.notnull().sum().to_dict()

        null_counts = dataframe.isnull().sum().to_dict()
        
        nunique = dataframe.nunique(dropna=True).to_dict()

        numeric_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
        
        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else {}

        lines = []

        lines.append("Schema (dtype):")
        for col, dtype in dtypes.items():
            lines.append(f"- {col}: {dtype}")
        lines.append("")

        lines.append("Null/Non-null counts:")
        for col in dataframe.columns:
            lines.append(f"- {col}: nulls={int(null_counts[col])}, non_nulls={int(non_null_counts[col])}")
        lines.append("")

        lines.append("Cardinality (unique values):")
        for col, unique_count in nunique.items():
            lines.append(f"- {col}: {unique_count}")
        lines.append("")

        if desc:
            lines.append("Numeric summary stats (describe):")
            for stat, values in desc.items():
                stat_line = ", ".join([f"{col}:{round(float(val), 2) if pd.notnull(val) else 'NaN'}" for col, val in values.items()])
                lines.append(f"- {stat}: {stat_line}")
        lines.append("")

        lines.append("Sample rows (CSV head):")
        lines.append(sample_data_csv)

        summary_text = "\n".join(lines)
        return summary_text
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"
    


def build_cleaning_prompt(df, selected_column):
    data_summary = summarize_dataframe(df)

    prompt = f"""
    You are an expert data scientist, extremely skilled in data cleaning and preprocessing 
    with more than 20 years of experience. 

    You are given a dataframe and you need to clean the data.
    Here is the summary of the data:
    {data_summary}

    Please clean the data and return the cleaned data.
    Make sure to handle the following:
    - Missing values
    - Outliers
    - Duplicate values
    - Standardize the data accordingly
    - Use one-hot encoding for categorical variables
    - Detect and drop identifier-like columns (e.g., names, IDs, emails, or other columns with extremely high cardinality).
    - Do not hard-code specific column names. Apply a general rule.

    ## IMPORTANT
    - Use exactly 4 spaces per indentation level. Do NOT use tabs. Ensure consistent indentation throughout.
    - The target column "{selected_column}" must NOT be modified, encoded, or standardized.
    - Keep the target column in the final output exactly as it is.
    - Apply cleaning only to the **feature columns**.
    - Print the shape of the dataframe before and after cleaning.

    THE GENERATED SCRIPT MUST BE FREE OF SYNTAX AND INDENTATION ERRORS

    ## REQUIREMENTS
    - Generate a standalone Python script inside a JSON property called "script".
    - The script should:
        1. SCRIPT MUST BE FREE OF SYNTAX AND INDENTATION ERRORS
        2. Read the dataset from "input.csv"
        3. Clean only the feature columns
        4. Keep the target column unchanged
        5. Save the final dataset (features + target) into "cleaned_data.csv"
    
    - Do NOT print anything to stdout or stderr.
    """
    return prompt



def check_dataframe(df, selected_column):
    data_summary = summarize_dataframe(df)

    prompt = f"""
    You are an expert machine learning engineer with more than 20 years of experience.
    You have been given a cleaned dataset called "cleaned_data.csv".  
    Your task is to **verify whether the dataset is fit for training a machine learning model**.  

    You are provided with the following information:
    - The dataset summary:
    {data_summary}

    - The target column is: "{selected_column}"

    Please check the following aspects carefully:
    1. **Missing Values** – confirm there are none left in the features or target.  
    2. **Duplicates** – confirm no duplicate rows exist.  
    3. **Outliers** – check whether extreme values are still present and if they are acceptable.  
    4. **Feature Scaling** – ensure numerical features are standardized or normalized as needed.  
    5. **Encoding** – confirm categorical features are properly one-hot encoded.  
    6. **Class Balance** – check if the target column has a reasonable class distribution (not heavily imbalanced).  
    7. **Data Types** – ensure all features are in the correct format for model training.  
    8. **Dataset Shape** – confirm there are enough rows and columns for meaningful training. 
    9. **No Data Leakage** – ensure no target information is present in the features. 
    10. **Unnessary Object Columns** – check for and flag any irrelevant or redundant features.

    ### OUTPUT FORMAT:
    Return your findings in a JSON object with the following properties:
    - "fit_for_training": true/false  
    - "issues": list of strings describing any problems found (empty list if none)  
    - "recommendations": list of suggestions for improvement (empty list if none)  

    Do NOT print anything to stdout or stderr.
    """
    return prompt

def build_fix_prompt(script: str, check_result: dict, df: pd.DataFrame, target: str) -> str:
    issues = "\n".join(f"- {i}" for i in check_result.get("issues", []))
    recs = "\n".join(f"- {r}" for r in check_result.get("recommendations", []))

    prompt = f"""
    You are an expert data scientist, extremely skilled in data cleaning and preprocessing. 
    with more than 20 years of experience.
    You have previously generated a data cleaning script, but the cleaned dataset was found to have some issues.
    Your task is to update the script to resolve these issues and improve the data quality.

    Here is current dataset schema:
    {df.dtypes.to_string()}

    The target column is: {target}, which should be excluded from cleaning steps but retained in the final output.
    The previous cleaning/training script was:
    {script}

    Validation of the dataset returned the following issues:
    {issues}

    And the following recommendations:
    {recs}

    Update or regenerate the script so that it resolves the above issues 
    and incorporates the recommendations. Output only the corrected Python script.
    """
    return prompt


def build_split_prompt(df: pd.DataFrame, target: str):
    prompt = f"""
    You are an expert data scientist with 20+ years of experience 
    in data preprocessing and Machine Learning model training. 

    You have been given a cleaned dataset and your task is to split it 
    into training, validation, and test sets, and then run baseline models.

    The target column is: "{target}"

    ### Instructions:
    1. Split the data into features (X) and target (y).
    2. Perform a 3-way split: training (70%), validation (15%), and test (15%).
    3. Determine whether the target is for classification or regression:
       - If the target column is of type object or categorical, treat as classification.
       - If the target column is numeric but has few unique values (<= 20 or <5% of total rows), treat as classification.
       - Otherwise, treat as regression.
       - Store this decision in a boolean variable `is_classification`.
    4. If `is_classification` is True, stratify all splits by the target column. If False, do not stratify.
    5. Always use random_state=42 for reproducibility.
    6. Store the shapes of all splits (X_train, X_val, X_test, y_train, y_val, y_test) in a JSON field called "shapes".
    7. Suggest up to 5 ML models suitable for the dataset. Ensure diversity:
       - At least one linear model
       - At least one tree-based model
       - At least one ensemble/boosting model
       - Optionally one deep learning model (sklearn MLP)
    8. Implement the suggested models in a for-loop:
       - Train on X_train, y_train
       - Evaluate on X_val, y_val and X_test, y_test
       - If `is_classification` is True: compute accuracy, precision, recall, F1. Store results in JSON.
       - If `is_classification` is False: compute RMSE, MAE, R². Store results in JSON.
       - Save all metrics in a JSON field called "results".
    9. Keep configurations simple (no heavy hyperparameter tuning) so execution is fast.
    10. Save each model in memory, serialize it with joblib, and also encode it in Base64.
        - For each model, return a JSON entry containing both:
            - "filename": the suggested filename (e.g., "logistic_regression_model.pkl")
            - "content": the Base64-encoded string of the serialized model.
        - Collect all models in a JSON field called "model_files".

    ### REQUIREMENTS:
    - Generate a standalone Python script inside a JSON property called "script".
    - The script should:
        1. Read the dataset from "input.csv"
        2. Include all necessary imports (pandas, numpy, sklearn, etc.)
        3. Be executable without modification
        4. Automatically detect classification vs regression using the rules above
        5. Use the `is_classification` variable to guide stratification and metric computation
        6. Script must not throw SyntaxError, IndentationError, TypeError, or any other Exceptions when executed
        7. Script must return the JSON called "model_files" containing the model filenames.

    IMPORTANT:
    - The script must ONLY output the final JSON object with shapes and results to stdout.
    - Suppress or redirect all warnings, errors, and logs to stderr (not stdout).
    - Example: use `warnings.filterwarnings("ignore")` at the start of the script.


    ### OUTPUT FORMAT:
    Return a JSON object with the following structure:

    {{
      "shapes": {{
          "X_train": [n_rows, n_features],
          "X_val": [n_rows, n_features],
          "X_test": [n_rows, n_features],
          "y_train": [n_rows],
          "y_val": [n_rows],
          "y_test": [n_rows]
      }},
      "results": [
          {{
            "model": "model_name",
            "metrics": {{
              "accuracy": 0.95,
              "precision": 0.96,
              "recall": 0.94,
              "f1_score": 0.95,
            }}
          }},
          {{
            "model": "model_name",
            "metrics": {{
              "rmse": 2.5,
              "mae": 1.8,
              "r2": 0.85,
            }}
          }}
      ],
        "model_files": {{
            "logistic_regression_model": {{
                "filename": "logistic_regression_model.pkl",
                "content": "<base64 string>"
            }},
            "random_forest_model": {{
                "filename": "random_forest_model.pkl",
                "content": "<base64 string>"
            }},
            "gradient_boosting_model": {{
                "filename": "gradient_boosting_model.pkl",
                "content": "<base64 string>"
            }},
            "svm_model": {{
                "filename": "svm_model.pkl",
                "content": "<base64 string>"
            }},
            "mlp_model": {{
                "filename": "mlp_model.pkl",
                "content": "<base64 string>"
            }}
        }}
    }}
    to the stdout.

    Do NOT print anything to stdout or stderr.
    """
    return prompt




    

load_dotenv()

# def get_openai_script(prompt: str) -> str:
#     try:
#         client = OpenAI(
#             api_key=os.getenv("OPENAI_API_KEY"),
#         )
#         response = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system", "content": (
#                     "You are a senior data scientist. Always return a strict JSON object matching the user's requested schema."
#                 )},
#                 {"role": "user", "content": prompt}
#             ],
#             response_format={"type": "json_object"},
#         )
#         if not response or not getattr(response, 'choices', None):
#             return None
#         text = response.choices[0].message.content or ""

#         try:
#             data = json.loads(text)
#             script_value = data.get("script", None)
#             if isinstance(script_value, str) and script_value.strip():
#                 return script_value
#         except Exception as e:
#             print(f"Error parsing JSON response: {text}")
    
#     except Exception as e:
#         st.error(f"Error generating script: {str(e)}")
#         return
    
def get_openai_json(prompt: str, key: str = None) -> dict | str | None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a senior data scientist. Always return a strict JSON object matching the user's requested schema."
                )},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
        )

        if not response or not getattr(response, "choices", None):
            return None

        text = response.choices[0].message.content or ""
        data = json.loads(text)

        if key is None:  # return full JSON
            return data
        return data.get(key)

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None


def get_local_script(prompt: str) -> str:
    try:
        # Point client to local Ollama server
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # dummy, required but ignored by Ollama
        )

        # Try models in order
        # models_to_try = ["mistral-small:latest", "llama3:latest"]

        try:
            model="mistral-small:latest",
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are a senior data scientist. Always return a strict JSON object matching the user's requested schema."
                    )},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )

            if response and getattr(response, "choices", None):
                text = response.choices[0].message.content or ""

                try:
                    data = json.loads(text)
                    script_value = data.get("script", None)
                    if isinstance(script_value, str) and script_value.strip():
                        return script_value
                except Exception:
                    print(f"Error parsing JSON from {model}: {text}")
        
        except Exception as e:
            print(f"Model {model} failed: {e}")
            # continue  # try the next 

        st.error("All local models failed to generate a script.")
        return None

    except Exception as e:
        st.error(f"Error setting up Ollama client: {str(e)}")
        return None 
    
def execute_in_daytona(script: str, csv_bytes: bytes) -> str:
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("DAYTONA_API_KEY environment variable not set")
    
    client = Daytona(DaytonaConfig(api_key=api_key))
    global sandboxID

    if sandboxID:
        sandbox = client.find_one(f"{sandboxID}")
        print(f"Reusing existing sandbox {sandboxID}")
        sandbox.start()
    else:
        # print(f"Error retrieving existing sandbox {sandboxID}: {str(e)}")
        sandbox = client.create()
    # sandbox = client.create()
    exec_info = {"exit_code": None, "stdout": "", "stderr": ""}

    print("Executing in Daytona...")
    try:
        sandbox.fs.upload_file(csv_bytes, "input.csv")

        cmd = "python -u - <<'PY'\n" + script + "\nPY"
        result = sandbox.process.exec(cmd, timeout=600, env={"PYTHONUNBUFFERED": "1"})
        exec_info["exit_code"] = getattr(result, 'exit_code', None)
        exec_info["stdout"] = getattr(result, 'result', '')
        exec_info["stderr"] = getattr(result, 'stderr', '')

        try:
            cleaned_bytes = sandbox.fs.download_file("cleaned_data.csv")
            return cleaned_bytes, exec_info
        except Exception as e:
            exec_info["stderr"] += f"\nError downloading cleaned_data.csv: {str(e)}"
            return None, exec_info

    
    except Exception as e:
        print(f"Error uploading file to Daytona: {str(e)}")
        return None
    
    finally:
        sandbox.stop()
        sandboxID = sandbox.id

    


# --- Helpers ---
def display_cleaned_data(cleaned_bytes, label="Cleaned Data"):
    if cleaned_bytes:
        cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
        with st.expander(label):
            st.dataframe(cleaned_df.head())
        return cleaned_df
    else:
        st.error(f"Failed to get {label.lower()}.")
        return None

def display_validation_report(check_result, iteration=None):
    title = "Check Result" if iteration is None else f"Check Result (Iteration {iteration})"
    with st.expander(title):
        st.subheader("Validation Report")
        if check_result.get("fit_for_training"):
            st.success("Dataset is fit for training")
        else:
            st.warning("Dataset is **not yet fit** for training")

        if check_result.get("issues"):
            st.markdown("### Issues Found")
            for issue in check_result["issues"]:
                st.markdown(f"- {issue}")

        if check_result.get("recommendations"):
            st.markdown("### Recommendations")
            for rec in check_result["recommendations"]:
                st.markdown(f"- {rec}")

def safe_execute(script, input_csv_bytes, label="Daytona Execution"):
    result = execute_in_daytona(script, input_csv_bytes)
    if not result:
        st.error(f"{label} failed — no result returned.")
        return None, {}
    cleaned_bytes, exec_info = result
    with st.expander(label):
        st.write(exec_info)
    return cleaned_bytes, exec_info


# --- Main Flow ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    selected_column = st.selectbox("Select the target column for prediction", df.columns.tolist())

    if st.button("Run AutoML"):
        with st.spinner("Running AutoML..."):
            # Initial cleaning prompt
            cleaning_prompt = build_cleaning_prompt(df, selected_column)
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)

            script = get_openai_json(cleaning_prompt, key="script")
            with st.expander("Generated Script"):
                st.code(script, language="python")

            input_csv_bytes = df.to_csv(index=False).encode("utf-8")

            # First execution
            cleaned_bytes, exec_info = safe_execute(script, input_csv_bytes, "Daytona Execution Info")
            if not cleaned_bytes:
                st.stop()

            cleaned_df = display_cleaned_data(cleaned_bytes, "Cleaned Data")
            if cleaned_df is None:
                st.stop()

            # First validation
            check_prompt = check_dataframe(cleaned_df, selected_column)
            with st.expander("Check Prompt"):
                st.write(check_prompt)

            check_result = get_openai_json(check_prompt)
            display_validation_report(check_result)

            # Iterative fixing
            max_iterations = 3
            iteration = 0
            while not check_result.get("fit_for_training") and iteration < max_iterations:
                iteration += 1
                st.subheader(f"AutoML Iteration {iteration}")

                # Build Fix Prompt
                fix_prompt = build_fix_prompt(script, check_result, cleaned_df, selected_column)
                print(fix_prompt)
                with st.expander(f"Fix Prompt (Iteration {iteration})"):
                    st.markdown(fix_prompt)

                # Generate new script
                new_script = get_openai_json(fix_prompt, key="script")
                with st.expander(f"Improved Script (Iteration {iteration})"):
                    st.code(new_script, language="python")

                input_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")

                # Execute new script
                cleaned_bytes, exec_info = safe_execute(new_script, input_csv_bytes, f"Daytona Execution Info (Iteration {iteration})")
                if not cleaned_bytes:
                    break

                cleaned_df = display_cleaned_data(cleaned_bytes, f"Cleaned Data (Iteration {iteration})")
                if cleaned_df is None:
                    break

                # Re-validate
                check_prompt = check_dataframe(cleaned_df, selected_column)
                with st.expander(f"Check Prompt (Iteration {iteration})"):
                    st.write(check_prompt)

                check_result = get_openai_json(check_prompt)
                display_validation_report(check_result, iteration=iteration)

                # carry forward script for next loop
                script = new_script

            # Final status
            if check_result.get("fit_for_training"):
                st.balloons()
                st.success("Dataset is now fit for training!")
                with st.expander("Final Cleaned Dataset"):
                    st.dataframe(cleaned_df.head())

                print(cleaned_df.shape)
                
                split_prompt = build_split_prompt(cleaned_df, selected_column)
                with st.expander("Data Splitting & Model Suggestion Prompt"):
                    st.write(split_prompt)
                split_script = get_openai_json(split_prompt, key="script")
                with st.expander("Data Splitting & Model Suggestion Script"):
                    st.code(split_script, language="python")

                input_csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")

                # input_csv_bytes = df.to_csv(index=False).encode("utf-8")

                split_bytes, exec_info = safe_execute(split_script, input_csv_bytes, "Data Splitting Execution Info")
                if split_bytes:
                    st.success("Data splitting script executed. Check execution info above.")
                print(exec_info.get("stdout", ""))
                # Suppose exec_info["stdout"] contains the JSON string
                stdout_json = exec_info.get("stdout", "").strip()

                if stdout_json:
                    try:
                        data = json.loads(stdout_json)  # parse JSON

                        # --- Show dataset splits ---
                        st.subheader("📊 Data Split Shapes")
                        st.dataframe(data["shapes"])  # pretty JSON display

                        # --- Show model results ---
                        st.subheader("🤖 Model Results")

                        results = pd.DataFrame(data["results"])
                        metrics_df = results.join(results.pop("metrics").apply(pd.Series))

                        # Turn model_files dict into a DataFrame for merging
                        model_files = data.get("model_files", {})
                        files_df = pd.DataFrame(list(model_files.items()), columns=["model", "model_file"])

                        # Merge model paths with metrics
                        metrics_df = metrics_df.merge(files_df, on="model", how="left")

                        import base64

                        # Add download column
                        def make_download_button(model_name):
                            if model_name in model_files and "content" in model_files[model_name]:
                                filedata = model_files[model_name]
                                file_bytes = base64.b64decode(filedata["content"])
                                return st.download_button(
                                    label=f"⬇️ Download {model_name}",
                                    data=file_bytes,
                                    file_name=filedata["filename"],
                                    mime="application/octet-stream",
                                    key=f"download_{model_name}"
                                )
                            else:
                                return "❌ Not available"

                        metrics_df["Download"] = metrics_df["model"].apply(make_download_button)

                        st.dataframe(metrics_df)

                        # Optional: highlight the best model
                        if "accuracy" in metrics_df.columns:
                            best_model = metrics_df.loc[metrics_df["accuracy"].idxmax()]
                            st.success(f"🏆 Best Model: {best_model['model']} (Accuracy={best_model['accuracy']:.3f})")

                    except json.JSONDecodeError:
                        st.error("❌ Failed to parse model output as JSON.")
                # print(split_bytes)
            else:
                st.warning("Reached max iterations but dataset is still not fit for training.")
