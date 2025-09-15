import streamlit as st
import pandas as pd
import json
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig, SessionExecuteRequest

st.title("AutoML Agent")
st.markdown("Upload a CSV file to get started.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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
    You are an expert data scientist, extremely skilled in data cleaning and preprocessing. 
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

    GENERATE A STANDALONE python script to clean the data, based on the data summary provided, in a json property called "script".
    Make sure you drop the target column "{selected_column}" from the input dataframe before cleaning.
    - DO NOT PRINT TO stdout or stderr.

    ## IMPORTANT
    - The script should be a pyhon script that can be executed to clean the data.  
    - The script should read the data from a file called "input.csv" and write the cleaned data to a file called "cleaned_data.csv".   
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

    The target column is: {target}

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
    sandbox = client.create()
    exec_info = {"exit_code": None, "stdout": "", "stderr": ""}

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
    
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    selected_column = st.selectbox("Select the target column for prediction", df.columns.tolist(), help="Choose the column you want to predict.")

    button = st.button("Run AutoML")

    if button:
        with st.spinner("Running AutoML..."):
            st.write(df.drop(columns=[selected_column]).head()) 
            st.write(f"Target Column: {selected_column}")
            cleaning_prompt = build_cleaning_prompt(df, selected_column)
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)
            script = get_openai_json(cleaning_prompt, key="script")
            # script = get_local_script(cleaning_prompt)
            print(script)
            with st.expander("Generated Script"):
                st.code(script)

            with st.spinner("Executing cleaning script in Daytona..."):
                input_csv_bytes = df.to_csv(index=False).encode('utf-8')
                cleaned_bytes, exec_info = execute_in_daytona(script, input_csv_bytes)

                with st.expander("Daytona Execution Info"):
                    st.write(exec_info)
                
                with st.expander("Cleaned Data"):
                    if cleaned_bytes:
                        cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                        st.dataframe(cleaned_df.head())

                    else:
                        st.error("Failed to get cleaned data.")
                
                with st.spinner("Validating cleaned data..."):
                    if cleaned_bytes:
                        cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                        check_prompt = check_dataframe(cleaned_df, selected_column)
                        with st.expander("Check Prompt"):
                            st.write(check_prompt)
                        check_result = get_openai_json(check_prompt)
                        print(check_result)
                        with st.expander("Check Result"):
                            st.subheader("Validation Report")

                            if check_result.get("fit_for_training"):
                                st.success("✅ Dataset is fit for training")
                            else:
                                st.warning("⚠️ Dataset is **not yet fit** for training")

                            # Show issues
                            if "issues" in check_result and check_result["issues"]:
                                st.markdown("### 🚨 Issues Found")
                                for issue in check_result["issues"]:
                                    st.markdown(f"- {issue}")

                            # Show recommendations
                            if "recommendations" in check_result and check_result["recommendations"]:
                                st.markdown("### 💡 Recommendations")
                                for rec in check_result["recommendations"]:
                                    st.markdown(f"- {rec}")
                            
                        max_iterations = 3
                        iteration = 0

                        while not check_result.get("fit_for_training") and iteration < max_iterations:
                            st.subheader(f"🔄 AutoML Iteration {iteration + 1}")

                            # Build Fix Prompt
                            fix_prompt = build_fix_prompt(script, check_result, cleaned_df, selected_column)
                            with st.expander(f"Fix Prompt (Iteration {iteration + 1})"):
                                st.write(fix_prompt)

                            # Generate new script
                            new_script = get_openai_json(fix_prompt, key="script")
                            with st.expander(f"Improved Script (Iteration {iteration + 1})"):
                                st.code(new_script, language="python")

                            # Execute script in Daytona
                            with st.spinner(f"Executing improved script in Daytona (Iteration {iteration + 1})..."):
                                cleaned_bytes, exec_info = execute_in_daytona(new_script, input_csv_bytes)

                                with st.expander(f"Daytona Execution Info (Iteration {iteration + 1})"):
                                    st.write(exec_info)

                                if cleaned_bytes:
                                    cleaned_df = pd.read_csv(io.BytesIO(cleaned_bytes))
                                    with st.expander(f"Cleaned Data (Iteration {iteration + 1})"):
                                        st.dataframe(cleaned_df.head())
                                else:
                                    st.error(f"❌ Failed to get cleaned data in Iteration {iteration + 1}")
                                    break  # stop loop if Daytona fails

                            # Validate cleaned data
                            with st.spinner(f"Re-validating cleaned data (Iteration {iteration + 1})..."):
                                if cleaned_bytes:
                                    check_prompt = check_dataframe(cleaned_df, selected_column)
                                    with st.expander(f"Check Prompt (Iteration {iteration + 1})"):
                                        st.write(check_prompt)

                                    check_result = get_openai_json(check_prompt)
                                    with st.expander(f"Check Result (Iteration {iteration + 1})"):
                                        st.subheader("Validation Report")
                                        if check_result.get("fit_for_training"):
                                            st.success("✅ Dataset is fit for training")
                                        else:
                                            st.warning("⚠️ Dataset is **not yet fit** for training")

                                        if "issues" in check_result and check_result["issues"]:
                                            st.markdown("### 🚨 Issues Found")
                                            for issue in check_result["issues"]:
                                                st.markdown(f"- {issue}")

                                        if "recommendations" in check_result and check_result["recommendations"]:
                                            st.markdown("### 💡 Recommendations")
                                            for rec in check_result["recommendations"]:
                                                st.markdown(f"- {rec}")
                                else:
                                    st.error(f"❌ No cleaned data to validate in Iteration {iteration + 1}")
                                    break

                            # Update for next loop
                            script = new_script  # carry forward the improved script
                            iteration += 1

                        if check_result.get("fit_for_training"):
                            st.balloons()
                            st.success("🎉 Dataset is now fit for training!")
                            with st.expander("Final Cleaned Dataset"):
                                st.dataframe(cleaned_df.head())



                    else:
                        st.error("No cleaned data to validate.")



