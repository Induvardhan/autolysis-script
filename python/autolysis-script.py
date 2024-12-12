import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

def ask_llm(prompt, max_tokens=500):
    """Helper function to query the LLM via AI Proxy."""
    aiproxy_token = os.getenv('AIPROXY_TOKEN')
    if not aiproxy_token:
        raise EnvironmentError("AIPROXY_TOKEN environment variable not set.")

    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aiproxy_token}"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract cost information from headers
        cost = response.headers.get('cost', 'Unknown')
        monthly_cost = response.headers.get('monthlyCost', 'Unknown')
        monthly_requests = response.headers.get('monthlyRequests', 'Unknown')
        
        print(f"Request Cost: ${cost}")
        print(f"Monthly Cost: ${monthly_cost}")
        print(f"Monthly Requests: {monthly_requests}")
        
        return response.json()['choices'][0]['message']['content'].strip()
    
    except requests.RequestException as e:
        print(f"Error in LLM request: {e}")
        return f"Error: {str(e)}"

def summarize_dataset(filename):
    """Loads and provides basic summary of the dataset using LLM."""
    data = pd.read_csv(filename)
    column_info = {
        col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)
    }
    missing_values = data.isnull().sum().to_dict()
    sample_data = data.head(5).to_dict()

    # Use LLM to generate a summary of the dataset
    prompt = (
        "You are a data analyst. Here is information about a dataset:\n"
        f"Columns and Types: {column_info}\n"
        f"Missing Values: {missing_values}\n"
        f"Sample Data: {sample_data}\n"
        "Summarize the dataset briefly and suggest initial analyses."
    )
    summary = ask_llm(prompt)
    return data, summary

def generate_analysis(data):
    """Ask the LLM to analyze the dataset and create visualizations."""
    # Summarize numerical columns for LLM input
    numeric_summary = data.describe().applymap(lambda x: x.item() if isinstance(x, (int, float)) else x)
    summary_stats = numeric_summary.to_dict()

    # Use LLM to suggest and generate analysis
    prompt = (
        "You are a Python programmer. Here is the numeric summary of a dataset:\n"
        f"{json.dumps(summary_stats, indent=2)}\n"
        "Write Python code to generate a meaningful visualization and insights."
    )
    analysis_code = ask_llm(prompt)

    # Execute the generated code safely
    exec_globals = {"pd": pd, "sns": sns, "plt": plt, "data": data}
    try:
        exec(analysis_code, exec_globals)
        insights = "Code executed successfully."
    except Exception as e:
        insights = f"Error executing LLM-generated code: {e}"
    return insights

def narrate_story(summary, insights):
    """Uses LLM to narrate a story based on the dataset analysis."""
    prompt = (
        "You are a storyteller. Here is a summary of a dataset:\n"
        f"{summary}\n"
        "Here are the insights from the analysis:\n"
        f"{insights}\n"
        "Write a compelling narrative about the dataset, including its implications and next steps."
    )
    story = ask_llm(prompt)
    return story

def write_results(summary, insights, story):
    """Writes the narrative and analysis to README.md."""
    with open("README.md", "w") as f:
        f.write("# Automated Analysis Report\n\n")
        f.write("## Dataset Summary\n")
        f.write(f"{summary}\n\n")
        f.write("## Insights\n")
        f.write(f"{insights}\n\n")
        f.write("## Narrative\n")
        f.write(f"{story}\n\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset_file = sys.argv[1]
    data, summary = summarize_dataset(dataset_file)
    insights = generate_analysis(data)
    story = narrate_story(summary, insights)
    write_results(summary, insights, story)

if __name__ == "__main__":
    main()