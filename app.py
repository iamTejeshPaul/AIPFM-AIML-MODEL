import os
import google.generativeai as genai
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from fpdf import FPDF
from datetime import datetime
import requests

# Configure Gemini AI API Key
genai.configure(api_key="AIzaSyC6vmcfzthKgnDu18c_DyK2le1qdFH9dUo")

# Firebase Database URL
FIREBASE_URL = "https://ai-manager-1086e-default-rtdb.firebaseio.com/users/{}.json"

# Load Model & Scaler
model = keras.models.load_model("spending_behavior_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

# Ensure directory exists
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Fetch Data for Selected User
@st.cache_data
def fetch_user_data(user_id):
    url = FIREBASE_URL.format(user_id)
    response = requests.get(url)
    if response.status_code == 200:
        user_data = response.json()
        if user_data:
            return user_data
        else:
            st.error(f"❌ No data found for user: {user_id}")
            return None
    else:
        st.error(f"❌ Error fetching data for user: {user_id}")
        return None

# Extract Expense Data
def extract_expense_data(user_data):
    expenses = user_data.get("expenses", {})
    income_data = user_data.get("income", {})
    goals_data = user_data.get("goals", {})

    latest_goal_amount = 0
    if goals_data:
        latest_goal_entry = max(goals_data.values(), key=lambda x: datetime.strptime(x["date"], "%m/%d/%Y, %I:%M:%S %p"))
        latest_goal_amount = int(latest_goal_entry.get("goalAmount", "0"))

    latest_income = max((int(entry.get("totalIncome", "0") or "0") for entry in income_data.values()), default=0)

    records = []
    for expense in expenses.values():
        date = pd.to_datetime(expense.get("date", "1970-01-01"))
        record = {
            "date": date,
            "food": int(expense.get("food", "0") or "0"),
            "entertainment": int(expense.get("entertainment", "0") or "0"),
            "transportation": int(expense.get("transportation", "0") or "0"),
            "medicines": int(expense.get("medicines", "0") or "0"),
            "clothing": int(expense.get("clothing", "0") or "0"),
            "total_amount": int(expense.get("totalAmount", "0") or "0"),
            "total_income": latest_income,
            "goal_amount": latest_goal_amount
        }
        records.append(record)
    return pd.DataFrame(records)

# Generate Bar Chart with Category Amounts
def generate_bar_chart(user_id, category_data):
    ensure_directory_exists("charts")
    plt.figure(figsize=(10, 5))

    bars = sns.barplot(x=list(category_data.keys()), y=list(category_data.values()), palette="coolwarm", edgecolor="black")

    # Annotate bars with the actual values
    for bar, value in zip(bars.patches, category_data.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'₹{value}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Categories")
    plt.ylabel("Amount Spent (INR)")
    plt.title(f'Spending Breakdown for {user_id}')
    plt.tight_layout()
    chart_path = f"charts/{user_id}_bar_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# Provide spending insights based on scenarios using Gemini API
def get_gemini_spending_insights(category_data, total_spending):
    categories = ", ".join([f"{category}: {amount} INR" for category, amount in category_data.items()])
    prompt = f"""
    Based on the following spending categories and predicted savings, provide detailed financial advice:
    Categories: {categories}
    Predicted Savings: {total_spending} INR

    Please analyze the data and provide insights on how to optimize spending, prioritize savings, and offer general financial advice.
    """

    # Create the model with generation config
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 40,
      "max_output_tokens": 8192,
      "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
      model_name="gemini-1.5-flash",
      generation_config=generation_config,
    )

    chat_session = model.start_chat(
      history=[]
    )

    # Send the message to Gemini
    response = chat_session.send_message(prompt)
    if response.text:
        return response.text.strip()
    else:
        return "AI analysis failed. Please try again later."

def spending_analysis(category_data, predicted_savings):
    insights = []
    total_spending = sum(category_data.values())

    if total_spending == 0:
        return "No spending data available."

    # AI-driven insights for spending using Gemini
    ai_insights = get_gemini_spending_insights(category_data, total_spending)
    insights.append(f"AIPFM Insights: \n{ai_insights}\n")

    # Add category-specific insights based on the percentages
    for category, amount in category_data.items():
        percentage = (amount / total_spending) * 100

        if percentage > 40:
            insights.append(f"Significant spending detected in {category}. Consider optimizing expenses and prioritizing essential needs.")
        elif percentage > 30:
            insights.append(f"Moderate expenditure in {category}. Keeping a close watch will help maintain financial stability.")
        elif percentage < 10:
            insights.append(f"Low allocation for {category}. Re-evaluating based on necessity and priority could be beneficial.")

    return "\n".join(insights)

# Function to calculate spending grade and savings grade
def calculate_grades(predicted_savings, total_income, total_spending):
    # Calculate Savings Grade
    if predicted_savings > total_income * 0.2:
        savings_grade = "Excellent"
    elif predicted_savings > total_income * 0.1:
        savings_grade = "Good"
    elif predicted_savings > total_income * 0.05:
        savings_grade = "Fair"
    else:
        savings_grade = "Poor"

    # Calculate Spending Ratio and Spending Grade
    spending_ratio = total_spending / total_income if total_income > 0 else 0
    if spending_ratio < 0.3:
        spending_grade = "Excellent"
    elif spending_ratio < 0.5:
        spending_grade = "Good"
    elif spending_ratio < 0.7:
        spending_grade = "Fair"
    else:
        spending_grade = "Poor"

    return savings_grade, spending_grade, spending_ratio

# Updated function to generate the PDF report
def generate_pdf_report(user_id, total_income, total_spending, predicted_savings, bar_chart_path, category_data, spending_advice, goal_amount, start_date, end_date):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", style='B', size=18)
    pdf.cell(200, 10, txt="AI Powered Finance Manager", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt=f"Spending Report for {user_id}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Total Income: INR {total_income}", ln=True)
    pdf.cell(200, 10, txt=f"Total Spending (Last 30 Days): INR {total_spending}", ln=True)
    pdf.cell(200, 10, txt=f"Goal Amount: INR {goal_amount}", ln=True)
    pdf.cell(200, 10, txt=f"Date Range: {start_date} to {end_date}", ln=True)

    # Calculate grades and ratios
    savings_grade, spending_grade, spending_ratio = calculate_grades(predicted_savings, total_income, total_spending)

    pdf.cell(200, 10, txt=f"Spending Ratio: {spending_ratio:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Spending Grade: {spending_grade}", ln=True)

    pdf.ln(10)

    pdf.set_font("Arial", style='B', size=10)
    pdf.cell(40, 10, "Category", border=1, align='C')
    pdf.cell(40, 10, "Amount (INR)", border=1, align='C')
    pdf.ln()

    pdf.set_font("Arial", size=10)
    for category, amount in category_data.items():
        pdf.cell(40, 10, category, border=1, align='C')
        pdf.cell(40, 10, f"INR {amount}", border=1, align='C')
        pdf.ln()

    pdf.ln(10)
    pdf.image(bar_chart_path, x=20, y=None, w=170)
    pdf.ln(10)

    # Spending Analytics Box
    pdf.set_font("Arial", style='B', size=12)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 10, "Spending Analytics:", border=1, ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 10, spending_advice.encode('latin-1', 'ignore').decode('latin-1'), border=1)
    pdf.ln(5)

    # Digitally Signed Note
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "AI Powered Finance Manager Development Team", ln=True, align='C')
    pdf.ln(10)

    # Ensure directory exists and generate the file path
    output_dir = "generated_reports"
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, f"{user_id}_report.pdf")

    # Save the PDF
    pdf.output(output_path, "F")

    # Return the file path
    return output_path

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Finance Manager", layout="wide")

    st.title("💰 AI Powered Finance Manager")
    
    user_id = st.text_input("Enter User ID:")
    
    if user_id:
        user_data = fetch_user_data(user_id)

        if user_data:
            st.subheader(f"📊 Expense Analysis for {user_id}")

            # Extract user expense data
            expense_df = extract_expense_data(user_data)

            if not expense_df.empty:
                # Date Range Picker for custom date selection
                st.write("### 🔍 Select a Date Range for Total Spending")
                start_date, end_date = st.date_input("Select Date Range", [expense_df["date"].min(), expense_df["date"].max()])
                
                # Filter the data based on the selected date range
                filtered_df = expense_df[(expense_df["date"] >= pd.to_datetime(start_date)) & (expense_df["date"] <= pd.to_datetime(end_date))]

                # Display filtered data
                st.write("### 📝 Filtered Expenses")
                st.dataframe(filtered_df[["date", "food", "entertainment", "transportation", "medicines", "clothing", "total_amount"]].sort_values(by="date", ascending=False))

                # Recalculate total spending and income within the selected date range
                category_data = {
                    "Food": filtered_df["food"].sum(),
                    "Entertainment": filtered_df["entertainment"].sum(),
                    "Transportation": filtered_df["transportation"].sum(),
                    "Medicines": filtered_df["medicines"].sum(),
                    "Clothing": filtered_df["clothing"].sum()
                }

                # Generate and display bar chart
                chart_path = generate_bar_chart(user_id, category_data)
                st.image(chart_path, caption="Spending Breakdown")

                # Prepare the model input with filtered data
                total_income = filtered_df["total_income"].max()
                total_spending = filtered_df["total_amount"].sum()
                input_data = [
                    total_income,          # total income
                    total_spending,       # total spending
                    category_data["Food"],           # food spending
                    category_data["Entertainment"],  # entertainment spending
                    category_data["Transportation"], # transportation spending
                    category_data["Medicines"]       # medicines spending
                ]
                
                # Transform the input using the scaler
                scaled_input = scaler.transform([input_data])

                # Predict the savings
                predicted_savings = model.predict(scaled_input)[0][0]

                # Automatically generate the PDF Report
                spending_advice = spending_analysis(category_data, predicted_savings)  # Fix: Pass predicted_savings
                goal_amount = filtered_df["goal_amount"].max()
        
                # Generate the PDF report in real-time
                pdf_path = generate_pdf_report(user_id, total_income, total_spending, predicted_savings, chart_path, category_data, spending_advice, goal_amount, start_date, end_date)
                
                # Use Streamlit's download button for the user to download the PDF
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name=f"{user_id}_report.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
