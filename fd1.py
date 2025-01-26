from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
import re
from langchain_groq import ChatGroq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# FastAPI initialization
app = FastAPI()

# Load datasets
food_data = pd.read_csv("nutrients_csvfile.csv")
user_data = pd.read_csv("food111.csv")

# Environment variables for LangChain
langsmith = "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"
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "DMCCLanggraph"

# Initialize LLM
llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0,groq_api_key="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")

# Data preparation for ML model
X = user_data[['age', 'weight', 'height', 'BMI', 'condition', 'activity_level']]
y = user_data[['protein', 'carbs', 'sugar', 'sodium', 'total_fat', 'cholesterol', 'dietary_fiber']]
X.fillna(method='ffill', inplace=True)
y.fillna(method='ffill', inplace=True)
X = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the ML model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# API request model
class QueryPayload(BaseModel):
    query: str

# Function to find foods based on nutrients
def find_foods_for_nutrient(nutrient: str, quantity: float):
    if nutrient not in food_data.columns:
        return None
    matching_foods = food_data[food_data[nutrient] >= quantity]
    return matching_foods

@app.post("/query/")
async def chatbot_query(payload: QueryPayload):
    user_query = payload.query

    # Regex to extract age, weight, height, BMI, condition, and activity level
    personal_details_regex = (
        r"age:\s*(\d+),\s*weight:\s*(\d+),\s*height:\s*(\d+),\s*BMI:\s*(\d+\.?\d*),\s*condition:\s*(\w+),\s*activity_level:\s*(\d+)"
    )
    match = re.match(personal_details_regex, user_query, re.IGNORECASE)

    if match:
        age = int(match.group(1))
        weight = int(match.group(2))
        height = int(match.group(3))
        bmi = float(match.group(4))
        condition = match.group(5).lower()
        activity_level = int(match.group(6))

        # Prepare the input for prediction
        new_user = pd.DataFrame([[age, weight, height, bmi, condition, activity_level]],
                                columns=["age", "weight", "height", "BMI", "condition", "activity_level"])
        new_user = pd.get_dummies(new_user, drop_first=True)

        # Reindex to match training data
        expected_columns = X.columns
        new_user = new_user.reindex(columns=expected_columns, fill_value=0)

        # Scale the input
        new_user_scaled = scaler.transform(new_user)

        # Predict the meal plan
        predicted_meal_plan = model.predict(new_user_scaled)[0]

        # Format the response
        nutrients = ["Protein", "Carbs", "Sugar", "Sodium", "Total Fat", "Cholesterol", "Dietary Fiber"]
        meal_plan_response = {nutrients[i]: f"{predicted_meal_plan[i]:.2f} g" for i in range(len(nutrients))}
        return {"response": "Personalized meal plan", "meal_plan": meal_plan_response}

    # Regex to extract nutrient and quantity
    nutrient_query_regex = r"(\d+\.?\d*)\s*(grams?|g|calories?|kcal|fat|carbs?|protein|fiber|sugar|sodium)"
    match = re.match(nutrient_query_regex, user_query, re.IGNORECASE)

    if match:
        quantity = float(match.group(1))
        nutrient = match.group(2).lower()

        # Map input to column names
        nutrient_map = {
            "protein": "Protein",
            "calories": "Calories",
            "fat": "Total Fat",
            "carbs": "Carbs",
        }
        nutrient = nutrient_map.get(nutrient, nutrient)

        # Find foods
        matching_foods = find_foods_for_nutrient(nutrient, quantity)

        if matching_foods is None:
            raise HTTPException(status_code=400, detail=f"'{nutrient}' is not a valid nutrient in the dataset.")

        if matching_foods.empty:
            return {"response": f"No foods found with at least {quantity} of {nutrient}."}

        # List matching foods
        food_list = [{"Food": row["Food"], nutrient: row[nutrient]} for _, row in matching_foods.iterrows()]
        return {"response": f"Foods with at least {quantity} {nutrient}", "foods": food_list}

    return {"response": "Invalid query. Please provide personal details or a nutrient query."}
if __name__ =="__manin__":
    import uvicorn
    uvicorn.run(app,host= "0.0.0.0",port=8000)