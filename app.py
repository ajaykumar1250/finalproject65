from flask import Flask, render_template, request, redirect, url_for, jsonify
import pyodbc
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from collections import Counter

app = Flask(__name__)

# Azure SQL Database connection
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=tcp:finalproject65.database.windows.net,1433;'
    'DATABASE=finalprojectdb;'
    'UID=kodaikar;'
    'PWD=#Ajaykumar1;'
    'MARS_Connection=yes;'
)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Route for Basket Analysis data using Machine Learning model
@app.route("/get_basket_analysis_ml", methods=["GET"])
def get_basket_analysis_ml():
    # Load transaction data
    query = """
    SELECT TOP 10000 Basket_num, Product_num
    FROM Transactions
    """
    df = pd.read_sql(query, conn)

    # Group products by Basket_num
    basket_groups = df.groupby("Basket_num")["Product_num"].apply(list)

    # Generate product pairs (combinations) for each basket
    product_pairs = Counter()
    for products in basket_groups:
        product_pairs.update(combinations(sorted(products), 2))

    # Convert to DataFrame for model input
    pairs_df = pd.DataFrame(
        [(pair[0], pair[1], count) for pair, count in product_pairs.items()],
        columns=["Product_A", "Product_B", "Frequency"]
    )

    # Encode Product_A and Product_B using LabelEncoder
    le = LabelEncoder()
    all_products = pd.concat([pairs_df['Product_A'], pairs_df['Product_B']]).unique()
    le.fit(all_products)
    pairs_df['Product_A_Encoded'] = le.transform(pairs_df['Product_A'])
    pairs_df['Product_B_Encoded'] = le.transform(pairs_df['Product_B'])

    # Feature engineering: we can use Frequency as the target variable
    X = pairs_df[['Product_A_Encoded', 'Product_B_Encoded']]
    y = pairs_df['Frequency']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Model evaluation
    y_pred = rf_model.predict(X_test)
    model_report = classification_report(y_test, y_pred, output_dict=True)

    # Get top product pairs and their frequencies
    top_pairs = pairs_df.sort_values(by="Frequency", ascending=False).head(10)

    # Format model report for easier parsing in the front-end
    formatted_report = {
        "classification_report": model_report,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": model_report['accuracy'],
        "recall": model_report['macro avg']['recall'],
        "f1_score": model_report['macro avg']['f1-score'],
        "auc": None  # This can be calculated separately, if needed
    }

    # Redirect to basket_analysis.html and pass the data as part of the render
    return render_template("basket_analysis.html", top_pairs=top_pairs.to_dict(orient='records'), model_report=formatted_report)


# Route for the index page (User Registration)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")

        # Display the welcome message briefly and redirect to index
        return redirect(url_for('index', username=username, email=email))

    # If a GET request or after redirection, pass the username and email for rendering
    username = request.args.get('username')
    email = request.args.get('email')

    return render_template("index.html", username=username, email=email)

# Route for the data upload page
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        table_name = request.form.get("table")
        file = request.files.get("file")

        if file and table_name:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            load_csv_to_sql(filepath, table_name)
            return redirect(url_for("search"))
    return render_template("upload.html")

# Function to load CSV data into SQL database
def load_csv_to_sql(filepath, table_name):
    df = pd.read_csv(filepath)

    cursor = conn.cursor()

    # Iterate over rows and insert only new entries
    for _, row in df.iterrows():
        # Build SQL query to insert new rows if they don't exist already
        if table_name == "Transactions":
            sql = f"""
                IF NOT EXISTS (
                    SELECT 1 FROM Transactions WHERE Basket_num = ? 
                )
                INSERT INTO Transactions (Basket_num, Hshd_num, PURCHASE_, Product_num, Spend, Units, STORE_R, Week_num, Year)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(sql, row['Basket_num'], row['Hshd_num'], row['PURCHASE_'], row['Product_num'],
                           row['Spend'], row['Units'], row['STORE_R'], row['Week_num'], row['Year'])

        elif table_name == "Households":
            sql = f"""
                IF NOT EXISTS (
                    SELECT 1 FROM Households WHERE Hshd_num = ? 
                )
                INSERT INTO Households (Hshd_num, Loyalty_flag, Age_range, Marital_status, Income_range, Homeowner_flag, Household_Composition, HH_size, Children)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """
            cursor.execute(sql, row['Hshd_num'], row['Loyalty_flag'], row['Age_range'], row['Marital_status'], row['Income_range'],
                           row['Homeowner_flag'], row['Household_Composition'], row['HH_size'], row['Children'])

        elif table_name == "Products":
            sql = f"""
                IF NOT EXISTS (
                    SELECT 1 FROM Products WHERE Product_num = ? 
                )
                INSERT INTO Products (Product_num, Department, Commodity, BRAND_TY, Natural_organic_flag)
                VALUES (?, ?, ?, ?, ?);
            """
            cursor.execute(sql, row['Product_num'], row['Department'], row['Commodity'],
                           row['BRAND_TY'], row['Natural_organic_flag'])

    conn.commit()

# Route for the search page
@app.route("/search", methods=["GET", "POST"])
def search():
    results = []
    if request.method == "POST":
        hshd_num = request.form.get("hshd_num")
        
        query = '''
            SELECT 
                T.Hshd_num, 
                T.Basket_num, 
                T.PURCHASE_, 
                T.Product_num, 
                P.Department, 
                P.Commodity,
                H.Loyalty_flag,
                H.Age_range,
                H.Marital_status,
                H.Income_range,
                H.Homeowner_flag,
                H.Household_composition,
                H.HH_size,
                H.Children,
                T.Spend,
                T.Units,
                T.STORE_R,
                T.Week_num,
                T.Year,
                P.BRAND_TY,
                P.Natural_organic_flag
            FROM 
                Transactions T
            JOIN 
                Households H ON T.Hshd_num = H.Hshd_num
            JOIN 
                Products P ON T.Product_num = P.Product_num
            WHERE 
                T.Hshd_num = ?
            ORDER BY 
                T.Hshd_num, 
                T.Basket_num, 
                T.PURCHASE_, 
                T.Product_num, 
                P.Department, 
                P.Commodity;
        '''
        
        cursor = conn.cursor()
        cursor.execute(query, (hshd_num,))
        results = cursor.fetchall()
    
    return render_template("search.html", results=results)

# Route for the dashboard page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Route for Demographics and Engagement data
@app.route("/get_demographics_data", methods=["GET"])
def get_demographics_data():
    query = """
    SELECT HH_size, Children, COUNT(*) as Frequency
    FROM Households
    GROUP BY HH_size, Children
    """
    df = pd.read_sql(query, conn)
    return jsonify(df.to_dict(orient='records'))

# Route for Engagement Over Time data
@app.route("/get_engagement_over_time", methods=["GET"])
def get_engagement_over_time():
    query = """
    SELECT Year, SUM(Spend) AS Total_Spend
    FROM Transactions
    GROUP BY Year
    ORDER BY Year
    """
    df = pd.read_sql(query, conn)
    return jsonify(df.to_dict(orient='records'))

# Route for Basket Analysis data
@app.route("/get_basket_analysis", methods=["GET"])
def get_basket_analysis():
    query = """
    SELECT Basket_num, Product_num
    FROM Transactions
    """
    df = pd.read_sql(query, conn)

    # Group products by Basket_num
    basket_groups = df.groupby("Basket_num")["Product_num"].apply(list)

    # Find product pairs and their frequencies
    from itertools import combinations
    from collections import Counter

    product_pairs = Counter()
    for products in basket_groups:
        product_pairs.update(combinations(sorted(products), 2))

    # Convert to DataFrame
    pairs_df = pd.DataFrame(
        [(pair[0], pair[1], count) for pair, count in product_pairs.items()],
        columns=["Product_A", "Product_B", "Frequency"]
    )

    # Get the top N product pairs
    top_pairs = pairs_df.sort_values(by="Frequency", ascending=False).head(10)

    return jsonify(top_pairs.to_dict(orient="records"))


# Route for Seasonal Trends data
@app.route("/get_seasonal_trends", methods=["GET"])
def get_seasonal_trends():
    query = """
    SELECT PURCHASE_, SUM(Spend) AS Monthly_Spend
    FROM Transactions
    GROUP BY PURCHASE_
    ORDER BY PURCHASE_
    """
    df = pd.read_sql(query, conn)
    return jsonify(df.to_dict(orient='records'))

# Route for Brand Preferences data
@app.route("/get_brand_preferences", methods=["GET"])
def get_brand_preferences():
    query = """
    SELECT BRAND_TY as Brand_type, COUNT(*) AS Frequency
    FROM Products
    GROUP BY BRAND_TY
    ORDER BY Frequency DESC
    """
    df = pd.read_sql(query, conn)
    return jsonify(df.to_dict(orient='records'))

# Route for Churn Prediction data
@app.route('/get_churn_predictions', methods=['GET'])
def get_churn_predictions():
    query = """
    SELECT Hshd_num, Loyalty_flag, Age_range, Marital_status, Income_range, 
           Homeowner_flag, Household_composition, HH_size, Children
    FROM Households
    """
    df = pd.read_sql(query, conn)

    # Convert 'HH_size' and 'Children' columns to numeric (int)
    df['HH_size'] = pd.to_numeric(df['HH_size'], errors='coerce')  # Handle non-numeric values gracefully
    df['Children'] = pd.to_numeric(df['Children'], errors='coerce')

    # Churn Prediction Logic
    def calculate_churn_risk(row):
        if row['Loyalty_flag'] == 'N' or (row['HH_size'] <= 2 and row['Children'] == 0):
            return 'High'
        return 'Low'

    df['Churn_Risk'] = df.apply(calculate_churn_risk, axis=1)

    # Aggregate data for graphical representation
    churn_by_age = df.groupby('Age_range')['Churn_Risk'].apply(lambda x: (x == 'High').sum()).reset_index()
    churn_by_age.columns = ['Age_range', 'High_Churn_Count']

    churn_distribution = df['Churn_Risk'].value_counts().reset_index()
    churn_distribution.columns = ['Churn_Risk', 'Count']

    return jsonify({
        'churn_by_age': churn_by_age.to_dict(orient='records'),
        'churn_distribution': churn_distribution.to_dict(orient='records')
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
