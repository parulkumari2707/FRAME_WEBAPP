'''
import os
import pandas as pd
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns
from frame.frame_selector import FRAMESelector # Assuming installed as library
from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  # Regression or Classification

app = Flask(__name__, template_folder=os.path.join('app', 'templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/images'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_features = []
    show_viz = False
    error = None
    columns = None  # To hold the columns of the uploaded CSV
    target_column = None  # To hold the user-selected target column

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                df = pd.read_csv(filepath)
                columns = df.columns.tolist()  # Extract columns from the CSV file

                # Check if the CSV file is empty
                if df.empty:
                    error = "CSV file is empty. Please upload a valid CSV file."
                    return render_template("index.html", error=error, columns=columns)

        # Check if a target column is selected
        target_column = request.form.get('target_column')

        if target_column:
            # Ensure the selected target column is present in the dataframe
            if target_column not in df.columns:
                error = f"The selected target column '{target_column}' is not present in the CSV."
                return render_template("index.html", error=error, columns=columns)

            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Determine if the target column is for classification or regression
            target_type = type_of_target(y)
            if target_type in ['continuous', 'binary', 'multiclass']:
                if target_type == 'continuous':
                    model = RandomForestRegressor()  # Use RandomForestRegressor for regression tasks
                else:
                    model = RandomForestClassifier()  # Use RandomForestClassifier for classification tasks

                # Initialize FRAMESelector with the appropriate model
                selector = FRAMESelector(model=model)
                X_selected = selector.fit_transform(X, y)
                selected_features = selector.selected_features_

                if request.form.get('show_viz') == 'yes':
                    show_viz = True

                    # Feature Importance Plot
                    importances = model.feature_importances_
                    imp_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
                    imp_df = imp_df.sort_values(by='Importance', ascending=False)

                    plt.figure(figsize=(10, 5))
                    sns.barplot(x='Importance', y='Feature', data=imp_df)
                    plt.tight_layout()
                    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'importance.png'))
                    plt.close()

                    # Heatmap of selected features
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(X_selected.corr(), annot=True, cmap='coolwarm')
                    plt.tight_layout()
                    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], 'heatmap.png'))
                    plt.close()

            else:
                error = "Target column must be continuous or categorical."
                return render_template("index.html", error=error, columns=columns)

    return render_template("index.html", selected_features=selected_features, show_viz=show_viz, error=error, columns=columns, target_column=target_column)

if __name__ == '__main__':
    app.run(debug=True)

'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.utils.multiclass import type_of_target

# Import FRAMESelector from your library
from frame.frame_selector import FRAMESelector

# Create Flask app with proper static folder configuration
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the CSV file
            df = pd.read_csv(filepath)
            # Get the column names
            columns = df.columns.tolist()
            # Create a session_id
            session_id = filename.split('.')[0]
            
            return jsonify({
                'columns': columns,
                'session_id': session_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    session_id = data.get('session_id')
    target_column = data.get('target_column')
    num_features = int(data.get('num_features', 5))
    top_k = int(data.get('top_k', 10))
    
    if not session_id or not target_column:
        return jsonify({'error': 'Missing session_id or target_column'}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.csv")
        df = pd.read_csv(filepath)
        
        # Check if target column exists
        if target_column not in df.columns:
            return jsonify({'error': f'Target column {target_column} not found in dataset'}), 400
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values for numerical columns
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            X[col] = X[col].fillna(X[col].mean())
        
        # Handle missing values for categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # Get column types for informational purposes
        column_types = {}
        for col in X.columns:
            column_types[col] = str(X[col].dtype)
        
        # Better detection of regression vs classification task
        is_classification = False
        unique_values = len(y.unique())
        
        # More robust checks for classification vs regression:
        # 1. Check if y contains float values
        has_float_values = y.dtype == float and not np.all(y == y.astype(int))
        
        # 2. Check if number of unique values is high compared to dataset size
        # More than 20% unique values often suggests regression
        high_cardinality = unique_values > min(20, len(y) * 0.2)
        
        # 3. Check if values follow an even distribution (like 0.1 increments)
        is_regression = has_float_values or high_cardinality
        is_classification = not is_regression
        
        # Define model based on detected task type
        if is_classification:
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        else:
            model = XGBRegressor()
        
        # Apply feature selection - using safe values for top_k and num_features
        safe_top_k = min(top_k, len(X.columns))
        safe_num_features = min(num_features, safe_top_k)
        
        frame_selector = FRAMESelector(model=model, num_features=safe_num_features, top_k=safe_top_k)
        frame_selector.fit(X, y)
        
        # Get selected features
        selected_features = frame_selector.selected_features_
        
        # Store results for visualization
        result = {
            'session_id': session_id,
            'selected_features': selected_features,
            'target_column': target_column,
            'column_types': column_types,
            'is_classification': is_classification
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    session_id = data.get('session_id')
    target_column = data.get('target_column')
    selected_features = data.get('selected_features')
    is_classification = data.get('is_classification', True)
    
    if not session_id or not target_column or not selected_features:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}.csv")
        df = pd.read_csv(filepath)
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            X[col] = X[col].fillna(X[col].mean())
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
        
        # 1. Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss') if is_classification else XGBRegressor()
        model.fit(X[selected_features], y)
        feature_importance = model.feature_importances_
        
        # Sort features by importance
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save to a bytes buffer
        importance_buffer = io.BytesIO()
        plt.savefig(importance_buffer, format='png')
        importance_buffer.seek(0)
        importance_plot = base64.b64encode(importance_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        selected_df = X[selected_features].copy()
        # Include target column for correlation
        selected_df[target_column] = y
        
        # Calculate correlation matrix
        corr_matrix = selected_df.corr(numeric_only=True)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix of Selected Features')
        plt.tight_layout()
        
        # Save to a bytes buffer
        corr_buffer = io.BytesIO()
        plt.savefig(corr_buffer, format='png')
        corr_buffer.seek(0)
        corr_plot = base64.b64encode(corr_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. Model Performance Comparison
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model with all features
        model_all = XGBClassifier(use_label_encoder=False, eval_metric='logloss') if is_classification else XGBRegressor()
        model_all.fit(X_train, y_train)
        y_pred_all = model_all.predict(X_test)
        
        # Model with selected features
        model_selected = XGBClassifier(use_label_encoder=False, eval_metric='logloss') if is_classification else XGBRegressor()
        model_selected.fit(X_train[selected_features], y_train)
        y_pred_selected = model_selected.predict(X_test[selected_features])
        
        # Calculate performance metrics
        if is_classification:
            metric_all = accuracy_score(y_test, y_pred_all)
            metric_selected = accuracy_score(y_test, y_pred_selected)
            metric_name = "Accuracy"
        else:
            metric_all = r2_score(y_test, y_pred_all)
            metric_selected = r2_score(y_test, y_pred_selected)
            metric_name = "R² Score"
            
            # Also calculate RMSE for regression problems
            rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))
            rmse_selected = np.sqrt(mean_squared_error(y_test, y_pred_selected))
        
        # Create plot
        plt.figure(figsize=(8, 6))
        models = ['All Features', 'Selected Features']
        metrics = [metric_all, metric_selected]
        
        bars = plt.bar(models, metrics, color=['skyblue', 'orange'])
        
        # For regression, R² can be negative so adjust y-limits
        if not is_classification:
            plt.ylim(min(0, metric_all, metric_selected) - 0.1, max(1, metric_all, metric_selected) + 0.1)
        else:
            plt.ylim(0, 1.1)
            
        plt.title(f'Model Performance Comparison ({metric_name})')
        plt.ylabel(metric_name)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to a bytes buffer
        perf_buffer = io.BytesIO()
        plt.savefig(perf_buffer, format='png')
        perf_buffer.seek(0)
        perf_plot = base64.b64encode(perf_buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Return all plots as base64 encoded strings
        result = {
            'importance_plot': importance_plot,
            'correlation_plot': corr_plot,
            'performance_plot': perf_plot,
            'metric_name': metric_name,
            'metric_all': float(metric_all),
            'metric_selected': float(metric_selected)
        }
        
        # Add RMSE metrics for regression
        if not is_classification:
            result['rmse_all'] = float(rmse_all)
            result['rmse_selected'] = float(rmse_selected)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
