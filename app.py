import os
import pandas as pd
from flask import Flask, render_template, request, session, send_file
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error
import numpy as np
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline  # For spline interpolation

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')  # Needed for session management

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Utility functions
def save_uploaded_file(file, filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return path

def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    if 'month' not in df.columns:
        raise ValueError(f"Error: 'month' column not found in {file_path}.")
    df['month'] = pd.to_datetime(df['month'], format='%b-%y')
    return df

def calculate_performance_metric(y_true, y_pred, metric):
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred),
        'max_error': max_error(y_true, y_pred)
    }
    return metrics.get(metric, "Error: Unsupported performance metric selected.")

def calculate_saturated_dissolved_oxygen(temp):
    temp_K = temp + 273.15  # Convert Celsius to Kelvin
    ln_Ox_sat = ( -139.3441 +
        (1.575701e5 / temp_K) -
        (6.642308e7 / (temp_K ** 2)) +
        (1.243800e10 / (temp_K ** 3)) -
        (8.621949e11 / (temp_K ** 4))
    )
    return np.exp(ln_Ox_sat)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        air_temp_file = request.files['air_temp']
        observed_temp_file = request.files['observed_temp']
        ml_model = request.form['ml_model']
        performance_metric = request.form['performance_metric']

        air_temp_path = save_uploaded_file(air_temp_file, 'air_temp.csv')
        observed_temp_path = save_uploaded_file(observed_temp_file, 'observed_temp.csv')

        air_temp_df = load_and_process_csv(air_temp_path)
        observed_temp_df = load_and_process_csv(observed_temp_path)

        if 'avg_air_temp' not in air_temp_df.columns:
            return "Error: 'avg_air_temp' column not found in the uploaded air temperature file."
        if 'avg_water_temp' not in observed_temp_df.columns:
            return "Error: 'avg_water_temp' column not found in the uploaded water temperature file."

        air_temperature = air_temp_df['avg_air_temp'].values.flatten()
        observed_temperature = observed_temp_df['avg_water_temp'].values.flatten()

        X = air_temperature.reshape(-1, 1)
        y = observed_temperature

        model_mapping = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(),
            'decision_tree': DecisionTreeRegressor(),
            'svm': SVR()
        }
        model = model_mapping.get(ml_model, None)
        if model is None:
            return "Error: Unsupported ML model selected."

        model.fit(X, y)
        predictions = model.predict(X)

        performance = calculate_performance_metric(y, predictions, performance_metric)
        if isinstance(performance, str):  # If an error message is returned
            return performance

        result_df = pd.DataFrame({
            'Month': air_temp_df['month'],
            'Air Temperature': air_temperature,
            'Observed Water Temperature': observed_temperature,
            'Predicted Water Temperature': predictions
        })

        result_path = os.path.join(OUTPUT_FOLDER, 'prediction_results.csv')
        result_df.to_csv(result_path, index=False)

        session['predicted_water_temp'] = predictions.tolist()
        session['result_path'] = result_path

        return render_template('result.html', 
                               predictions=predictions.tolist(), 
                               performance=performance, 
                               performance_metric=performance_metric,
                               enumerate=enumerate)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return str(e)

@app.route('/generate_graphs')
def generate_graphs():
    try:
        predictions = session.get('predicted_water_temp')
        if not predictions:
            return "Error: No prediction data available."

        air_temp_file_path = os.path.join(UPLOAD_FOLDER, 'air_temp.csv')
        observed_temp_file_path = os.path.join(UPLOAD_FOLDER, 'observed_temp.csv')

        if not os.path.exists(air_temp_file_path) or not os.path.exists(observed_temp_file_path):
            return "Error: Required files for generating graphs are missing."

        air_temp_df = load_and_process_csv(air_temp_file_path)
        observed_temp_df = load_and_process_csv(observed_temp_file_path)
        
        air_temperature = air_temp_df['avg_air_temp'].values.flatten()
        observed_temperature = observed_temp_df['avg_water_temp'].values.flatten()
        dates = air_temp_df['month'].values.flatten()

        predictions = np.array(predictions).flatten()

        # Ensure dates are sorted (strictly increasing sequence)
        sorted_indices = np.argsort(dates)
        dates = dates[sorted_indices]
        observed_temperature = observed_temperature[sorted_indices]
        predictions = predictions[sorted_indices]

        # Spline interpolation
        dates_num = mdates.date2num(dates)
        spline_dates = np.linspace(dates_num.min(), dates_num.max(), 500)
        spline_dates_dt = mdates.num2date(spline_dates)

        spline_observed_temp = make_interp_spline(dates_num, observed_temperature, k=3)(spline_dates)
        spline_predictions = make_interp_spline(dates_num, predictions, k=3)(spline_dates)

        # Time Series Plot
        plt.figure(figsize=(14, 7))
        plt.plot(spline_dates_dt, spline_observed_temp, label='Observed Water Temperature', color='blue')
        plt.plot(spline_dates_dt, spline_predictions, label='Predicted Water Temperature', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Water Temperature')
        plt.title('Time Series: Observed vs Predicted Water Temperature')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gcf().autofmt_xdate(rotation=45)
        
        time_series_path = os.path.join(OUTPUT_FOLDER, 'time_series_graph.jpg')
        plt.savefig(time_series_path, format='jpg')
        plt.close()

        # Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(observed_temperature, predictions, color='blue', edgecolor='k', alpha=0.7)
        
        # Adding diagonal line from (20, 20) to (25, 25)
        plt.plot([20, 25], [20, 25], color='red', linestyle='--', linewidth=2)

        plt.xlabel('Observed Water Temperature')
        plt.ylabel('Predicted Water Temperature')
        plt.title('Scatter Plot: Observed vs Predicted Water Temperature')
        
        # Dynamic axis limits
        min_temp = min(min(observed_temperature), min(predictions))
        max_temp = max(max(observed_temperature), max(predictions))
        buffer = (max_temp - min_temp) * 0.1  # Adding 10% buffer to each side
        plt.xlim(min_temp - buffer, max_temp + buffer)
        plt.ylim(min_temp - buffer, max_temp + buffer)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        scatter_path = os.path.join(OUTPUT_FOLDER, 'scatter_graph.jpg')
        plt.savefig(scatter_path, format='jpg')
        plt.close()

        session['time_series_graph'] = time_series_path
        session['scatter_graph'] = scatter_path

        with open(time_series_path, 'rb') as f:
            time_series_graph_url = base64.b64encode(f.read()).decode()
        with open(scatter_path, 'rb') as f:
            scatter_graph_url = base64.b64encode(f.read()).decode()

        return render_template('graphs.html', 
                               time_series_graph_url=time_series_graph_url, 
                               scatter_graph_url=scatter_graph_url)

    except Exception as e:
        logging.error(f"Error during graph generation: {e}")
        return str(e)
    
@app.route('/calculate_oxygen', methods=['POST'])
def calculate_oxygen():
    option = request.form['oxygen_option']

    try:
        if option == 'select_data':
            river_temp_file = request.files['river_temp']
            river_temp_path = save_uploaded_file(river_temp_file, 'river_temp.csv')
            river_temp_df = pd.read_csv(river_temp_path)
            if 'avg_water_temp' not in river_temp_df.columns:
                return "Error: 'avg_water_temp' column not found in the uploaded file."
            river_temp = river_temp_df['avg_water_temp'].values
        elif option == 'simulate_data':
            if 'predicted_water_temp' not in session:
                return "Error: No predicted data available for simulation."
            river_temp = np.array(session['predicted_water_temp'])
        else:
            return "Error: Unsupported option selected."

        river_temp = river_temp.flatten()
        logging.debug(f"River Temperature Array: {river_temp}")

        saturated_dissolved_oxygen = calculate_saturated_dissolved_oxygen(river_temp)
        logging.debug(f"Saturated Dissolved Oxygen: {saturated_dissolved_oxygen}")

        result_df = pd.DataFrame({'River Temperature': river_temp, 'Saturated Dissolved Oxygen': saturated_dissolved_oxygen})
        result_path = os.path.join(OUTPUT_FOLDER, 'oxygen_results.csv')
        result_df.to_csv(result_path, index=False)

        return render_template('result.html', 
                               predictions=saturated_dissolved_oxygen.tolist(), 
                               performance=None, 
                               performance_metric=None,
                               enumerate=enumerate)
    except Exception as e:
        logging.error(f"Error during oxygen calculation: {e}")
        return str(e)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

@app.route('/download_graph/<graph_type>')
def download_graph(graph_type):
    if graph_type not in ['time_series', 'scatter']:
        return "Error: Invalid graph type requested."
    
    graph_path = session.get(f'{graph_type}_graph')
    if not graph_path or not os.path.exists(graph_path):
        return "Error: Graph file not available or does not exist."

    return send_file(graph_path, mimetype='image/jpeg', as_attachment=True, download_name=f'{graph_type}.jpg')

if __name__ == "__main__":
    app.run(debug=True)