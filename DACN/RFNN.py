import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Button
import numpy as np
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tkinter import filedialog, Tk

data = None
scaled_data = None
future_dates = []
future_close = []
future_volume = []
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('GOLD PRICE PREDICTION CHART', fontsize=30, fontweight='bold', x=0.5)


def process_data(file_path):
    global data, scaled_data
    if file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("File không được hỗ trợ. Vui lòng chọn file Excel.")

    if 'Day' in data.columns and 'Month' in data.columns and 'Year' in data.columns:
        data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
    else:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    if data['Date'].isnull().any():
        raise ValueError("Cột Date chứa giá trị không hợp lệ hoặc bị trống.")

    data = data[data['Date'] >= '2024-01-01']
    data = data.sort_values('Date')

    volume_column = None
    for col in data.columns:
        if "Vol" in col or "Volume" in col:
            volume_column = col
            break

    if volume_column is None:
        raise KeyError("Không tìm thấy cột Volume hoặc Vol. trong dữ liệu.")

    data.rename(columns={volume_column: 'Volume'}, inplace=True)
    data['Volume'] = data['Volume'].astype(str).str.replace('K', '').astype(float) * 1000

    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)

    if data.empty:
        raise ValueError("Dữ liệu sau khi lọc rỗng. Vui lòng kiểm tra file Excel.")

    data.set_index('Date', inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])


def calculate_rmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)


def predict_volume_using_rf(data, future_dates=None):
    if data is None or data.empty:
        print("Dữ liệu không hợp lệ hoặc trống!")
        return None

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Dữ liệu thiếu các cột: {', '.join(missing_columns)}")
        return None

    features = data[['Open', 'High', 'Low', 'Close']].values
    target = data['Volume'].values

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(features, target)

    if future_dates is None or len(future_dates) == 0:
        predicted_volume = rf_model.predict(features[-1:])
    else:
        predicted_volume = rf_model.predict(features[-len(future_dates):])

    return predicted_volume


def gaussian_rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s ** 2))


def predict_rfnn(days_to_predict):
    global data, scaled_data

    X = scaled_data
    y_close = data['Close'].values
    y_volume = data['Volume'].values

    centroids = X[:5]  
    spreads = np.std(X, axis=0)  

    rbf_outputs = np.array([[gaussian_rbf(x, c, s) for c, s in zip(centroids, spreads)] for x in X])

    weights_close = np.random.uniform(size=rbf_outputs.shape[1])
    weights_volume = np.random.uniform(size=rbf_outputs.shape[1])

    fuzzy_close = rbf_outputs @ weights_close  
    fuzzy_volume = rbf_outputs @ weights_volume

    predictions_close = []
    predictions_volume = []

    base_close = y_close[-1]
    base_volume = y_volume[-1]

    for day in range(days_to_predict):
        close_variation = fuzzy_close[-1] + np.random.normal(0, base_close * 0.01)
        volume_variation = fuzzy_volume[-1] + np.random.normal(0, base_volume * 0.05)

        predicted_close = base_close + close_variation
        predicted_volume = base_volume + volume_variation

        predictions_close.append(predicted_close)
        predictions_volume.append(predicted_volume)

        base_close = predicted_close
        base_volume = predicted_volume

    return predictions_close, predictions_volume


tooltip = None


def update_tooltip(event):
    global tooltip  

    if tooltip:
        tooltip.remove()
        tooltip = None

    if event.inaxes == ax1 and event.xdata and event.ydata:
        if event.xdata >= mdates.date2num(data.index[-1]):
            closest_index = min(range(len(future_dates)),
                                key=lambda i: abs(mdates.date2num(future_dates[i]) - event.xdata))
            future_date = future_dates[closest_index]

            tooltip_text = (
                f"Date: {future_date.strftime('%Y-%m-%d')}\n"
                f"Open: {future_close[closest_index] * 0.99:.2f}\n"
                f"High: {future_close[closest_index]:.2f}\n"
                f"Low: {future_close[closest_index] * 0.98:.2f}\n"
                f"Close: {future_close[closest_index]:.2f}\n"
                f"Volume: {int(future_volume[closest_index]):,}"
            )
        else: 
            closest_index = min(range(len(data.index)), key=lambda i: abs(mdates.date2num(data.index[i]) - event.xdata))
            row = data.iloc[closest_index]
            tooltip_text = (
                f"Date: {row.name.strftime('%Y-%m-%d')}\n"
                f"Open: {row['Open']:.2f}\n"
                f"High: {row['High']:.2f}\n"
                f"Low: {row['Low']:.2f}\n"
                f"Close: {row['Close']:.2f}\n"
                f"Volume: {int(row['Volume']):,}"
            )
        tooltip = ax1.annotate(
            tooltip_text,
            (event.xdata, event.ydata),
            xytext=(15, 15), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="lightyellow"),
            fontsize=10, color="black"
        )
        plt.draw()

    if event.inaxes == ax2:  
        if event.xdata and event.ydata:
            closest_index = min(range(len(data.index)), key=lambda i: abs(mdates.date2num(data.index[i]) - event.xdata))
            row = data.iloc[closest_index]
            actual_volume = row['Volume']

            if event.xdata >= mdates.date2num(data.index[-1]):
                future_index = min(range(len(future_dates)),
                                   key=lambda i: abs(mdates.date2num(future_dates[i]) - event.xdata))
                predicted_volume = future_volume[future_index]

                rmse = calculate_rmse([actual_volume], [predicted_volume])
                tooltip_text = f"Date: {future_dates[future_index].strftime('%Y-%m-%d')}\n" \
                               f"Volume: {predicted_volume:.0f}\n" \
                               f"RMSE : {rmse:.2f}"
            else:
                tooltip_text = f"Date: {row.name.strftime('%Y-%m-%d')}\n" \
                               f"Actual Volume: {int(actual_volume):,}"

            tooltip = ax2.annotate(
                tooltip_text,
                (event.xdata, event.ydata),
                xytext=(15, 15), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="lightyellow"),
                fontsize=10, color="black"
            )
            plt.draw()


future_volume = predict_volume_using_rf(data)

fig.canvas.mpl_connect('motion_notify_event', update_tooltip)


def show_current_data():
    global future_dates, future_close, future_volume

    future_dates = []
    future_close = []
    future_volume = []

    ax1.clear()
    ax2.clear()

    ax1.plot(data.index, data['Close'], label='Price', color='blue', linewidth=1)
    ax1.fill_between(data.index, data['Close'], color='lightblue', alpha=0.5)

    ax1.set_ylabel('PRICE', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=12)

    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.6, width=1.0)

    ax2.set_ylabel('VOLUME', fontsize=12)
    ax2.set_xlabel('TIME', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.draw()

def update_plot(interval):
    global future_dates, future_close, future_volume

    if data is None:
        print("Chưa có dữ liệu. Vui lòng chọn file trước!")
        return

    days_map = {'day': 1, 'week': 7, 'month': 30, 'year': 365}
    days_to_predict = days_map[interval]

    future_close, future_volume = predict_rfnn(days_to_predict)
    base_date = data.index[-1]
    future_dates = [base_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]

    ax1.clear()
    ax2.clear()

    if interval == 'day':
        ax1.scatter(future_dates[0], future_close[0], color='orange', s=5)
    ax1.plot(data.index, data['Close'], label='Price', color='blue', linewidth=1)
    ax1.fill_between(data.index, data['Close'], color='lightblue', alpha=0.5)
    ax1.plot(future_dates, future_close, color='#FFC107', linewidth=2, label='Predict')
    ax1.fill_between(future_dates, future_close, color='#FFC107', alpha=0.5)

    ax1.set_ylabel('PRICE', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=12)

    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.6, width=1.0)

    predicted_colors = ['purple' if i > 0 and future_volume[i] < future_volume[i - 1] else 'orange' for i in
                        range(len(future_volume))]
    ax2.bar(
        future_dates,
        future_volume,
        color=predicted_colors,
        alpha=0.6,
        width=1.0,
        label='Volume Prediction'
    )

    ax2.set_ylabel('VOLUME', fontsize=12)
    ax2.set_xlabel('TIME', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.draw()


ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

def select_file(event=None):
    global current_rmse_annotation
    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        try:

            process_data(file_path)
            show_current_data()
        except Exception as e:
            print(f"Lỗi khi xử lý file: {e}")

ax_button_browse = plt.axes([0.05, 0.01, 0.1, 0.04])
button_browse = Button(ax_button_browse, 'BROWSE', color='lightyellow', hovercolor='skyblue')
button_browse.on_clicked(select_file)

ax_button_day = plt.axes([0.2, 0.01, 0.1, 0.04])
button_day = Button(ax_button_day, '1 DAY', color='lightyellow', hovercolor='skyblue')
button_day.on_clicked(lambda x: update_plot('day'))

ax_button_week = plt.axes([0.35, 0.01, 0.1, 0.04])
button_week = Button(ax_button_week, '1 WEEK', color='lightyellow', hovercolor='skyblue')
button_week.on_clicked(lambda x: update_plot('week'))

ax_button_month = plt.axes([0.5, 0.01, 0.1, 0.04])
button_month = Button(ax_button_month, '1 MONTH', color='lightyellow', hovercolor='skyblue')
button_month.on_clicked(lambda x: update_plot('month'))

ax_button_year = plt.axes([0.65, 0.01, 0.1, 0.04])
button_year = Button(ax_button_year, '1 YEAR', color='lightyellow', hovercolor='skyblue')
button_year.on_clicked(lambda x: update_plot('year'))

ax_button_now = plt.axes([0.8, 0.01, 0.1, 0.04])
button_now = Button(ax_button_now, 'NOW', color='lightyellow', hovercolor='skyblue')
button_now.on_clicked(lambda x: show_current_data())

plt.show()
