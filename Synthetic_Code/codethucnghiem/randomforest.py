from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Đọc dữ liệu từ file csv
data = pd.read_csv('femaleonly.csv')

# Chuyển cột "sex" sang biến nhị phân
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Xác định biến độc lập (features) và biến phụ thuộc (label)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Sử dụng StratifiedKFold với k=10
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists để lưu trữ các kết quả đánh giá từ các fold khác nhau
accuracy_list = []
precision_list = []
recall_list = []

# Lặp qua các fold
for train_index, test_index in k_fold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Khởi tạo mô hình RandomForestClassifier
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Huấn luyện mô hình
    random_forest_model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử
    y_pred = random_forest_model.predict(X_test)

    # Đánh giá hiệu suất của mô hình và lưu vào list
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, pos_label='positive')
    recall = metrics.recall_score(y_test, y_pred, pos_label='positive')

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

# In kết quả trung bình của các fold
print(f'Average Accuracy: {sum(accuracy_list) / len(accuracy_list)}')
print(f'Average Precision: {sum(precision_list) / len(precision_list)}')
print(f'Average Recall: {sum(recall_list) / len(recall_list)}')
