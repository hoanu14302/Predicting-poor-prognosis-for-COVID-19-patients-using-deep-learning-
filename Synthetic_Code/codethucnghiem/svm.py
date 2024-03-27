import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu từ CSV
data = pd.read_csv('femaleonly.csv')

# Chuyển cột "sex" sang biến nhị phân
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Xác định biến độc lập (features) và biến phụ thuộc (label)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Sử dụng StratifiedKFold với k=10
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Khởi tạo mô hình Support Vector Machine với tham số kiểm soát (C) giảm
model = SVC(C=6, kernel='rbf', gamma='scale', random_state=42)

# Lists để lưu trữ các kết quả đánh giá từ mỗi fold
accuracy_list = []
precision_list = []
recall_list = []

# Lặp qua các fold
for train_index, test_index in k_fold.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='positive')
    recall = recall_score(y_test, y_pred, pos_label='positive')

    # Lưu kết quả vào danh sách
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

# In kết quả trung bình từ tất cả các fold
print(f'Average Accuracy: {sum(accuracy_list) / len(accuracy_list)}')
print(f'Average Precision: {sum(precision_list) / len(precision_list)}')
print(f'Average Recall: {sum(recall_list) / len(recall_list)}')
