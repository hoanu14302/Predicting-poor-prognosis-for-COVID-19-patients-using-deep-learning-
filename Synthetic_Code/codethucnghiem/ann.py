import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Đọc dữ liệu từ CSV
data = pd.read_csv('femaleonly.csv')

# Chuyển cột "sex" sang biến nhị phân
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Chuyển cột nhãn "ketqua" sang kiểu số
data['ketqua'] = pd.to_numeric(data['ketqua'], errors='coerce')

# Loại bỏ các dòng có giá trị NaN trong cột nhãn
data = data.dropna(subset=['ketqua'])

# Chuyển đổi kiểu dữ liệu của cột nhãn thành số nguyên (nếu cần)
data['ketqua'] = data['ketqua'].astype(int)

# Xác định biến độc lập (features) và biến phụ thuộc (label)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Sử dụng StratifiedKFold với k=10
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists để lưu trữ các kết quả đánh giá từ mỗi fold
accuracy_list = []
precision_list = []
recall_list = []

# Lặp qua các fold
for train_index, test_index in k_fold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Khởi tạo mô hình ANN
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sửa activation thành 'sigmoid' cho bài toán phân loại
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Sửa loss thành 'binary_crossentropy'

    # Huấn luyện mô hình
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Dự đoán trên tập kiểm thử
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    # Lưu kết quả vào danh sách
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)

# In kết quả trung bình từ tất cả các fold
print(f'Average Accuracy: {sum(accuracy_list) / len(accuracy_list)}')
print(f'Average Precision: {sum(precision_list) / len(precision_list)}')
print(f'Average Recall: {sum(recall_list) / len(recall_list)}')
