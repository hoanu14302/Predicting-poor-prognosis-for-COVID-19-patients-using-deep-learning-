import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings

# Đọc dữ liệu từ file csv
data = pd.read_csv('femaleonly.csv')

# Chuyển cột "sex" sang biến nhị phân
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])

# Xác định biến độc lập (features) và biến phụ thuộc (label)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Sử dụng StratifiedKFold để thực hiện k-fold cross-validation
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Chọn giá trị cho pos_label
if 'positive' in set(y):
    pos_label = 'positive'
elif 'negative' in set(y):
    pos_label = 'negative'
else:
    raise ValueError("Không tìm thấy nhãn 'positive' hoặc 'negative' trong dữ liệu.")

# Tiền xử lý: Scaling dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lists để lưu kết quả đánh giá từ mỗi fold
accuracies = []
precisions = []
recalls = []

# Tắt cảnh báo ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Lặp qua các fold
for train_index, test_index in k_fold.split(X_scaled, y):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Huấn luyện mô hình trên fold hiện tại
    model = LogisticRegression(max_iter=1000, solver='lbfgs')  
    model.fit(X_train_fold, y_train_fold)

    # Dự đoán trên tập kiểm thử của fold hiện tại
    y_pred_fold = model.predict(X_test_fold)

    # Đánh giá hiệu suất của mô hình trên fold hiện tại
    accuracy_fold = metrics.accuracy_score(y_test_fold, y_pred_fold)
    precision_fold = metrics.precision_score(y_test_fold, y_pred_fold, pos_label=pos_label)
    recall_fold = metrics.recall_score(y_test_fold, y_pred_fold, pos_label=pos_label)

    # Lưu kết quả vào danh sách
    accuracies.append(accuracy_fold)
    precisions.append(precision_fold)
    recalls.append(recall_fold)

# Bật cảnh báo trở lại
warnings.filterwarnings("default")

# In kết quả trung bình từ tất cả các fold
print(f'Average Accuracy: {sum(accuracies) / len(accuracies)}')
print(f'Average Precision: {sum(precisions) / len(precisions)}')
print(f'Average Recall: {sum(recalls) / len(recalls)}')
