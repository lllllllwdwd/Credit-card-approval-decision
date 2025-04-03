import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

# 文件路径（请根据实际路径修改）
file_path = r"C:\Users\月亮姐姐\Downloads\bankloan.csv"  # 请替换为你的 CSV 文件路径

# 加载CSV数据
data = pd.read_csv(file_path)

# 查看数据
print("数据预览:")
print(data.head())

# 特征和目标变量分离
X = data.drop(columns=["ID", "Personal.Loan"])  # 特征
y = data["Personal.Loan"]  # 目标变量

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=100)

# ----------------------------------------------
# 使用支持向量机(SVM)模型
svm_model = SVC(random_state=42, kernel='linear', probability=True)  # 设置 kernel='linear' 来获取 coef_
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]  # 预测概率，用于AUC曲线

# 评估模型
print("\nSVM - Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM - Classification Report:\n", classification_report(y_test, y_pred_svm))
print("SVM - Cohen Kappa:", cohen_kappa_score(y_test, y_pred_svm))

# 计算AUC
auc = roc_auc_score(y_test, y_prob_svm)
print("SVM - AUC:", auc)

# 绘制AUC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curve - SVM')
plt.legend(loc='lower right')
plt.savefig('output/SVM_AUC.png')

# ----------------------------------------------
# 特征重要性分析：SVM 使用线性核时可以通过 coef_ 获取特征重要性
importances_svm = abs(svm_model.coef_.flatten())
features = X.columns

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(features, importances_svm)
plt.xlabel('Importance')
plt.title('Feature Importance - SVM')
plt.savefig('output/SVM_Feature_Importance.png')
'''
SVM - Accuracy: 0.945
SVM - Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.99      0.97       891
           1       0.90      0.56      0.69       109

    accuracy                           0.94      1000
   macro avg       0.92      0.78      0.83      1000
weighted avg       0.94      0.94      0.94      1000

SVM - Cohen Kappa: 0.66086227308603
SVM - AUC: 0.9603115765195276
'''