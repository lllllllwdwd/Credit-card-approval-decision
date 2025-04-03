# ----------------------------------------------
# 使用随机森林模型
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)
y_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]  # 预测概率，用于AUC曲线

# 评估模型
print("\nRandom Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest - Cohen Kappa:", cohen_kappa_score(y_test, y_pred_rf))

# 计算AUC
auc_rf = roc_auc_score(y_test, y_prob_rf)
print("Random Forest - AUC:", auc_rf)

# 绘制AUC曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'AUC = {auc_rf:.4f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.savefig('output/Random_Forest_AUC.png')

# ----------------------------------------------
# 特征重要性分析：随机森林模型提供了 feature_importances_ 属性
importances_rf = random_forest_model.feature_importances_
features_rf = X.columns

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(features_rf, importances_rf)
plt.xlabel('Importance')
plt.title('Feature Importance - Logistic Regression')
plt.savefig('output/Random_Forest_Feature_Importance.png')
'''
Random Forest - Accuracy: 0.985
Random Forest - Classification Report:
               precision    recall  f1-score   support

           0       0.99      1.00      0.99       891
           1       0.96      0.90      0.93       109

    accuracy                           0.98      1000
   macro avg       0.97      0.95      0.96      1000
weighted avg       0.98      0.98      0.98      1000

Random Forest - Cohen Kappa: 0.9205356953656417
Random Forest - AUC: 0.9976523646248416
'''