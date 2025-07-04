import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, confusion_matrix
)

# === Load và tiền xử lý dữ liệu ===
def load_and_preprocess(path):
    df = pd.read_csv(path, encoding='utf-8-sig')

    # Chuyển đổi ngày tháng
    for col in ['StartDate', 'EndDate', 'CheckoutDate', 'TempResidenceExpiryDate']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Cột đích: HasCheckout
    df['HasCheckout'] = df['CheckoutDate'].notnull().astype(int)

    # Số ngày thực ở lại
    df['ActualStayDays'] = (df['CheckoutDate'] - df['StartDate']).dt.days
    df['ActualStayDays'] = df['ActualStayDays'].fillna((pd.Timestamp.now() - df['StartDate']).dt.days)
    df['ActualStayDays'] = df['ActualStayDays'].clip(lower=0)

    # Số ngày trong hợp đồng
    df['TotalContractDays'] = (df['EndDate'] - df['StartDate']).dt.days.clip(lower=1)

    # Tỷ lệ thực ở lại so với hợp đồng
    df['StayRatio'] = df['ActualStayDays'] / df['TotalContractDays']

    # Có phương tiện hay không
    df['VehicleOwned'] = df['VehiclePlateType'].notnull().astype(int)

    # Xử lý tuổi
    df['Age'] = df['Age'].fillna(df['Age'].median()).clip(lower=0)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 19, 25, 100], labels=[0, 1, 2], include_lowest=True).astype(int)

    # Số lượng tiện nghi trong phòng
    df['AmenityCount'] = df['Amenities'].fillna('').apply(lambda x: len(str(x).split(',')))

    # Các cột phân loại cần encode
    categorical_cols = ['Gender', 'AppRegistered', 'TemporaryResidenceRegistered',
                        'VehiclePlateType', 'Status', 'RoomGender', 'RoomType']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Các đặc trưng sử dụng để huấn luyện 
    features = [
        'Age', 'VehicleOwned', 'AmenityCount',
        'ActualStayDays', 'TotalContractDays', 'StayRatio',
        'Gender', 'AppRegistered', 'TemporaryResidenceRegistered',
        'VehiclePlateType', 'Status', 'RoomGender', 'RoomType', 'AgeGroup'
    ]

    return df, features, label_encoders

# === Train & đánh giá mô hình ===
def train_and_save(data_path, model_dir):
    df, features, label_encoders = load_and_preprocess(data_path)

    X = df[features]
    y = df['HasCheckout']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
    print("🔍 Classification Report:\n", classification_report(y_test, y_pred))
    print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("🎯 ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'model_churn.pkl'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    joblib.dump(features, os.path.join(model_dir, 'feature_list.pkl'))

    print(f"✅ Model & encoders saved to: {model_dir}")

    # Biểu đồ hệ số ảnh hưởng
    importance = pd.Series(np.abs(model.coef_[0]), index=features)
    importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6), title='Feature Importance')
    plt.xlabel('Hệ số (|coef|)')
    plt.tight_layout()
    plt.show()

# === Chạy script ===
if __name__ == "__main__":
    # Đường dẫn thư mục hiện tại chứa script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Đường dẫn tương đối đến file dữ liệu
    data_path = os.path.join(base_dir, "data", "customer_behavior.csv")

    # Đường dẫn thư mục lưu model
    model_dir = os.path.join(base_dir, "models")

    train_and_save(data_path, model_dir)
