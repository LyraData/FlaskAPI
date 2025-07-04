from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==== Cấu hình đường dẫn ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# models nằm trong thư mục con của BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CUSTOMER_DATA_PATH = os.path.join(BASE_DIR, 'data', 'customer_behavior.csv')
ROOM_DATA_PATH = os.path.join(BASE_DIR, 'data', 'room_inventory.csv')
AVAILABLE_DISTRICTS = ["Quận 10", "Quận 9", "Quận 3", "Gò Vấp"]

# ==== Load dữ liệu và model ====
def load_resources():
    model = joblib.load(os.path.join(MODEL_DIR, 'model_churn.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    features = joblib.load(os.path.join(MODEL_DIR, 'feature_list.pkl'))

    df_customers = pd.read_csv(CUSTOMER_DATA_PATH, encoding='utf-8-sig')
    df_customers['StartDate'] = pd.to_datetime(df_customers['StartDate'], errors='coerce')

    df_rooms = pd.read_csv(ROOM_DATA_PATH, encoding='utf-8-sig')
    df_rooms.dropna(subset=['RoomID', 'RoomType', 'RoomGender', 'Price', 'Address'], inplace=True)
    df_rooms['Price'] = df_rooms['Price'].astype(int)

    return model, encoders, features, df_customers, df_rooms

model, label_encoders, features, df_all, df_rooms = load_resources()

# ==== Gợi ý phòng ====
def suggest_room_for_new_customer(df_rooms, gender, max_price, preferred_districts=None, top_n=3):
    gender_filter = df_rooms['RoomGender'].isin([gender, 'Nam/Nữ'])
    price_filter = df_rooms['Price'] <= max_price

    if preferred_districts:
        def match_district(address):
            address_lower = address.lower()
            return any(district.lower() in address_lower for district in preferred_districts)
        district_filter = df_rooms['Address'].apply(match_district)
    else:
        district_filter = True

    filtered = df_rooms[gender_filter & price_filter & district_filter].drop_duplicates(subset='RoomID')

    return filtered.sort_values(by='Price').head(top_n).to_dict(orient='records') if not filtered.empty else []

# ==== Trang chủ ====
@app.route('/')
def home():
    return render_template('home.html')

# ==== Gợi ý phòng ====
@app.route('/suggest_room_form', methods=['GET', 'POST'])
def suggest_room_form():
    if request.method == 'POST':
        gender = request.form.get('Gender', 'Nam')
        preferred_districts = request.form.getlist('PreferredDistricts')

        try:
            max_price = int(request.form.get('MaxPrice', 9999999))
        except ValueError:
            return render_template(
                'suggest_room.html',
                rooms=[],
                reason_room="❌ Lỗi: Giá thuê tối đa phải là số nguyên.",
                reason_district="Không xác định",
                districts=AVAILABLE_DISTRICTS,
                gender=gender,
                max_price=request.form.get('MaxPrice'),
                preferred_districts=preferred_districts
            )

        suggestions = suggest_room_for_new_customer(df_rooms, gender, max_price, preferred_districts)

        if suggestions:
            return render_template(
                'suggest_room.html',
                rooms=suggestions,
                reason_room="🔍 Gợi ý dựa trên giới tính, giá thuê và quận ưu tiên.",
                reason_district=", ".join(preferred_districts) if preferred_districts else "Không có quận ưu tiên",
                districts=AVAILABLE_DISTRICTS,
                gender=gender,
                max_price=max_price,
                preferred_districts=preferred_districts
            )
        else:
            return render_template(
                'suggest_room.html',
                rooms=[],
                reason_room="❌ Không tìm thấy phòng phù hợp.",
                reason_district="Vui lòng thử mức giá cao hơn hoặc chọn quận khác.",
                districts=AVAILABLE_DISTRICTS,
                gender=gender,
                max_price=max_price,
                preferred_districts=preferred_districts
            )

    return render_template(
        'suggest_room.html',
        districts=AVAILABLE_DISTRICTS,
        gender='Nam',
        max_price='',
        preferred_districts=[]
    )

# ==== Dự đoán churn ====
@app.route('/predict_churn_form', methods=['GET', 'POST'])
def predict_churn_by_id():
    if request.method == 'POST':
        customer_id = request.form.get('CustomerID', '').strip()
        if not customer_id:
            return "❌ Vui lòng nhập mã khách hàng.", 400

        df_customer = df_all[df_all['CustomerID'].astype(str) == customer_id]
        if df_customer.empty:
            return f"❌ Không tìm thấy khách hàng với mã: {customer_id}", 404

        row = df_customer.sort_values('StartDate', ascending=False).iloc[0].copy()

        row['HasCheckout'] = int(pd.notnull(row['CheckoutDate']))
        row['ActualStayDays'] = (
            (pd.to_datetime(row['CheckoutDate']) - pd.to_datetime(row['StartDate'])).days
            if pd.notnull(row['CheckoutDate']) else
            (pd.Timestamp.now() - pd.to_datetime(row['StartDate'])).days
        )
        row['ContractDays'] = max((pd.to_datetime(row['EndDate']) - pd.to_datetime(row['StartDate'])).days, 1)
        row['StayRatio'] = row['ActualStayDays'] / row['ContractDays']
        row['VehicleOwned'] = int(str(row['VehiclePlateType']).strip() != 'Chưa cập nhật')
        row['Age'] = max(row['Age'], 0)
        row['AmenityCount'] = len(str(row['Amenities']).split(','))
        row['AgeGroup'] = int(pd.cut([row['Age']], bins=[0, 19, 25, 100], labels=[0, 1, 2])[0])

        for col in features:
            if col not in row:
                row[col] = 0 if pd.api.types.is_numeric_dtype(df_all[col]) else "Unknown"

        for col, le in label_encoders.items():
            val = str(row.get(col, "Unknown"))
            if val not in le.classes_:
                le.classes_ = np.append(le.classes_, val)
            row[col] = le.transform([val])[0]

        df_input = pd.DataFrame([row[features]])
        prob = model.predict_proba(df_input)[0][1]
        prediction = int(prob > 0.5)

        message = (
            "😟 Khách có khả năng **rời đi**. Gợi ý: giữ phòng cũ, tặng ưu đãi, liên hệ hỏi thăm."
            if prediction == 1 else
            "😊 Khách **có khả năng ở lại**."
        )

        return render_template(
            'predict_churn.html',
            customer_id=customer_id,
            prediction=prediction,
            probability=round(prob, 4),
            message=message
        )

    return render_template('predict_churn.html')

if __name__ == '__main__':
    app.run(debug=True)
