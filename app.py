"""
AutoValue — Flask Backend
Uses random_forest_model.onnx for real predictions.
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
import onnxruntime as rt
import numpy as np
import os

app = Flask(__name__)

# ── Load ONNX model ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_model.onnx")
sess = rt.InferenceSession(MODEL_PATH)
print(f"ONNX model loaded: {MODEL_PATH}")

# ── Feature layout (152 total) ───────────────────────────────
# [0]  Present_Price
# [1]  Kms_Driven
# [2]  CarAge
# [3]  Fuel_Type_CNG
# [4]  Fuel_Type_Diesel
# [5]  Fuel_Type_Petrol
# [6]  Seller_Type_Dealer
# [7]  Seller_Type_Individual
# [8]  Transmission_Automatic
# [9]  Transmission_Manual
# [10] Owner_0 … [13] Owner_3
# [14..151] Car_Name OHE (138 cars, alphabetical)

CAR_NAMES = [
    "Ambassador Classic 1500 DSL BSIII",
    "Audi A4 1.8 TFSI",
    "Audi A4 2.0 TDI 177bhp",
    "Audi A4 2.0 TDI",
    "Audi A6 2.0 TDI",
    "Audi Q3 2.0 TDI quattro",
    "Audi Q5 2.0 TDI",
    "Audi Q7 3.0 TDI",
    "BMW 3 Series 320d Luxury Line",
    "BMW 3 Series 320d Sedan",
    "BMW 5 Series 520d",
    "BMW 7 Series 730Ld",
    "BMW X1 sDrive20d",
    "BMW X5 xDrive30d",
    "Chevrolet Beat Diesel",
    "Chevrolet Beat LT",
    "Chevrolet Cruze LTZ",
    "Chevrolet Enjoy 1.3 LS 7 STR",
    "Chevrolet Optra 1.6",
    "Chevrolet Sail 1.2 Base",
    "Chevrolet Sail UVA 1.2 LS",
    "Chevrolet Spark 1.0",
    "Chevrolet Tavera Neo 3 LS-10 STR BS-III",
    "Datsun GO Plus T",
    "Datsun GO T",
    "Ford EcoSport 1.0 Ecoboost Titanium BSIV",
    "Ford EcoSport 1.5 TDCi Titanium",
    "Ford EcoSport 1.5 Ti-VCT Titanium",
    "Ford Endeavour 2.2 Trend MT 4X2",
    "Ford Figo 1.2 Duratec Titanium",
    "Ford Figo 1.4 Duratorq EXI",
    "Ford Figo Aspire 1.5 TDCi Titanium",
    "Ford Fusion 1.4 TDCi Diesel",
    "Ford Ikon 1.3 Flair",
    "Honda Amaze 1.2 SMT i VTEC",
    "Honda Amaze 1.5 IDTEC V",
    "Honda Brio S(O)MT",
    "Honda City 1.5 EXi",
    "Honda City 1.5 V MT",
    "Honda City 2014-2015 1.5 V MT",
    "Honda City i-DTEC V",
    "Honda CR-V 2.0 AT 2WD",
    "Honda Jazz 1.5 V i DTEC",
    "Honda Jazz 1.2 V i VTEC",
    "Hyundai Accent GLE",
    "Hyundai Creta 1.4 CRDi S",
    "Hyundai Creta 1.6 CRDi SX",
    "Hyundai Creta 1.6 VTVT S",
    "Hyundai EON D-Lite Plus",
    "Hyundai EON Era Plus",
    "Hyundai Elite i20 1.2 Asta (O)",
    "Hyundai Elite i20 1.4 CRDI Asta",
    "Hyundai Grand i10 1.2 CRDi Asta",
    "Hyundai Grand i10 1.2 Kappa Magna",
    "Hyundai Grand i10 1.2 Kappa Sportz",
    "Hyundai i10 Era",
    "Hyundai i10 Magna",
    "Hyundai i10 Sportz",
    "Hyundai i20 1.2 Asta",
    "Hyundai i20 1.4 CRDI Asta",
    "Hyundai i20 Asta 1.4 CRDi 6 Speed",
    "Hyundai Santa Fe 4X4 AT",
    "Hyundai Sonata Transform 2.4 GDi",
    "Hyundai Tucson 2WD AT GL",
    "Hyundai Verna 1.6 CRDi SX",
    "Hyundai Verna 1.6 VTVT S",
    "Hyundai Xcent 1.2 Kappa S",
    "Jaguar XF 2.2 Diesel",
    "Jeep Compass 1.4 Multiair Limited",
    "Land Rover Freelander 2 SE",
    "Mahindra Bolero Power Plus SLX",
    "Mahindra KUV 100 K2",
    "Mahindra Scorpio VLX 2WD Airbag BSIII",
    "Mahindra Scorpio VLX 2WD Airbag BSIV",
    "Mahindra TUV 300 T8",
    "Mahindra XUV500 W10 FWD",
    "Mahindra XUV500 W8 2WD",
    "Maruti 800 AC",
    "Maruti Alto 800 LXI",
    "Maruti Alto LXi",
    "Maruti Baleno Alpha 1.2",
    "Maruti Baleno Delta 1.2",
    "Maruti Celerio VXI",
    "Maruti Ciaz ZXI Plus",
    "Maruti Dzire LXI",
    "Maruti Dzire VXI",
    "Maruti Ertiga VDI",
    "Maruti Ertiga VXI",
    "Maruti Esteem LXi",
    "Maruti Ignis Delta 1.2",
    "Maruti Omni E STD BS-IV",
    "Maruti Ritz VDI BS-IV",
    "Maruti Ritz Vxi",
    "Maruti S-Cross Alpha 1.3",
    "Maruti S-Cross Zeta 1.3",
    "Maruti S-Presso VXI Plus",
    "Maruti Swift Dzire LDI",
    "Maruti Swift Dzire LXI",
    "Maruti Swift Dzire VDI",
    "Maruti Swift Dzire VXI",
    "Maruti Swift LDI",
    "Maruti Swift LXI",
    "Maruti Swift VDI",
    "Maruti Swift VXI",
    "Maruti SX4 Green LXi CNG",
    "Maruti Wagon R LXI",
    "Maruti Wagon R VXI",
    "Maruti Zen Estilo LXI BS-IV",
    "Mercedes-Benz C-Class 220 CDI Avantgarde",
    "Mercedes-Benz CLA 200",
    "Mercedes-Benz E-Class E 200 CGI Avantgarde",
    "Mercedes-Benz GL-Class GL 350 CDI",
    "Mercedes-Benz GLA Class 200 CDI Style",
    "Mercedes-Benz M-Class ML 250 CDI",
    "Mitsubishi Pajero Sport 2.5 MT",
    "Nissan Micra XL",
    "Nissan Sunny XV",
    "Nissan Terrano XL D Plus",
    "Renault Duster 110 PS RXL",
    "Renault KWID 1.0 RXT",
    "Renault KWID RXT",
    "Skoda Octavia Elegance 2.0 TDI AT",
    "Skoda Rapid 1.5 TDI Ambition",
    "Skoda Rapid 1.5 TDI Elegance",
    "Tata Bolt XE Revotron",
    "Tata Nano Genx XT",
    "Tata Nexon XM",
    "Tata Safari Storme EX",
    "Tata Tiago XZ",
    "Tata Tigor XZ",
    "Tata Zest XE 75PS Diesel",
    "Toyota Corolla Altis 1.8 VL AT",
    "Toyota Corolla Altis G",
    "Toyota Etios GD",
    "Toyota Etios Liva GD",
    "Toyota Fortuner 2.8 2WD AT",
    "Toyota Innova 2.0 VX 7 STR BS-IV",
    "Toyota Innova 2.5 VX 7 STR BS3",
    "Volkswagen Cross Polo 1.2 MPI",
    "Volkswagen Polo Comfortline 1.2L",
    "Volkswagen Vento Comfortline Diesel",
    "Volkswagen Vento Diesel Highline",
]

# Encoding maps
FUEL_MAP   = {"CNG": 3, "Diesel": 4, "Petrol": 5}
SELLER_MAP = {"Dealer": 6, "Individual": 7}
TRANS_MAP  = {"Automatic": 8, "Manual": 9}
OWNER_MAP  = {0: 10, 1: 11, 2: 12, 3: 13}
CAR_NAME_OFFSET = 14   # car names start at index 14


def build_input(present_price, kms_driven, year, fuel_type,
                seller_type, transmission, owner, car_name):
    vec = np.zeros((1, 152), dtype=np.float32)
    car_age = 2025 - int(year)

    vec[0, 0] = float(present_price)
    vec[0, 1] = float(kms_driven)
    vec[0, 2] = float(car_age)

    if fuel_type in FUEL_MAP:
        vec[0, FUEL_MAP[fuel_type]] = 1.0
    if seller_type in SELLER_MAP:
        vec[0, SELLER_MAP[seller_type]] = 1.0
    if transmission in TRANS_MAP:
        vec[0, TRANS_MAP[transmission]] = 1.0

    owner_int = int(owner)
    if owner_int in OWNER_MAP:
        vec[0, OWNER_MAP[owner_int]] = 1.0

    if car_name in CAR_NAMES:
        idx = CAR_NAMES.index(car_name)
        vec[0, CAR_NAME_OFFSET + idx] = 1.0

    return vec, car_age


@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as f:
        return f.read()


@app.route("/predict", methods=["POST"])
def predict():
    d = request.json
    try:
        vec, car_age = build_input(
            present_price = d["present_price"],
            kms_driven    = d["kms_driven"],
            year          = d["year"],
            fuel_type     = d["fuel_type"],
            seller_type   = d["seller_type"],
            transmission  = d["transmission"],
            owner         = d["owner"],
            car_name      = d["car_name"],
        )
        raw = sess.run(["variable"], {"float_input": vec})[0][0][0]
        price_lakhs = round(max(0.1, float(raw) / 100000), 2)

        present_lakhs = float(d["present_price"])
        retention = round((price_lakhs / present_lakhs) * 100, 1) if present_lakhs else 0
        depreciation = round(100 - retention, 1)

        return jsonify({
            "predicted_price":    price_lakhs,
            "car_age":            car_age,
            "retention_pct":      retention,
            "depreciation_pct":   depreciation,
            "range_low":          round(price_lakhs * 0.93, 2),
            "range_high":         round(price_lakhs * 1.07, 2),
            "car_name":           d["car_name"],
            "present_price_display": float(d["present_price"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/cars")
def cars():
    return jsonify(CAR_NAMES)


if __name__ == "__main__":
    print("AutoValue running -> http://localhost:5000")
    app.run(debug=True, port=5000)
