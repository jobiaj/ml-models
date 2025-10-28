import joblib
import pandas as pd

pipe = joblib.load("au_house_price_model.joblib")

sample = pd.DataFrame([{
    "Rooms": 3,
    "Bedroom2": 3,
    "Bathroom": 2,
    "Car": 1,
    "Landsize": 350.0,
    "BuildingArea": 120.0,
    "YearBuilt": 1995,
    "Distance": 12.0,
    "Propertycount": 8000,
    "Postcode": 3070,
    "Lattitude": -37.76,
    "Longtitude": 144.99,
    "Suburb": "Northcote",
    "Address": "123 Smith St",
    "Type": "h",          # h=house, u=unit, t=townhouse (dataset codes)
    "Method": "S",        # S=sold, SP=Sold Prior, etc.
    "SellerG": "Nelson",
    "CouncilArea": "Darebin",
    "Regionname": "Northern Metropolitan"
}])

pred = pipe.predict(sample)[0]
print(f"Predicted price: ${pred:,.0f}")
