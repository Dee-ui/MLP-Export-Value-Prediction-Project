from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf

# Initialize FastAPI
app = FastAPI()

# Load saved models for each target
model_paths = [
    "model_fruits_vegetables.joblib",
    "model_non_food_items.joblib",
    "model_other_food_items.joblib",
    "model_sugar_honey.joblib",
    "model_tobacco.joblib"
]
models = [joblib.load(path) for path in model_paths]

# Define available country names for one-hot encoding
country_names = [
    'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla',
    'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia',
    'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus',
    "Area_Afghanistan", "Area_Albania", "Area_Algeria", "Area_American Samoa", "Area_Andorra", "Area_Angola", "Area_Anguilla",
    "Area_Antarctica", "Area_Antigua and Barbuda", "Area_Argentina", "Area_Armenia", "Area_Aruba", "Area_Australia", "Area_Austria",
    "Area_Azerbaijan", "Area_Bahamas", "Area_Bahrain", "Area_Bangladesh", "Area_Barbados", "Area_Belarus", "Area_Belgium", 
    "Area_Belize", "Area_Benin", "Area_Bermuda", "Area_Bhutan", "Area_Bolivia (Plurinational State of)", "Area_Bonaire",
    "Sint Eustatius and Saba", "Area_Bosnia and Herzegovina", "Area_Botswana", "Area_Brazil", "Area_British Virgin Islands", 
    "Area_Brunei Darussalam", "Area_Bulgaria", "Area_Burkina Faso", "Area_Burundi", "Area_Cabo Verde", "Area_Cambodia", 
    "Area_Cameroon", "Area_Canada", "Area_Cayman Islands", "Area_Central African Republic", "Area_Chad", "Area_Channel Islands",
    "Area_Chile", "Area_China","Area_China, Hong Kong SAR","Area_China, Macao SAR","Area_China, Taiwan Province of",
    "Area_China, mainland", "Area_Christmas Island", "Area_Cocos (Keeling) Islands", "Area_Colombia", "Area_Comoros",
    "Area_Congo", "Area_Cook Islands", "Area_Costa Rica", "Area_Croatia", "Area_Cuba", "Area_Curaçao", "Area_Cyprus",
    "Area_Czechia", "Area_Côte d'Ivoire", "Area_Democratic People's Republic of Korea", "Area_Democratic Republic of the Congo",
    "Area_Denmark", "Area_Djibouti", "Area_Dominica", "Area_Dominican Republic", "Area_Ecuador", "Area_Egypt", "Area_El Salvador",
    "Area_Equatorial Guinea", "Area_Eritrea", "Area_Estonia", "Area_Eswatini", "Area_Ethiopia", "Area_Falkland Islands (Malvinas)", 
    "Area_Faroe Islands","Area_Fiji","Area_Finland","Area_France","Area_French Guiana","Area_French Polynesia",
    "Area_French Southern Territories","Area_Gabon","Area_Gambia","Area_Georgia","Area_Germany","Area_Ghana","Area_Gibraltar",
    "Area_Greece","Area_Greenland","Area_Grenada","Area_Guadeloupe","Area_Guam","Area_Guatemala","Area_Guernsey","Area_Guinea",
    "Area_Guinea-Bissau","Area_Guyana","Area_Haiti","Area_Holy See","Area_Honduras","Area_Hungary","Area_Iceland","Area_India",
    "Area_Indonesia","Area_Iran (Islamic Republic of)","Area_Iraq","Area_Ireland","Area_Isle of Man","Area_Israel","Area_Italy",
    "Area_Jamaica","Area_Japan","Area_Jersey","Area_Jordan","Area_Kazakhstan","Area_Kenya","Area_Kiribati","Area_Kuwait",
    "Area_Kyrgyzstan","Area_Lao People's Democratic Republic","Area_Latvia","Area_Lebanon","Area_Lesotho","Area_Liberia",
    "Area_Libya","Area_Liechtenstein","Area_Lithuania","Area_Luxembourg","Area_Madagascar","Area_Malawi","Area_Malaysia",
    "Area_Maldives","Area_Mali","Area_Malta","Area_Marshall Islands","Area_Martinique","Area_Mauritania","Area_Mauritius",
    "Area_Mayotte","Area_Mexico","Area_Micronesia (Federated States of)","Area_Midway Island","Area_Monaco","Area_Mongolia",
    "Area_Montenegro","Area_Montserrat","Area_Morocco","Area_Mozambique","Area_Myanmar","Area_Namibia","Area_Nauru","Area_Nepal",
    "Area_Netherlands (Kingdom of the)","Area_Netherlands Antilles (former)","Area_New Caledonia","Area_New Zealand",
    "Area_Nicaragua","Area_Niger","Area_Nigeria","Area_Niue","Area_Norfolk Island","Area_North Macedonia",
    "Area_Northern Mariana Islands","Area_Norway","Area_Oman","Area_Pakistan","Area_Palau","Area_Palestine","Area_Panama",
    "Area_Papua New Guinea","Area_Paraguay","Area_Peru","Area_Philippines","Area_Pitcairn","Area_Poland","Area_Portugal",
    "Area_Puerto Rico","Area_Qatar","Area_Republic of Korea","Area_Republic of Moldova","Area_Romania","Area_Russian Federation",
    "Area_Rwanda","Area_Réunion","Area_Saint Barthélemy","Area_Saint Helena, Ascension and Tristan da Cunha",
    "Area_Saint Kitts and Nevis","Area_Saint Lucia","Area_Saint Martin (French part)","Area_Saint Pierre and Miquelon",
    "Area_Saint Vincent and the Grenadines","Area_Samoa","Area_San Marino","Area_Sao Tome and Principe","Area_Saudi Arabia",
    "Area_Senegal","Area_Serbia","Area_Serbia and Montenegro","Area_Seychelles","Area_Sierra Leone","Area_Singapore",
    "Area_Sint Maarten (Dutch part)","Area_Slovakia","Area_Slovenia","Area_Solomon Islands","Area_Somalia","Area_South Africa",
    "Area_South Georgia and the South Sandwich Islands","Area_South Sudan","Area_Spain","Area_Sri Lanka","Area_Sudan",
    "Area_Sudan (former)","Area_Suriname","Area_Svalbard and Jan Mayen Islands","Area_Sweden","Area_Switzerland",
    "Area_Syrian Arab Republic","Area_Tajikistan","Area_Thailand","Area_Timor-Leste","Area_Togo","Area_Tokelau",
    "Area_Tonga","Area_Trinidad and Tobago","Area_Tunisia","Area_Turkmenistan","Area_Turks and Caicos Islands",
    "Area_Tuvalu","Area_Türkiye","Area_Uganda","Area_Ukraine","Area_United Arab Emirates",
    "Area_United Kingdom of Great Britain and Northern Ireland","Area_United Republic of Tanzania",
    "Area_United States Virgin Islands","Area_United States of America","Area_Uruguay","Area_Uzbekistan",
    "Area_Vanuatu","Area_Venezuela (Bolivarian Republic of)","Area_Viet Nam","Area_Wake Island","Area_Wallis and Futuna Islands",
    "Area_Western Sahara","Area_Yemen","Area_Zambia","Area_Zimbabwe","Area_Åland Islands"


]

# Define the input data model
class PredictionInput(BaseModel):
    year: int
    standard_dev_temp_dec_jan_feb: float
    standard_dev_temp_jun_jul_aug: float
    standard_dev_temp_mar_apr_may: float
    standard_dev_temp_meteorological_year: float
    standard_dev_temp_sept_oct_nov: float
    change_temp_dec_jan_feb: float
    change_temp_jun_jul_aug: float
    change_temp_mar_apr_may: float
    change_temp_meteorological_year: float
    change_temp_sept_oct_nov: float
    avg_exchange_rate: float
    yield_cereals_primary: float
    yield_citrus_fruit_total: float
    yield_fibre_crops: float
    yield_fruit_primary: float
    yield_oilcrops_cake_equivalent: float
    yield_oilcrops_oil_equivalent: float
    yield_pulses_total: float
    yield_roots_tubers_total: float
    yield_sugar_crops_primary: float
    yield_treenuts_total: float
    yield_vegetables_primary: float
    agri_land_mass: float
    land_mass_agriculture: float
    arable_land_mass: float
    area_land_mass: float
    cropland_mass: float
    land_area: float
    land_area_irrigation: float
    permanent_crops_area: float
    meadows_pastures_total_area: float
    total_area_temporary_crops: float
    total_area_temporary_fallow: float
    total_area_temporary_meadows: float
    country_start_country_name_with_Area_: str  # Add a 'country' field to accept country input

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    # Convert input data into an array format that matches model input
    input_data = [
        data.year, data.standard_dev_temp_dec_jan_feb, data.standard_dev_temp_jun_jul_aug,
        data.standard_dev_temp_mar_apr_may, data.standard_dev_temp_meteorological_year,
        data.standard_dev_temp_sept_oct_nov, data.change_temp_dec_jan_feb, data.change_temp_jun_jul_aug,
        data.change_temp_mar_apr_may, data.change_temp_meteorological_year, data.change_temp_sept_oct_nov,
        data.avg_exchange_rate, data.yield_cereals_primary, data.yield_citrus_fruit_total,
        data.yield_fibre_crops, data.yield_fruit_primary, data.yield_oilcrops_cake_equivalent,
        data.yield_oilcrops_oil_equivalent, data.yield_pulses_total, data.yield_roots_tubers_total,
        data.yield_sugar_crops_primary, data.yield_treenuts_total, data.yield_vegetables_primary,
        data.agri_land_mass, data.land_mass_agriculture, data.arable_land_mass, data.area_land_mass,
        data.cropland_mass, data.land_area, data.land_area_irrigation, data.permanent_crops_area,
        data.meadows_pastures_total_area, data.total_area_temporary_crops,
        data.total_area_temporary_fallow, data.total_area_temporary_meadows
    ]

    # Create one-hot encoding array for countries
    one_hot_countries = [1 if country == data.country else 0 for country in country_names]

    # Append one-hot encoded country data to the input features
    input_data.extend(one_hot_countries)
    input_data = np.array([input_data])

    # Placeholder to store predictions for each target label
    predictions = []

    # Generate predictions for each label for the next 3 years
    for model in models:
        yearly_predictions = []
        for year_ahead in range(1, 4):  # Predict for the next 1, 2, and 3 years
            input_data[0][0] = data.year + year_ahead  # Update year in the input data
            yearly_predictions.append(model.predict(input_data)[0])
        predictions.append(yearly_predictions)

    # Structure the response with predictions for each target label
    target_labels = [
        "Export Value of Fruits and Vegetables (USD)",
        "Export Value of Non-food Items (USD)",
        "Export Value of Other food Items (USD)",
        "Export Value of Sugar and Honey Items (USD)",
        "Export Value of Tobacco (USD)"
    ]

    # Construct the final output in a structured format
    response = {
        target_labels[i]: {
            "1_year_ahead": predictions[i][0],
            "2_years_ahead": predictions[i][1],
            "3_years_ahead": predictions[i][2]
        } for i in range(len(target_labels))
    }

    return response
