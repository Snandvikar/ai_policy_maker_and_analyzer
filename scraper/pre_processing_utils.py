
import ast
import re
import pandas as pd

def clean_column_name(col):
    
    col = str(col).lower()
    col = col.strip()
    
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    
    return col.strip("_")



def extract_avg(value):
    if pd.isna(value):
        return value

    # ✅ Case 1: already a dictionary
    if isinstance(value, dict):
        return value.get("avg", value)

    # ✅ Case 2: string representation of dict
    if isinstance(value, str):
        value = value.strip()

        if value.startswith("{"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, dict):
                    return parsed.get("avg", value)
            except:
                return value

    return value


def extract_year(value):

    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    
    if match:
        return int(match.group())
    
    return value


def convert_numeric(df):

    for col in df.columns:
        
        if df[col].dtype == "object":
            
            numeric = pd.to_numeric(df[col], errors="coerce")
            
            if numeric.notna().sum() > len(df) * 0.5:
                df[col] = numeric
    
    return df

COLUMN_MAPPINGS = {

"gender_wise_total_msme_7973": {
"i7973_6": "Male",
"i7973_7": "Female",
"i7973_8": "Others",
"i7973_9": "Unidentified",
"i7973_10": "Total"
},

"slum_census_2011_9022": {
"i9022_3": "Share_of_slum_population"
},

"vulnerable_women_socio_economic_9429": {

"i9429_3": "Female_population",
"i9429_4": "Sex_ratio",
"i9429_5": "illiteracy_percent",
"i9429_6": "Women_headed_households",
"i9429_7": "Maternal_mortality_rate",
"i9429_8": "Crime_against_women",
"i9429_9": "Dependent_women_percent",
"i9429_10": "Gender_vulnerability_index"

},

"electronic_transaction_aggregation_per_1000_population_9131": {

"i9131_4": "Population",
"i9131_5": "Number_of_e_transactions",
"i9131_6": "E_transactions_per_1000_population"

},

"socio_economic_census_7086": {

"i7086_4": "Households",
"i7086_5": "Landless_manual_labour",
"i7086_6": "Non_agri_enterprises",
"i7086_7": "income_tax_households",
"i7086_8": "Destitute_households",
"i7086_9": "Govt_salaried",
"i7086_10": "Public_salaried",
"i7086_11": "Private_salaried",
"i7086_12": "income_less_5000",
"i7086_13": "income_5000_10000",
"i7086_14": "income_greater_10000",
"i7086_15": "Govt_income_gt_5000",
"i7086_16": "Motorized_vehicle",
"i7086_17": "Two_wheeler",
"i7086_18": "Three_wheeler",
"i7086_19": "Four_wheeler",
"i7086_20": "Fishing_boat",
"i7086_21": "Refrigerator",
"i7086_22": "Landline",
"i7086_23": "Mobile_only",
"i7086_24": "Landline_and_mobile",
"i7086_25": "No_phone",
"i7086_26": "Agri_equipment",
"i7086_27": "Kisan_credit",
"i7086_28": "Unirrigated_land",
"i7086_29": "irrigated_land",
"i7086_30": "Other_land",
"i7086_31": "irrigation_equipment",
"i7086_32": "No_land_irrigation",
"i7086_33": "No_land_kisan_credit"

}

}

KEEP_COLUMNS = {

"gender_wise_total_msme_7973": [
"Male",
"Female",
"Others",
"Unidentified",
"Total"
],

"slum_census_2011_9022":[
"Share_of_slum_population"
],

"vulnerable_women_socio_economic_9429":[
"Female_population",
"Sex_ratio",
"illiteracy_percent",
"Women_headed_households",
"Maternal_mortality_rate",
"Crime_against_women",
"Dependent_women_percent",
"Gender_vulnerability_index"
],

"electronic_transaction_aggregation_per_1000_population_9131":[
"Population",
"Number_of_e_transactions",
"E_transactions_per_1000_population"
],

"socio_economic_census_7086":[
"Households",
"Landless_manual_labour",
"Non_agri_enterprises",
"Private_salaried",
"income_greater_10000",
"Mobile_only",
"Refrigerator",
"Four_wheeler"
]

}

def rename_ndap_columns(df, dataset_name):
    print(f"RENAMED_COLUMNS - {dataset_name} -------------------------------------------")
    mapping = COLUMN_MAPPINGS[dataset_name]
    
    new_columns = {}
    
    for col in df.columns:
        
        col_clean = col.lower().strip()
        
        # check if column contains ID
        for key in mapping:
            if key.lower() in col_clean:
                new_columns[col] = mapping[key]
                print(f"{col}  →  {mapping[key]}")
    
    df = df.rename(columns=new_columns)
    
    return df

def preprocess_dataset(df, dataset_name):

    # print("############### DEBUGGING ##########")
    # if (dataset_name == "slum_census_2011_9022"):
    #     print(df["I9022_3"].iloc[0])
    #     print(extract_avg(df["I9022_3"].iloc[0]))
        # print(df.columns)

    # extract avg
    df = df.apply(lambda col: col.map(extract_avg))

    # extract year
    for col in df.columns:
        if "year" in col.lower():
            df[col] = df[col].apply(extract_year)

    # drop CalendarDay
    df = df.drop(columns=["CalendarDay"], errors="ignore")

    df = rename_ndap_columns(df, dataset_name)


    # clean names
    df.columns = [clean_column_name(c) for c in df.columns]

    geo_cols = [
        "country",
        "statename",
        "statecode",
        "state",
        "city",
        "district", "year" ,"districtname","districtcode"
    ]

    # # keep only required columns
    # keep_cols = KEEP_COLUMNS[dataset_name]
    # for i in geo_cols :
    #     keep_cols.append(i)

    # final_cols = [
    #     col for col in keep_cols
    #     if col in df.columns
    # ]

    # df = df[final_cols]

    # convert numeric
    df = convert_numeric(df)

    return df