from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
import pandas as pd
# Load dataset
#df = pd.read_csv("Experience_LLM_30thJuly_1349.csv")

df=pd.read_excel('All_LLM_Data_everything.xlsx')
gen_df = df[["Gen3", "Gen4", "Gen5", "Gen6", "Gen7", "Gen8"]].dropna()

kmo_all, kmo_model = calculate_kmo(gen_df)
bartlett_chi_square, bartlett_p_value = calculate_bartlett_sphericity(gen_df)

print("KMO Measure:", kmo_model)
print("Bartlett’s Test:", bartlett_chi_square, "p =", bartlett_p_value)
