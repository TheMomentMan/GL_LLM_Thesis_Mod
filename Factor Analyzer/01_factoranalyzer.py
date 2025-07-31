from factor_analyzer import FactorAnalyzer
import pandas as pd

df = pd.read_csv('For_CMB_test.csv')
df_numeric = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5','Gen3','Gen4','Gen5','Gen6','Gen7','Gen8']]  # or include Gen3–Gen8 if applicable

fa = FactorAnalyzer()
fa.fit(df_numeric)
ev, v = fa.get_eigenvalues()
print("Explained Variance:", ev)
