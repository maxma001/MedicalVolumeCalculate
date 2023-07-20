from function import create3DArrayPh,calallvalues

labels_outPh = create3DArrayPh()
df = calallvalues(labels_outPh).reset_index(drop=True)
print(df)