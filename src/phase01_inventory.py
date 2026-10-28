from common import inventory, OUT
df = inventory()
path = OUT/"metadata"/"video_inventory.csv"
df.to_csv(path, index=False)
print("Videos found:", len(df))
if len(df): print(df.groupby(["category","label"]).size().to_string())
else: print("Add videos beneath dataset/videos/<Category>/")
print("Saved:", path)
