import pandas as pd
import random
from datetime import datetime, timedelta

# Dati sporchi per testare l'AI
vendors = {
    "AMZN MKTPLACE": "Shopping",
    "STARBUCKS COFFEE": "Food",
    "UBER TRIP": "Transport",
    "ESSELUNGA MILANO": "Groceries",
    "NETFLIX.COM": "Subscriptions",
    "ZARA BERSHKA": "Shopping"
}

data = []
for i in range(50):
    raw_desc = random.choice(list(vendors.keys())) + " " + str(random.randint(100, 999))
    date = datetime.now() - timedelta(days=random.randint(0, 30))
    amount = round(random.uniform(5.0, 150.0), 2)
    card = random.choice(["Visa ...1234", "Amex ...5678"])
    data.append([date, raw_desc, amount, card])

df = pd.DataFrame(data, columns=["Date", "Raw_Description", "Amount", "Card"])
df.head()
df.to_csv("data/transactions.csv", index=False)
print("✅ File data/transactions.csv generato con successo!")