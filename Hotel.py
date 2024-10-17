#Three lines to make our compiler able to draw:
import sys
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

full_health_data = pd.read_csv("data.csv", header=0, sep=",")

print(df.head())
print(df.describe())
print(df.info())

df.dropna(inplace=True)

df['booking_lead_time'] = (pd.to_datetime(df['arrival_date']) - pd.to_datetime(df['booking_date'])).dt.days
df['total_guests'] = df['adults'] + df['children'] + df['babies']



df['month'] = pd.to_datetime(df['arrival_date']).dt.month
monthly_rates = df.groupby('month')['adr'].mean().reset_index()
plt.plot(monthly_rates['month'], monthly_rates['adr'])
plt.title('Average Daily Rate by Month')
plt.xlabel('Month')
plt.ylabel('Average Daily Rate')
plt.show()

length_of_stay = df.groupby('length_of_stay')['adr'].mean().reset_index()
plt.bar(length_of_stay['length_of_stay'], length_of_stay['adr'])
plt.title('ADR by Length of Stay')
plt.xlabel('Length of Stay (nights)')
plt.ylabel('Average Daily Rate')
plt.show()

special_requests = df.groupby('special_requests')['adr'].mean().reset_index()
plt.bar(special_requests['special_requests'], special_requests['adr'])
plt.title('ADR by Number of Special Requests')
plt.xlabel('Number of Special Requests')
plt.ylabel('Average Daily Rate')
plt.show()

hotel_comparison = df.groupby('hotel_type')['adr'].mean().reset_index()
plt.bar(hotel_comparison['hotel_type'], hotel_comparison['adr'])
plt.title('Comparison of ADR between City and Resort Hotels')
plt.xlabel('Hotel Type')
plt.ylabel('Average Daily Rate')
plt.show()

from sklearn.model_selection import train_test_split

X = df[['total_guests', 'booking_lead_time', 'length_of_stay']]
y = df['cancellation']  # Assuming cancellation is a binary variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


#Two lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
