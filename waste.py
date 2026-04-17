from matplotlib import pyplot as plt
import numpy as np

years = ["Year 1", "Year 2", "Year 3"]

device_sales = [800000, 2000000, 4500000]
subscriptions = [500000, 1500000, 3500000]
b2b = [600000, 1800000, 4000000]
premium = [100000, 500000, 1500000]

x = np.arange(len(years))
width = 0.2

plt.figure()
plt.bar(x - 1.5*width, device_sales, width, label="Device Sales")
plt.bar(x - 0.5*width, subscriptions, width, label="Subscriptions")
plt.bar(x + 0.5*width, b2b, width, label="B2B Sales")
plt.bar(x + 1.5*width, premium, width, label="Premium Features")

plt.xticks(x, years)
plt.title("Revenue Streams")
plt.legend()
plt.show()

plt.close()

