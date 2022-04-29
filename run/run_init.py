import pandas as pd

order_data = pd.read_csv("finalOrderData/orderData1.csv")
data = order_data.drop_duplicates(subset=['merchant_latitude', 'merchant_longitude'], keep='first')[
    ['merchant_latitude', 'merchant_longitude', 'arrival_id']]
# data.reset_index(drop=True)
for _ in range(29):
    filename = "finalOrderData/orderData" + str(_ + 2) + ".csv"
    dd = pd.read_csv(filename)
    data1 = dd.drop_duplicates(subset=['merchant_latitude', 'merchant_longitude'], keep='first')[
        ['merchant_latitude', 'merchant_longitude', 'arrival_id']]
    data = pd.concat([data, data1], axis=0)
data = data.drop_duplicates(subset=['merchant_latitude', 'merchant_longitude'], keep='first')
data.reset_index(drop=True)
data.to_csv("finalOrderData/shopInitialization.csv")
# import pandas as pd
#
# t = 0
# for _ in range(30):
#     filename = "finalOrderData/orderData" + str(_ + 1) + ".csv"
#     dd = pd.read_csv(filename)
#     t += dd.shape[0]
# print(t)
