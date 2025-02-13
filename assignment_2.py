producers = {"price": [200, 240, 450, 600, 700],
             "quantity": [250, 75, 100, 150, 200]}

consumers = {"price":[900, 700, 650, 500, 350],
             "quantity":[150, 100, 250, 100, 150]}



Utility = consumers["quantity"][0] * consumers["price"][0]
cost_of_electricity = producers["quantity"][0] * producers["price"][0]
print(Utility - cost_of_electricity)