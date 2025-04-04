producers = {"price": [200, 240, 450, 600, 700],
             "quantity": [250, 75, 100, 150, 200]}

consumers = {"price":[900, 700, 650, 500, 350],
             "quantity":[150, 100, 250, 100, 150]}



Utility = consumers["quantity"][0] * consumers["price"][0] + \
        consumers["quantity"][1] * consumers["price"][1] + \
        consumers["quantity"][2] * consumers["price"][2] + \
        consumers["quantity"][3] * consumers["price"][3] + \
        consumers["quantity"][4] * consumers["price"][4]

cost_of_electricity =   250 * producers["price"][0] + \
                        75 * producers["price"][1] + \
                        100 * producers["price"][2] + \
                        150 * producers["price"][3] + \
                        175 * producers["price"][4]
print(Utility - cost_of_electricity)