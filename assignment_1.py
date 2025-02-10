import numpy as np
import matplotlib.pyplot as plt

task2 = True
while task2:
    def demand_curve(q):
        return 290 - 1.5*q

    def supply_curve(q):
        return 15 + 0.8*q

    def revenue_curve(q):
        return q*demand_curve(q)

    def cost_curve(q):
        return q*supply_curve(q)

    def marginal_revenue_curve(q):
        return 290 - 3*q

    def marginal_cost_curve(q):
        return 15 + 0.8*q
        
    competition = 'monopoly'  # 'perfect' or 'monopoly'
    market_price, quantity = None, None
    if competition == 'perfect':
        # Find the intersection point
        def find_intersection():
            for x in np.linspace(0, 150, 1000):
                if np.isclose(demand_curve(x), supply_curve(x), atol=0.1):
                    return x, demand_curve(x)
            return None, None

        intersection_x, intersection_y = find_intersection()
        print(intersection_x, intersection_y)
        market_price = intersection_y

    elif competition == 'monopoly':
        # Find the intersection point
        def find_intersection():
            for x in np.linspace(0, 200, 1000):
                if np.isclose(marginal_revenue_curve(x), marginal_cost_curve(x), atol=1):
                    return x, demand_curve(x)
            return None, None

        intersection_x, intersection_y = find_intersection()
        print(intersection_x, intersection_y)
        market_price, quantity = intersection_y, intersection_x

    xs = np.linspace(0, 150, 1000)
    ys_demand = demand_curve(xs)
    ys_supply = supply_curve(xs)
    ys_marginal_revenue = marginal_revenue_curve(xs)
    ys_marginal_cost = marginal_cost_curve(xs)

    figure = plt.figure()
    plt.plot(xs, ys_demand, label='Demand')
    plt.plot(xs, ys_supply, label='Supply')
    plt.plot(xs, ys_marginal_revenue, label='Marginal Revenue')
    plt.plot(xs, ys_marginal_cost, label='Marginal Cost')
    plt.axhline(y=market_price, color='green', linestyle='--')  # Add horisontal
    plt.fill_between(xs, ys_demand, market_price, where=(ys_demand > market_price), interpolate=True, color='blue', alpha=0.3)
    plt.fill_between(xs, ys_supply, market_price, where=np.logical_and(market_price > ys_supply, xs < quantity), interpolate=True, color='red', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-')  # Add vertical
    plt.axhline(y=0, color='gray', linestyle='-')  # Add horisontal

    # Add text to the middle of the shaded areas
    plt.text(22.5, 210, 'Consumer Surplus', horizontalalignment='center', color='blue')
    plt.text(30, 120, 'Producer Surplus', horizontalalignment='center', color='red')

    # Add point at the intersection
    if intersection_x is not None and intersection_y is not None:
        plt.scatter(intersection_x, intersection_y, color='black')
        plt.text(intersection_x, intersection_y+5, f'({intersection_x:.2f}, {intersection_y:.2f})', horizontalalignment='left')


    plt.title('Single Producer Market, Monopoly')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.legend(loc='upper right')
    plt.show()

task3 = False
while task3:
    plants = {  "Plant 1":[5000, 0],
                "Plant 2":[2500, 70],
                "Plant 3":[2000, 90]
            }

    def demand_curve(q):
        return 280 - 0.02*q

    def supply_curve(q, plants):

        if q <= plants["Plant 1"][0]:
            return plants["Plant 1"][1]
        elif q <= plants["Plant 1"][0] + plants["Plant 2"][0]:
            return plants["Plant 2"][1]
        elif q <= plants["Plant 1"][0] + plants["Plant 2"][0] + plants["Plant 3"][0]:
            return plants["Plant 3"][1]
        else: 
            return 0


    def revenue_curve(q):
        return q*demand_curve(q)

    def cost_curve(q, plants):
        return q*supply_curve(q, plants)

    def marginal_revenue_curve(q):
        return 280 - 0.04*q

    def marginal_cost_curve(q, plants):
        if q <= plants["Plant 1"][0]:
            return plants["Plant 1"][1]
        elif q <= plants["Plant 1"][0] + plants["Plant 2"][0]:
            return plants["Plant 2"][1]
        elif q <= plants["Plant 1"][0] + plants["Plant 2"][0] + plants["Plant 3"][0]:
            return plants["Plant 3"][1]
        else: 
            return 0


    xs = np.linspace(0,10000,2000)
    vectorized_supply_curve = np.vectorize(supply_curve)
    vectorized_marginal_cost_curve = np.vectorize(marginal_cost_curve)
    ys_demand = demand_curve(xs)
    ys_supply = vectorized_supply_curve(xs, plants)
    ys_marginal_revenue = marginal_revenue_curve(xs)
    ys_marginal_cost = vectorized_marginal_cost_curve(xs, plants)

    # Find the intersection point
    def find_intersection():
        for x in xs:
            if np.isclose(marginal_revenue_curve(x), vectorized_marginal_cost_curve(x,plants), atol=0.1):
                return x, demand_curve(x)
        return None, None

    intersection_x, intersection_y = find_intersection()
    print(round(intersection_x), round(intersection_y))
    market_price = round(intersection_y)
    quantity = round(intersection_x)


    figure = plt.figure()
    plt.axvline(x=0, color='gray', linestyle='-')  # Add vertical
    plt.axhline(y=0, color='gray', linestyle='-')  # Add horisontal
    plt.plot(xs, ys_demand, label='Demand')
    plt.plot(xs, ys_supply, label='Supply')
    #plt.plot(xs, ys_marginal_revenue, label='Marginal Revenue')
    #plt.plot(xs, ys_marginal_cost, label='Marginal Cost')
    #plt.axhline(y=market_price, color='green', linestyle='--')  # Add horisontal
    #plt.fill_between(xs, ys_demand, market_price, where=(ys_demand > market_price), interpolate=True, color='blue', alpha=0.3)
    #plt.fill_between(xs, ys_supply, market_price, where=np.logical_and(market_price > ys_supply, xs < quantity), interpolate=True, color='red', alpha=0.3)


    # Add text to the middle of the shaded areas
    #plt.text(2500, 190, 'Consumer Surplus', horizontalalignment='center', color='blue')
    #plt.text(2500, 100, 'Producer Surplus', horizontalalignment='center', color='red')

    # Add point at the intersection
    #if intersection_x is not None and intersection_y is not None:
    #    plt.scatter(quantity, market_price, color='black')
    #    plt.text(quantity, market_price+5, f'({quantity}, {market_price})', horizontalalignment='left')


    plt.title('Different Market Solutions')
    plt.xlabel('Quantity [MW]')
    plt.ylabel('Price [NOK/MWh]')
    plt.legend(loc='upper right')
    plt.show()