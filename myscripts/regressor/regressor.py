import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score

def preprocess_features(X):
    output = pd.DataFrame(index=X.index)
    for col, col_data in X.iteritems():
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)
        output = output.join(col_data)
    return output

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn import grid_search

def performance_metric(y_true, y_predict):
    return r2_score(y_true, y_predict)

def fit_model(X, Y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)
    
    regressor = DecisionTreeRegressor()
    params = {'max_depth': list(range(1,10))}
    scoring_fnc = make_scorer(performance_metric)
    grid = grid_search.GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, Y)
    return grid.best_estimator_

def main():
    data = pd.read_csv('housing.csv')
    feature_cols = list(data.columns[:-1])
    target_col = data.columns[-1]

    X_all = data[feature_cols]
    Y_all = data[target_col]

    X_all = preprocess_features(X_all)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

    reg_obj = fit_model(X_train, Y_train)
    print "Parameter 'max_depth' is {} for the optimal model.".format(reg_obj.get_params()['max_depth'])    
    client_data = [[5, 34, 15], # Client 1
                    [4, 55, 22], # Client 2
                    [8, 7, 12]]  # Client 3
    # Show predictions
    for i, price in enumerate(reg_obj.predict(client_data)):
        print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)    

if __name__ == '__main__':
    main()



