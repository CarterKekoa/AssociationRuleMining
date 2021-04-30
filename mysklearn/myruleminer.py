##############################################
# Programmer: Carter Mooring
# Class: CPCS 322-02, Spring 2021
# Programming Assignment #7
# April 29th, 2021
# 
# Description: 
##############################################

import mysklearn.myutils as myutils
from tabulate import tabulate

class MyAssociationRuleMiner:
    """Represents an association rule miner.

    Attributes:
        minsup(float): The minimum support value to use when computing supported itemsets
        minconf(float): The minimum confidence value to use when generating rules
        X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
        rules(list of dict): The generated rules

    Notes:
        Implements the apriori algorithm
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, minsup=0.25, minconf=0.8):
        """Initializer for MyAssociationRuleMiner.

        Args:
            minsup(float): The minimum support value to use when computing supported itemsets
                (0.25 if a value is not provided and the default minsup should be used)
            minconf(float): The minimum confidence value to use when generating rules
                (0.8 if a value is not provided and the default minconf should be used)
        """
        self.minsup = minsup
        self.minconf = minconf
        self.X_train = None 
        self.rules = None

    def fit(self, X_train):
        """Fits an association rule miner to X_train using the Apriori algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)

        Notes:
            Store the list of generated association rules in the rules attribute
            If X_train represents a non-market basket analysis dataset, then: 
                Attribute labels should be prepended to attribute values in X_train
                    before fit() is called (e.g. "att=val", ...).
                Make sure a rule does not include the same attribute more than once
        """
        self.X_train = X_train
        tag = None

        # some attributes have similar value names, if so we need to assign attribute tags to their values
        if(myutils.att_necessary(X_train)): 
            tag = ['att'+str(i) for i in range(len(X_train[0]))] # attach att# tag to front
            myutils.prepend_attribute_label(X_train, tag)
        
        self.rules = myutils.apriori(X_train, self.minsup, self.minconf)

        # compute the rules confidence, support, lift
        for rule in self.rules:
            myutils.interestingness_measure(rule, X_train)

        return self.rules

    def print_association_rules(self):
        """Prints the association rules in the format "IF val AND ... THEN val AND...", one rule on each line.

        Notes: 
            Each rule's output should include an identifying number, the rule, the rule's support, the rule's confidence, and the rule's lift 
            Consider using the tabulate library to help with this: https://pypi.org/project/tabulate/
        """
        header = ["Number", "Association Rule", "Support", "Confidence", "Lift"]
        table = []

        # iterate through all the rules
        for i in range(len(self.rules)):
            row = myutils.ruleMine_pretty_print(self.rules[i], (i + 1))
            table.append(row)

        print(tabulate(table, headers=header))