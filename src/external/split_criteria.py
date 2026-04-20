                        
 
                                                               
 
                                            
 
 
from collections import Counter

import numpy as np

                                                                                           
                                                                             
_EPSILON = 1e-10


 
 
 
def gini(left, right):
    """
    Calculate the Gini impurity for a split.

    Gini impurity is a measure of how often a randomly chosen element
    from the set would be incorrectly labeled if it was randomly labeled
    according to the distribution of labels in the subset.
    Lower values indicate less impurity (better split).

    Formula: Gini = (n_left / n_total) * Gini(left) + (n_right / n_total) * Gini(right)
             Gini(subset) = 1 - sum(p_i^2) for class i in subset

    Parameters
    ----------
    left : array-like
        Labels of samples in the left subset.
    right : array-like
        Labels of samples in the right subset.

    Returns
    -------
    float
        The weighted Gini impurity of the split. Returns np.inf if the
        total number of samples is zero (invalid state).
    """
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right

    if n_total == 0:
                                                   
        return np.inf                                                  

    gini_total = 0.0

                                                 
    if n_left > 0:
        p_left = n_left / n_total
        counter_left = Counter(left)
                                                    
        impurity_left = 1.0 - sum(
            (count / n_left) ** 2 for count in counter_left.values()
        )
        gini_total += p_left * impurity_left

                                                  
    if n_right > 0:
        p_right = n_right / n_total
        counter_right = Counter(right)
                                                     
        impurity_right = 1.0 - sum(
            (count / n_right) ** 2 for count in counter_right.values()
        )
        gini_total += p_right * impurity_right

    return gini_total


def twoing(left_label, right_label):
    """
    Calculate the Twoing criterion value for a split, returned as a value
    to be minimized (negative of the original criterion).

    The original Twoing criterion (from OC1/Breiman et al.) measures the
    difference in class distributions between the left and right nodes.
    It is maximized for the best split.
    Formula (Original): T = (n_left / n_total) * (n_right / n_total) / 4 * [ sum(|PL_i - PR_i|) ]^2
                        where PL_i = proportion of class i in left node
                              PR_i = proportion of class i in right node

    This function returns -T, so that minimizing this value corresponds
    to maximizing the original Twoing criterion. Lower (more negative)
    values indicate a better split. A value of 0 indicates no difference
    in distributions.

    Parameters
    ----------
    left_label : array-like
        Labels of samples in the left subset.
    right_label : array-like
        Labels of samples in the right subset.

    Returns
    -------
    float
        The negative of the Twoing criterion value. Returns np.inf if the
        total number of samples is zero (invalid state).
    """
    n_left = len(left_label)
    n_right = len(right_label)
    n_total = n_left + n_right

    if n_total == 0:
                                          
        return np.inf                                                  

                                                                    
                                                     
    if n_left > 0 and n_right > 0:
        labels = np.concatenate((left_label, right_label))
    elif n_left > 0:
        labels = left_label
    elif n_right > 0:
        labels = right_label
    else:                                                             
        return 0.0                                                                               

    unique_classes = np.unique(labels)

    sum_abs_diff = 0.0
    for i in unique_classes:
                                                              
        count_left = np.count_nonzero(left_label == i)
        pl_i = (count_left / n_left) if n_left > 0 else 0.0

                                                                
        count_right = np.count_nonzero(right_label == i)
        pr_i = (count_right / n_right) if n_right > 0 else 0.0

        sum_abs_diff += abs(pl_i - pr_i)

                                         
    p_left = n_left / n_total
    p_right = n_right / n_total
    twoing_value = (p_left * p_right / 4.0) * (sum_abs_diff**2)

                                                        
    return -twoing_value
