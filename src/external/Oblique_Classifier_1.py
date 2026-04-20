                                                                 
 
                                                              
 
from warnings import warn

import numpy as np

                                                                           
try:
    import src.external.split_criteria as split_criteria
    from src.external.OC1_tree_structure import LeafNode, Node, Tree
except ImportError:
                                                                     
                                                   
                                                                                     
    print("Warning: Could not import OC1 helper modules. Using placeholders.")

    class Node:
        pass

    class LeafNode(Node):
        pass

    class Tree:
        pass

    class split_criteria:
        @staticmethod
        def gini(left, right):
            """Handle gini."""
            return 0.5         

        @staticmethod
        def twoing(left, right):
            """Handle twoing."""
            return 0.5         


from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier

                                                                      
epsilon = 1e-6


 
                                                    
 
class BaseObliqueTree(BaseEstimator):

    def __init__(
        self,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_features_split,
        num_tries=5,
        random_state=42,
    ):

                                           
        """Initialize the instance."""
        self.criterion = (
            criterion                                                                 
        )
        self.max_depth = max_depth                             
        self.min_samples_split = (
            min_samples_split                                                
        )
        self.min_features_split = (
            min_features_split                                                 
        )
        self.min_samples_leaf = (
            min_samples_leaf                                                   
        )
        self.tree_ = None                                             

        self.num_tries = num_tries                                        

        self.random_state = random_state                                    

    def get_min_samples_leaf(self, n_samples):
        """Get min samples leaf."""
        min_samples = 1                                                         
        if isinstance(self.min_samples_leaf, int):
            if self.min_samples_leaf < 1:
                warn("min_samples_leaf specified less than 1; setting to value 1.")
                min_samples = 1
            else:
                min_samples = self.min_samples_leaf
        elif isinstance(self.min_samples_leaf, float):
                                                      
            if not 0.0 < self.min_samples_leaf <= 1.0:
                                                                                   
                                                                             
                warn(
                    "min_samples_leaf (float) not between (0, 1]; setting to default value of 1."
                )
                min_samples = 1
            else:
                                                       
                min_samples = max(1, int(np.ceil(self.min_samples_leaf * n_samples)))
        else:
            warn("Invalid type for min_samples_leaf; setting to default value of 1")
            min_samples = 1

                                                                                        
                                                                                                           
                                                                                                                
                                                                                  

        return min_samples

                                                  
    def get_min_samples_split(self, n_samples):
        """Get min samples split."""
        min_samples = 2                 
        if isinstance(self.min_samples_split, int):
            if self.min_samples_split < 2:
                warn("min_samples_split specified less than 2; setting to value 2.")
                min_samples = 2
            else:
                min_samples = self.min_samples_split
        elif isinstance(self.min_samples_split, float):
            if not 0.0 < self.min_samples_split <= 1.0:
                warn(
                    "min_samples_split not between (0, 1]; setting to default value of 2."
                )
                min_samples = 2
            else:
                min_samples = max(
                    2, int(np.ceil(self.min_samples_split * n_samples))
                )                     
        else:
            warn("Invalid type for min_samples_split; setting to default value of 2")
            min_samples = 2

        return min_samples

                                                   
    def get_min_features_split(self, n_features):
        """Get min features split."""
        min_features = 1                 
        if isinstance(self.min_features_split, int):
            if self.min_features_split < 1:
                warn("min_features_split specified less than 1; setting to value 1.")
                min_features = 1
            else:
                min_features = self.min_features_split
        elif isinstance(self.min_features_split, float):
            if not 0.0 < self.min_features_split <= 1.0:
                warn(
                    "min_features_split not between (0, 1]; setting to default value of 1."
                )
                min_features = 1
            else:
                min_features = max(
                    1, int(np.ceil(self.min_features_split * n_features))
                )                     
        else:
            warn("Invalid type for min_features_split; setting to default value of 1")
            min_features = 1

        return min_features

    def fit(self, X, y):
                                                 
        """Fit values."""
        np.random.seed(self.random_state)
                                                          
        X_train = np.array(X, dtype=float)
        y_train = np.array(y)

                                                  
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)

                                                                      
        if X_train.ndim == 1:                                                     
            if len(y_train) > 1:                                    
                X_train = X_train.reshape(-1, 1)
            elif len(y_train) == 1:                           
                X_train = X_train.reshape(1, -1)                           
            else:                     
                raise ValueError("Invalid X and y: No samples provided.")
        elif X_train.shape[0] != len(y_train):
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {X_train.shape[0]} != {len(y_train)}"
            )

                                                            
        if isinstance(self.criterion, str):
            if self.criterion == "gini":
                self._criterion_func = split_criteria.gini
            elif self.criterion == "twoing":
                self._criterion_func = split_criteria.twoing
            else:
                                        
                raise ValueError(
                    "Unrecognized string split criterion specified. Allowed strings are: "
                    '"gini", "twoing". Alternatively, provide a callable function.'
                )
        elif callable(self.criterion):
            self._criterion_func = self.criterion                               
        else:
            raise TypeError(
                "Criterion must be 'gini', 'twoing', or a callable function."
            )
                                               

        n_samples, n_features = X_train.shape
        min_samples_split = self.get_min_samples_split(n_samples)
                                                     
        min_samples_leaf = self.get_min_samples_leaf(n_samples)
                               
        min_features_split = self.get_min_features_split(n_features)

        is_clf = is_classifier(self)                              

         
                                            
         
        self.root_node_, self.learned_depth_ = build_oblique_tree_oc1(
            X_train,
            y_train,
            is_classification=is_clf,
            classes=self.classes_,                   
            criterion_func=self._criterion_func,
            max_depth=self.max_depth,
            min_samples_split=min_samples_split,
                                        
            min_samples_leaf=min_samples_leaf,
                              
            min_features_split=min_features_split,
            num_tries=self.num_tries,
        )
         
                              
         
                                                              
        if "Tree" in globals():
            self.tree_ = Tree(n_features=n_features, is_classifier=is_clf)
            self.tree_.set_root_node(self.root_node_)
            self.tree_.set_depth(self.learned_depth_)
        else:
            warn("Tree class not available. Skipping tree object creation.")
            self.tree_ = None                                         

        return self                                          

    def predict(self, X):
        """Predict values."""
        if self.tree_ is None and self.root_node_ is None:
            raise RuntimeError("The tree has not been fitted yet.")
                                                                         
        root = self.root_node_ if self.tree_ is None else self.tree_.root_node

                                                               
        X_pred = np.array(X, dtype=float)
        if X_pred.ndim == 1:                         
            X_pred = X_pred.reshape(1, -1)

                                                               
                                                                                  
        y = root.predict(X_pred)
        y = np.array(y, dtype=int)                                            
        return y

                                  
    def predict_proba(self, X):
        """Predict class probabilities for X."""
        if self.tree_ is None and self.root_node_ is None:
            raise RuntimeError("The tree has not been fitted yet.")

                                      
        X_pred = np.array(X, dtype=float)
        if X_pred.ndim == 1:
            X_pred = X_pred.reshape(1, -1)

                           
        root = self.root_node_ if self.tree_ is None else self.tree_.root_node

                                               
        probas = [root.get_proba(sample) for sample in X_pred]

        return np.array(probas)

    def get_params(self, deep=True):                         
        """Get params."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
                                            
            "min_samples_leaf": self.min_samples_leaf,
                             
            "min_features_split": self.min_features_split,
            "num_tries": self.num_tries,
        }

    def set_params(self, **parameters):
        """Set params."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


 
                                                    
 
class ObliqueClassifier1(ClassifierMixin, BaseObliqueTree):
    def __init__(
        self,
        criterion="gini",
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        min_features_split=1,
        random_state=42,
        num_tries=5,
    ):
        """Initialize the instance."""
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_features_split=min_features_split,
            num_tries=num_tries,
            random_state=random_state,
        )

                                                          
                                                      
                                               


                                                                                                       
def build_oblique_tree_oc1(
    X,                       
    y,                       
    classes,
    is_classification,
    criterion_func,                               
    max_depth,
    min_samples_split,
    min_samples_leaf,
    min_features_split,
    current_depth=0,
    current_features=None,
    current_samples=None,
    num_tries=5,
    debug=False,                                                                    
):

    """Build oblique tree oc1."""
    n_samples, n_features = X.shape

                                                  
    if current_depth == 0:
        current_features = np.arange(n_features)
        current_samples = np.arange(n_samples)

                                                   
    if is_classification:
                                                                                                                    
        if n_samples == 0:
                                                                               
                                                     
            warn("Reached node with zero samples.")
                                                      
            probabilities = np.zeros(len(classes)) 
                                    
            probabilities.fill(1.0 / len(classes))

            return (
                LeafNode(
                    is_classifier=is_classification,
                    value=0,
                    conf=0.0,
                    samples=np.array([], dtype=int),
                    features=current_features,
                    probabilities=probabilities,                    
                ),
                current_depth,
            )

                                                            
        majority_res = mode(y)

                                                                     
        if hasattr(majority_res, "mode"):                                           
            mode_val_internal = majority_res.mode
            count_val_internal = majority_res.count
        else:                               
            mode_val_internal, count_val_internal = majority_res

                                                               
                                                                                    
        if isinstance(mode_val_internal, np.ndarray):
            if mode_val_internal.ndim > 0:
                                                                                              
                label = mode_val_internal[0]
            else:
                                                                            
                label = mode_val_internal.item()
        else:
                                                                              
            label = mode_val_internal

                                                                                
        if isinstance(count_val_internal, np.ndarray):
            if count_val_internal.ndim > 0:
                current_count = count_val_internal[0]
            else:
                current_count = count_val_internal.item()
        else:
            current_count = count_val_internal

                                                                                
                                                                                       
        if n_samples > 0:
            conf = current_count / n_samples
        else:
            conf = 0.0                            
                                                   
    else:                                                                                   
        if n_samples == 0:
            warn("Reached node with zero samples.")
            return (
                LeafNode(
                    is_classifier=is_classification,
                    value=np.nan,
                    conf=0.0,
                    samples=np.array([], dtype=int),
                    features=current_features,
                ),
                current_depth,
            )

        mean_val = np.mean(y)
        std_val = np.std(y)
        label = mean_val
                                                                                   
        if (
            std_val > epsilon
        ):                                                                  
            conf = np.sum(np.abs(y - mean_val) <= std_val) / n_samples
        else:
            conf = 1.0 if n_samples > 0 else 0.0                           


                                               
    n_node_samples = len(y)
    if is_classification and n_node_samples > 0:
                                                                   
        unique_labels, counts = np.unique(y, return_counts=True)
        current_dist = dict(zip(unique_labels, counts))

                                                                          
        probabilities = np.zeros(len(classes))
        for i, c in enumerate(classes):
            probabilities[i] = current_dist.get(c, 0) / n_node_samples
    elif is_classification:             
        probabilities = np.zeros(len(classes))
        probabilities.fill(1.0 / len(classes))                                      
    else:                  
        probabilities = None
                     


                                                                       
                                                                               
                                                                                   
                                                                                                            
                                                             
    is_pure = conf == 1.0

    if (
        current_depth >= max_depth                      
        or n_samples < min_samples_split                                  
        or n_features < min_features_split                                  
        or is_pure                                                  
    ):
        
        

                                            
        if "LeafNode" in globals():
            leaf = LeafNode(
                is_classifier=is_classification,
                value=label,
                conf=conf,
                samples=current_samples,                             
                features=current_features,                                          
                probabilities=probabilities                   
            )
        else:
            warn("LeafNode class not available. Returning placeholder.")
            leaf = None               
        return leaf, current_depth

                                              

                                                 
    feature_splits = get_best_splits(
        X, y, criterion=criterion_func
    )                               
    f_best_axis_parallel = np.argmin(feature_splits[:, 1])
    best_split_score = feature_splits[f_best_axis_parallel, 1]
    threshold = feature_splits[f_best_axis_parallel, 0]

                                                                               
                                                                        
    w = np.zeros(n_features)
    w[f_best_axis_parallel] = 1.0
    b = -threshold

    stagnant = 0                                        

                               
    for k in range(num_tries):
        m = np.random.randint(0, n_features)                                            

                                       
        margin = X @ w + b                              

                                                                                 
                                                                                            
        X_m = X[:, m]                    
        u = np.full_like(X_m, np.nan, dtype=float)                         

                                                                               
        valid_mask = np.abs(X_m) > epsilon

                                            
                                                                                        
        np.divide(w[m] * X_m - margin, X_m, out=u, where=valid_mask)
                                                                                                                             
                                                                                                 

                                                                             
        u_valid = u[
            valid_mask & ~np.isnan(u)
        ]                                                           

        if len(u_valid) < 2:                                                       
            continue                                                                 

                                                          
                                                       
        possible_wm = np.convolve(np.sort(u_valid), [0.5, 0.5])[1:-1]

        if len(possible_wm) == 0:                                                     
            continue

        scores = np.empty_like(possible_wm)
        best_wm_in_loop, best_wm_score_in_loop = (
            w[m],
            np.inf,
        )                                          

                                                           
        wNew = np.array(w)                                     
        for i, wm_candidate in enumerate(possible_wm):
            wNew[m] = wm_candidate                             
            margin_new = X @ wNew + b                                    
            left_indices, right_indices = (margin_new <= 0), (margin_new > 0)

                                                            
            if (
                np.count_nonzero(left_indices) == 0
                or np.count_nonzero(right_indices) == 0
            ):
                wm_score = np.inf                                       
            else:
                                                                        
                wm_score = criterion_func(y[left_indices], y[right_indices])

            scores[i] = wm_score

            if wm_score < best_wm_score_in_loop:
                best_wm_score_in_loop = wm_score
                best_wm_in_loop = wm_candidate

                                                                         
        if best_wm_score_in_loop < best_split_score:
            best_split_score = best_wm_score_in_loop
            w[m] = best_wm_in_loop
            stagnant = 0                            
        elif (
            np.abs(best_wm_score_in_loop - best_split_score) < epsilon
        ):                                    
                                                                 
            if np.random.rand() <= np.exp(-stagnant):
                                                                                                                    
                w[m] = best_wm_in_loop
            stagnant += 1                                                                                                      

                                                         
        if best_split_score < epsilon:
            break
                                   

                                                                
                                                           
    if np.any(np.isinf(w)) or np.isinf(b):
        warn(
            "Infinity encountered in weights or bias. Replacing with heuristic values."
        )
        w[np.isinf(w) & (w > 0)] = 1.0                       
        w[np.isinf(w) & (w < 0)] = -1.0                        
                                      
        w[np.isnan(w)] = 0.0                      
        if np.isinf(b):
            b = 10.0 if b > 0 else -10.0
        if np.isnan(b):
            b = 0.0                           

                                                         
    margin = X @ w + b
    left_mask, right_mask = margin <= 0, margin > 0
    n_left, n_right = np.count_nonzero(left_mask), np.count_nonzero(right_mask)

                                             
                                                                          
    split_violates_min_leaf = n_left < min_samples_leaf or n_right < min_samples_leaf
                           

                                                                
                                                              
                                                                      
                                
    if n_left == 0 or n_right == 0 or split_violates_min_leaf:
                                        

                                                                                  
        if split_violates_min_leaf and not (n_left == 0 or n_right == 0):
            warn(
                f"Split invalidated by min_samples_leaf: "
                f"left_count={n_left}, right_count={n_right}, "
                f"min_leaf={min_samples_leaf} at depth {current_depth}. Making a leaf node."
            )
        elif n_left == 0 or n_right == 0:
            warn(
                f"Trivial split found (counts: {n_left}, {n_right}) at depth {current_depth}. Making a leaf node."
            )
                                  

        if "LeafNode" in globals():
            leaf = LeafNode(
                is_classifier=is_classification,
                value=label,                                               
                conf=conf,
                samples=current_samples,
                features=current_features,
                probabilities=probabilities,                    
            )
        else:
            warn("LeafNode class not available. Returning placeholder.")
            leaf = None
        return leaf, current_depth                                           

    else:                                                       
        if "Node" in globals():
            decision_node = Node(
                w,
                b,
                is_classifier=is_classification,
                value=label,                                                         
                conf=conf,                                       
                samples=current_samples,                                         
                features=current_features,                                      
            )
        else:
            warn("Node class not available. Cannot create decision node.")
                                                                          
            if "LeafNode" in globals():
                leaf = LeafNode(
                    is_classifier=is_classification,
                    value=label,
                    conf=conf,
                    samples=current_samples,
                    features=current_features,
                )
            else:
                leaf = None
            return leaf, current_depth

                                  
        left_node, left_depth = build_oblique_tree_oc1(
            X[left_mask, :],
            y[left_mask],
            classes, 
            is_classification,
            criterion_func,
            max_depth,
            min_samples_split,
                                                
            min_samples_leaf,
                              
            min_features_split,
            current_depth=current_depth + 1,
            current_samples=(
                current_samples[left_mask] if current_samples is not None else None
            ),
            current_features=current_features,
            num_tries=num_tries,
        )
        decision_node.add_left_child(left_node)

                                   
        right_node, right_depth = build_oblique_tree_oc1(
            X[right_mask, :],
            y[right_mask],
            classes,
            is_classification,
            criterion_func,
            max_depth,
            min_samples_split,
                                                
            min_samples_leaf,
                              
            min_features_split,
            current_depth=current_depth + 1,
            current_samples=(
                current_samples[right_mask] if current_samples is not None else None
            ),
            current_features=current_features,
            num_tries=num_tries,
        )
        decision_node.add_right_child(right_node)

        return decision_node, max(left_depth, right_depth)


                                                                   
def get_best_splits(X, y, criterion):                                               
    """Get best splits."""
    n_samples, n_features = X.shape
    all_splits = np.full(
        (n_features, 2), fill_value=np.inf
    )                                           

    for f in range(n_features):
        feature_values = X[:, f]
        unique_values = np.unique(
            feature_values
        )                                              

        if len(unique_values) <= 1:                                           
            continue

                                                                                
        feature_splits = np.convolve(np.sort(unique_values), [0.5, 0.5])[1:-1]

        best_split_for_feature = None
        best_score_for_feature = np.inf

                                                               
        for s in feature_splits:
            left_mask, right_mask = (feature_values <= s), (feature_values > s)

                                                                   
            if np.count_nonzero(left_mask) > 0 and np.count_nonzero(right_mask) > 0:
                score = criterion(
                    y[left_mask], y[right_mask]
                )                               
                if score < best_score_for_feature:
                    best_score_for_feature = score
                    best_split_for_feature = s

                                                     
        if best_split_for_feature is not None:
            all_splits[f, 0] = best_split_for_feature
            all_splits[f, 1] = best_score_for_feature
                                                                             

                                                           
    if np.all(np.isinf(all_splits[:, 1])):
                                                                            
                                                                                                     
                                                                        
                                                                                     
                                                                                                    
        pass                      

                                                                
                                                                                        
                                                                                 

    return all_splits
