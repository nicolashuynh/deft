                                
                                                                                                  

import collections             
from warnings import warn

import numpy as np
import src.external.split_criteria as split_criteria
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.utils import check_random_state
from src.external.OC1_tree_structure import LeafNode, Node, Tree

                                                                      
epsilon = 1e-6


                                                                                                
def _get_node_label_conf(y_node, is_classification_flag, epsilon_val):
    """Get node label conf."""
    n_s = len(y_node)
    if n_s == 0:
        default_label = 0 if is_classification_flag else np.nan
        return default_label, 0.0

    if is_classification_flag:
        majority_res_mode = mode(y_node)
        if hasattr(majority_res_mode, "mode"):
            mode_val = majority_res_mode.mode
            count_val = majority_res_mode.count
        else:               
            mode_val, count_val = majority_res_mode

                                           
        node_label = (
            mode_val.item()
            if isinstance(mode_val, np.ndarray) and mode_val.ndim == 0
            else (
                mode_val[0]
                if isinstance(mode_val, np.ndarray) and mode_val.ndim > 0
                else mode_val
            )
        )

        node_count = (
            count_val.item()
            if isinstance(count_val, np.ndarray) and count_val.ndim == 0
            else (
                count_val[0]
                if isinstance(count_val, np.ndarray) and count_val.ndim > 0
                else count_val
            )
        )

        node_conf = node_count / n_s
    else:              
        node_label = np.mean(y_node)
        std_val = np.std(y_node)
        if std_val > epsilon_val:
            node_conf = np.sum(np.abs(y_node - node_label) <= std_val) / n_s
        else:
            node_conf = 1.0 if n_s > 0 else 0.0
    return node_label, node_conf


                                                                               
def get_best_splits(X, y, criterion):
    """Get best splits."""
    n_samples_node, n_features_node = X.shape
    all_splits_found = np.full((n_features_node, 2), fill_value=np.inf)

    for f_idx in range(n_features_node):
        feature_values_col = X[:, f_idx]
        unique_vals = np.unique(feature_values_col)

        if len(unique_vals) <= 1:
            continue

        candidate_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        best_thresh_for_feature = None
        best_score_for_feature = np.inf

        for s_val in candidate_thresholds:
            left_mask_split, right_mask_split = (feature_values_col <= s_val), (
                feature_values_col > s_val
            )

            if (
                np.count_nonzero(left_mask_split) > 0
                and np.count_nonzero(right_mask_split) > 0
            ):
                current_score = criterion(y[left_mask_split], y[right_mask_split])
                if current_score < best_score_for_feature:
                    best_score_for_feature = current_score
                    best_thresh_for_feature = s_val

        if best_thresh_for_feature is not None:
            all_splits_found[f_idx, 0] = best_thresh_for_feature
            all_splits_found[f_idx, 1] = best_score_for_feature

    return all_splits_found


class BaseObliqueTreeBFS(BaseEstimator):
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
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_features_split = min_features_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.num_tries = num_tries
        self.random_state = random_state

    def get_min_samples_leaf(self, n_samples_total):                        
        """Get min samples leaf."""
        min_s_leaf = 1
        if isinstance(self.min_samples_leaf, int):
            min_s_leaf = max(1, self.min_samples_leaf)
        elif isinstance(self.min_samples_leaf, float):
            min_s_leaf = max(1, int(np.ceil(self.min_samples_leaf * n_samples_total)))
        else:
            warn("Invalid type for min_samples_leaf; setting to default value of 1.")
            min_s_leaf = 1
        return min_s_leaf

    def get_min_samples_split(self, n_samples_total):                        
        """Get min samples split."""
        min_s_split = 2
        if isinstance(self.min_samples_split, int):
            min_s_split = max(2, self.min_samples_split)
        elif isinstance(self.min_samples_split, float):
            min_s_split = max(2, int(np.ceil(self.min_samples_split * n_samples_total)))
        else:
            warn("Invalid type for min_samples_split; setting to default value of 2.")
            min_s_split = 2
        return min_s_split

    def get_min_features_split(self, n_features_total):                        
        """Get min features split."""
        min_f_split = 1
        if isinstance(self.min_features_split, int):
            min_f_split = max(1, self.min_features_split)
        elif isinstance(self.min_features_split, float):
            min_f_split = max(
                1, int(np.ceil(self.min_features_split * n_features_total))
            )
        else:
            warn("Invalid type for min_features_split; setting to default value of 1.")
            min_f_split = 1
        return min_f_split

    def fit(self, X, y):
        """Fit values."""
        X_train = np.array(X, dtype=float)
        y_train = np.array(y)

        if X_train.ndim == 1:
            X_train = (
                X_train.reshape(-1, 1) if len(y_train) > 1 else X_train.reshape(1, -1)
            )
        if X_train.shape[0] != len(y_train):
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {X_train.shape[0]} != {len(y_train)}"
            )

        self.rng_ = check_random_state(self.random_state)                  

        if isinstance(self.criterion, str):
            if self.criterion == "gini":
                self._criterion_func = split_criteria.gini
            elif self.criterion == "twoing":
                self._criterion_func = split_criteria.twoing
            else:
                raise ValueError("Unrecognized string split criterion.")
        elif callable(self.criterion):
            self._criterion_func = self.criterion
        else:
            raise TypeError(
                "Criterion must be 'gini', 'twoing', or a callable function."
            )

        n_s_total, n_f_total = X_train.shape
        min_s_split_abs = self.get_min_samples_split(n_s_total)
        min_s_leaf_abs = self.get_min_samples_leaf(n_s_total)
        min_f_split_abs = self.get_min_features_split(n_f_total)

        is_clf_flag = is_classifier(self)

        self.root_node_, self.learned_depth_ = build_oblique_tree_bfs_oc1(
            X_train,
            y_train,
            is_clf_flag,
            self._criterion_func,
            self.max_depth,
            min_s_split_abs,
            min_s_leaf_abs,
            min_f_split_abs,
            self.num_tries,
            self.rng_,                         
        )

        if (
            "Tree" in globals()
        ):                                                           
            self.tree_ = Tree(n_features=n_f_total, is_classifier=is_clf_flag)
            self.tree_.set_root_node(self.root_node_)
            self.tree_.set_depth(self.learned_depth_)
        else:
            self.tree_ = None                                                
            warn("Tree class not fully available. Tree object might be incomplete.")

        return self

    def predict(self, X):
        """Predict values."""
        if self.tree_ is None and self.root_node_ is None:
            raise RuntimeError("The tree has not been fitted yet.")

        root_to_use = self.root_node_ if self.tree_ is None else self.tree_.root_node
        if root_to_use is None:                                           
            raise RuntimeError("Tree fitting failed or resulted in no root node.")

        X_pred_arr = np.array(X, dtype=float)
        if X_pred_arr.ndim == 1:
            X_pred_arr = X_pred_arr.reshape(1, -1)

                                                                                   
        y_predictions = root_to_use.predict(X_pred_arr)
        return np.array(
            y_predictions, dtype=int
        )                                            

    def get_params(self, deep=True):
        """Get params."""
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_features_split": self.min_features_split,
            "num_tries": self.num_tries,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        """Set params."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class ObliqueClassifier1BFS(ClassifierMixin, BaseObliqueTreeBFS):
    def __init__(
        self,
        criterion="gini",
        max_depth=3,
        min_samples_leaf=1,
        min_samples_split=2,
        min_features_split=1,
        num_tries=5,
        random_state=None,
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


def build_oblique_tree_bfs_oc1(
    X_orig_data,
    y_orig_data,
    is_classification_tree,
    criterion_function,
    max_depth_limit,
    min_samples_split_val,
    min_samples_leaf_val,
    min_features_split_val,
    num_perturb_tries,
    rng_instance,                             
):
    """Build oblique tree bfs oc1."""
    n_total_s, n_total_f = X_orig_data.shape
    initial_sample_indices = np.arange(n_total_s)

    queue = collections.deque()

    if n_total_s > 0:
        queue.append(
            {
                "data_indices": initial_sample_indices,
                "depth": 0,
                "parent_obj": None,
                "attach_point": "root",
            }
        )

    final_root_node = None
    actual_learned_depth = 0

    while queue:
        current_task = queue.popleft()
        node_data_indices = current_task["data_indices"]
        node_depth = current_task["depth"]
        parent_node_to_link = current_task["parent_obj"]
        attachment_method = current_task["attach_point"]

        actual_learned_depth = max(actual_learned_depth, node_depth)

        X_current_node = X_orig_data[node_data_indices]
        y_current_node = y_orig_data[node_data_indices]

        n_s_curr_node, n_f_curr_node = (
            X_current_node.shape
        )                                     

        node_label_val, node_conf_val = _get_node_label_conf(
            y_current_node, is_classification_tree, epsilon
        )

        is_node_pure = node_conf_val == 1.0

                                                 
        if (
            node_depth >= max_depth_limit
            or n_s_curr_node < min_samples_split_val
            or n_f_curr_node
            < min_features_split_val                                               
            or is_node_pure
        ):

            leaf_node_created = LeafNode(
                is_classifier=is_classification_tree,
                value=node_label_val,
                conf=node_conf_val,
                samples=node_data_indices,
                features=np.arange(n_total_f),
            )
            if attachment_method == "root":
                final_root_node = leaf_node_created
            elif attachment_method == "left":
                parent_node_to_link.add_left_child(leaf_node_created)
            elif attachment_method == "right":
                parent_node_to_link.add_right_child(leaf_node_created)
            continue

                                
        axis_parallel_splits = get_best_splits(
            X_current_node, y_current_node, criterion=criterion_function
        )

        if np.all(np.isinf(axis_parallel_splits[:, 1])):                                
            leaf_node_created = LeafNode(
                is_classifier=is_classification_tree,
                value=node_label_val,
                conf=node_conf_val,
                samples=node_data_indices,
                features=np.arange(n_total_f),
            )
            if attachment_method == "root":
                final_root_node = leaf_node_created
            elif attachment_method == "left":
                parent_node_to_link.add_left_child(leaf_node_created)
            elif attachment_method == "right":
                parent_node_to_link.add_right_child(leaf_node_created)
            continue

        best_ap_feature_idx = np.argmin(axis_parallel_splits[:, 1])
        current_best_split_score = axis_parallel_splits[best_ap_feature_idx, 1]
        ap_threshold = axis_parallel_splits[best_ap_feature_idx, 0]

                                                             
        w_hplane = np.zeros(n_f_curr_node)
        w_hplane[best_ap_feature_idx] = 1.0
        b_hplane = -ap_threshold

        stagnation_count = 0

        for _ in range(num_perturb_tries):
            pert_feature_idx = rng_instance.randint(0, n_f_curr_node)
            node_margin_vals = X_current_node @ w_hplane + b_hplane

            X_m_feature_col = X_current_node[:, pert_feature_idx]
            u_proj_vals = np.full_like(X_m_feature_col, np.nan, dtype=float)
            div_valid_mask = np.abs(X_m_feature_col) > epsilon

                                        
            np.divide(
                w_hplane[pert_feature_idx] * X_m_feature_col - node_margin_vals,
                X_m_feature_col,
                out=u_proj_vals,
                where=div_valid_mask,
            )

            u_proj_valid = u_proj_vals[div_valid_mask & ~np.isnan(u_proj_vals)]

            if len(u_proj_valid) < 2:
                continue

                                                      
            sorted_u_proj = np.sort(
                np.unique(u_proj_valid)
            )                                               
            if len(sorted_u_proj) < 2:
                continue
            candidate_w_m_vals = (sorted_u_proj[:-1] + sorted_u_proj[1:]) / 2.0

            if len(candidate_w_m_vals) == 0:
                continue

            best_wm_candidate = w_hplane[pert_feature_idx]
            best_wm_score_loop = np.inf

            w_hplane_temp = np.array(w_hplane)                         
            for wm_val in candidate_w_m_vals:
                w_hplane_temp[pert_feature_idx] = wm_val
                margin_pert_temp = X_current_node @ w_hplane_temp + b_hplane
                left_idx_pert, right_idx_pert = (margin_pert_temp <= 0), (
                    margin_pert_temp > 0
                )

                if (
                    np.count_nonzero(left_idx_pert) == 0
                    or np.count_nonzero(right_idx_pert) == 0
                ):
                    score_pert_temp = np.inf
                else:
                    score_pert_temp = criterion_function(
                        y_current_node[left_idx_pert], y_current_node[right_idx_pert]
                    )

                if score_pert_temp < best_wm_score_loop:
                    best_wm_score_loop = score_pert_temp
                    best_wm_candidate = wm_val

            if best_wm_score_loop < current_best_split_score:
                current_best_split_score = best_wm_score_loop
                w_hplane[pert_feature_idx] = best_wm_candidate
                stagnation_count = 0
            elif np.abs(best_wm_score_loop - current_best_split_score) < epsilon:
                if rng_instance.rand() <= np.exp(-stagnation_count):
                    w_hplane[pert_feature_idx] = best_wm_candidate
                stagnation_count += 1

            if current_best_split_score < epsilon:
                break                         

                                         
        if (
            np.any(np.isinf(w_hplane))
            or np.isinf(b_hplane)
            or np.any(np.isnan(w_hplane))
            or np.isnan(b_hplane)
        ):
            warn(
                "Infinity/NaN encountered in weights or bias. Applying heuristic replacement.",
                UserWarning,
            )
            w_hplane[np.isinf(w_hplane) & (w_hplane > 0)] = 1.0
            w_hplane[np.isinf(w_hplane) & (w_hplane < 0)] = -1.0
            w_hplane[np.isnan(w_hplane)] = 0.0
            if np.isinf(b_hplane):
                b_hplane = 10.0 if b_hplane > 0 else -10.0
            if np.isnan(b_hplane):
                b_hplane = 0.0

                                                                
        final_node_margins = X_current_node @ w_hplane + b_hplane
        final_left_mask = final_node_margins <= 0
        final_right_mask = final_node_margins > 0

        n_left_child = np.count_nonzero(final_left_mask)
        n_right_child = np.count_nonzero(final_right_mask)

        violates_min_leaf = (
            n_left_child < min_samples_leaf_val or n_right_child < min_samples_leaf_val
        )

        if n_left_child == 0 or n_right_child == 0 or violates_min_leaf:
                                      
            leaf_node_created = LeafNode(
                is_classifier=is_classification_tree,
                value=node_label_val,
                conf=node_conf_val,
                samples=node_data_indices,
                features=np.arange(n_total_f),
            )
            if attachment_method == "root":
                final_root_node = leaf_node_created
            elif attachment_method == "left":
                parent_node_to_link.add_left_child(leaf_node_created)
            elif attachment_method == "right":
                parent_node_to_link.add_right_child(leaf_node_created)
            continue

                              
        decision_node_created = Node(
            w_hplane,
            b_hplane,
            is_classifier=is_classification_tree,
            value=node_label_val,
            conf=node_conf_val,                                      
            samples=node_data_indices,
            features=np.arange(n_total_f),
        )

        if attachment_method == "root":
            final_root_node = decision_node_created
        elif attachment_method == "left":
            parent_node_to_link.add_left_child(decision_node_created)
        elif attachment_method == "right":
            parent_node_to_link.add_right_child(decision_node_created)

                                
        left_child_orig_indices = node_data_indices[final_left_mask]
        right_child_orig_indices = node_data_indices[final_right_mask]

        queue.append(
            {
                "data_indices": left_child_orig_indices,
                "depth": node_depth + 1,
                "parent_obj": decision_node_created,
                "attach_point": "left",
            }
        )
        queue.append(
            {
                "data_indices": right_child_orig_indices,
                "depth": node_depth + 1,
                "parent_obj": decision_node_created,
                "attach_point": "right",
            }
        )

                                                                                  
    if final_root_node is None:
                                                                                    
                                                                         
        default_label, default_conf = _get_node_label_conf(
            np.array([]), is_classification_tree, epsilon
        )
        final_root_node = LeafNode(
            is_classifier=is_classification_tree,
            value=default_label,
            conf=default_conf,
            samples=np.array([], dtype=int),
            features=np.arange(n_total_f),
        )
        warn(
            "Tree construction resulted in an empty tree, possibly due to no initial data. Created a default root leaf.",
            UserWarning,
        )
        actual_learned_depth = 0                                                

    return final_root_node, actual_learned_depth
