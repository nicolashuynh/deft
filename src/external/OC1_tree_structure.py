 
 
                                                                                                                
 
 
 
                                         
 
 
import numpy as np
 
 
  
                                                 
 
 
class Tree:
    def __init__(self, n_features, is_classifier):                                         
        """Initialize the instance."""
        self.n_features = n_features                                                                                
        self.is_classifier = is_classifier                                             
        self.root_node = None                                                                     
        self.depth = -1                                                 
        self.num_leaf_nodes = -1                                                                 

    def set_root_node(self, root_node):
        """Set root node."""
        self.root_node = root_node

    def set_depth(self, depth):
        """Set depth."""
        self.depth = depth

    def get_depth(self):
        """Get depth."""
        if self.depth == -1:
            NotImplementedError('TODO: depth first traversal')
        return self.depth

    def predict(self, X):
        """Predict values."""
        return self.root_node.predict(X)


class Node:
    def __init__(self, w, b, is_classifier=True, value=None, conf=0.0, samples=None, features=None):
        """Initialize the instance."""
        self.w = w                           
        self.b = b                        
        self.value = value                                                           
        self.conf = conf                                                         
        self.samples = samples                                      
        self.features = features                                
        self.is_classifier = is_classifier
        self.left_child = None
        self.right_child = None
        self.is_fitted = False                                                 

    def add_left_child(self, child):
        """Handle add left child."""
        self.left_child = child

    def add_right_child(self, child):
        """Handle add right child."""
        self.right_child = child

    def is_leaf(self):
        """Check whether leaf."""
        return (self.left_child is None) and (self.right_child is None)

    def predict(self, X):

                                               
        """Predict values."""
        y = (np.dot(X, self.w) + self.b).squeeze()
        left, right = (y <= 0), (y > 0)
        y[left] = self.left_child.predict(X[left, :])
        y[right] = self.right_child.predict(X[right, :])

        return y
    
    def get_proba(self, x):
        """Get proba."""
        margin = self.w @ x + self.b
        if margin <= 0:
            return self.left_child.get_proba(x)
        else:
            return self.right_child.get_proba(x)

    


class LeafNode(Node):
    def __init__(self, is_classifier=True, value=None, conf=0.0, samples=None, features=None, probabilities=None):
        
        """Initialize the instance."""
        super(LeafNode, self).__init__(w=None, b=None, is_classifier=is_classifier,
                                       value=value, conf=conf, samples=samples, features=features)
        self.probabilities = probabilities

    def predict(self, X):

                                      
        """Predict values."""
        return np.full((X.shape[0], ), self.value)

    def get_proba(self, x):
        """Get proba."""
        return self.probabilities

