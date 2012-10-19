from __future__ import division
from math import log
from itertools import izip, imap
from operator import mul

f = open('monks-1-train.csv')
x = [map(int,line.strip('\r\n').split(',')) for line in f]

f1 = open('monks-1-test.csv')
x1 = [map(int,line.strip('\r\n').split(',')) for line in f1]

class Node:
    """
    The tree is constructed by creating a node which classifies the data given
    to choose the feature which generates the greatest information gain.  It
    then recursively calls a function on each of its children to determine which
    feature to best classify them upon.  It continues to do so untill a leaf is 
    found which is fully classified or the maximum depth of the tree we wish to 
    grow has been reached.  It determines the greatest gain by calculating the 
    entropy of the parent node and subtracting from it the sum of the classes
    probability and their entropy.
    """
    def __init__(self, data, is_leaf = False, depth = 2):
        self.data = data    
        self.is_leaf = is_leaf
        self.depth = depth
        self.children = []
        self.feature_set = [[0,1], [1,2,3], [1,2,3], [1,2], [1,2,3], [1,2,3,4], [1,2]]
        self.learning_rate = None
        self.decide_on = None
        self.pass_ratio = None

    def testing_error(self,testing = None):
        if not testing:
            data = self.data
        else:
            data = testing
        passed = 0
        tested = 0
        for monk in data:
            test_node = self.children[monk[self.decide_on] - 1]
            while True:
                passed +=  (test_node.pass_ratio[0] > .5 and monk[0] == 1) or \
                               (test_node.pass_ratio[1] > .5 and monk[0] == 0)
                tested += 1
                if not test_node.is_leaf:
                    test_node = test_node.children[monk[test_node.decide_on] - 1]
                else:
                    break
            
        print tested
        print passed
        print passed/tested 

    """
    Given the index of a feature split the data and return a list
    of lists in which each element in the list contains all the 
    data sets which share that feature
    """
    def classify_by_feature(self, feature):
        return [[x for x in self.data if x[feature] == feature_value] 
                    for feature_value in self.feature_set[feature]]

    """
    Given a set of data return a tuple of the ratio of the data
    which passes and fails the test divided by the length of the data set
    """
    def pass_fail_ratio(self, data_set):
        if not data_set:
            return (0,0)
        data_passes_rule = sum([1 for data in data_set if data[0] == 1])
        data_fails_rule = len(data_set) - data_passes_rule
        return (data_passes_rule / len(data_set), data_fails_rule / len(data_set))

    """
    Choose the root feature to split on and then recursively create child nodes
    for each decision which subsequently does the same untill we reach our maximum
    depth or find a feature which is fully classified based up it's parents decision.
    """        
    def choose_root_feature(self):
        child_classes = None
        max_entropy = 0
        learning_ratio = []
        """
        These lambda functions are used for the calculations of entropy for each node
        and it's weight for 
        """
        entropy = lambda (x, y): (0 not in [x,y]) and -x * log(x,2) * x - y * log(y,2) * y
        prob = lambda f_class: len(f_class) / len(self.data)

        """
        The learning rate and pass ratio of the parent are initially calculated.
        The parent's entropy will later be used when calculating the total entropy
        based on making a given decision while the pass fail ratio was just for my
        own observations.
        """
        parent_entropy = entropy(self.pass_fail_ratio(self.data))
        self.pass_ratio = self.pass_fail_ratio(self.data)
        for feature in xrange(1,7):
            # split the data based on a feature
            classes = self.classify_by_feature(feature)
            class_ratios = map(self.pass_fail_ratio, classes)
            classes_entropy = map(entropy, imap(self.pass_fail_ratio, classes))
            tree_entropy = parent_entropy - sum(imap(mul, imap(entropy, class_ratios), imap(prob, classes))) 
            # if this feature gains the most information choose it
            if tree_entropy > max_entropy:
                max_entropy = tree_entropy
                child_classes = classes
                learning_ratio = class_ratios
                self.decide_on = feature
        # if no decision has been made we have reached a leaf 
        if not self.decide_on:
            self.is_leaf = True
            return
        self.learning_rate = parent_entropy
        for child_data, l_ratio in izip(child_classes, learning_ratio):
            """
            if this child is not fully decided and we have not reached
            the bottom of the tree we wish to grow continue choosing features
            """
            if set(l_ratio) not in set([0,1]):
                self.children.append(Node(child_data, depth = self.depth - 1))
            else:
                self.children.append(Node(child_data, is_leaf = True))
        for child in self.children:
            if not child.is_leaf and child.depth > 0:
                # if there are children to choose based on this decision do so
                child.choose_root_feature()
            else:
                # if its a leaf or maximum depth node determine its pass fail ratio
                # entropy
                child.learning_rate = entropy(child.pass_fail_ratio(child.data))
                child.pass_ratio = child.pass_fail_ratio(child.data)
                child.is_leaf = True
            
test = Node(x,depth = 3)
test.choose_root_feature()
print "decision made on %s"%(test.decide_on)
print "pass: %s\nfail: %s"%(test.pass_ratio)
print "learning rate %s"%(test.learning_rate)
print "Is_leaf %s"%(test.is_leaf)
for child in test.children:
    print "\tdecision made on %s"%(child.decide_on)
    print "\tpass: %s\n\tfail: %s"%(child.pass_ratio)
    print "\tlearning rate %s"%(child.learning_rate)
    print "\tIs_leaf %s"%(child.is_leaf)
    print
    for c in child.children:
        print "\t\tdecision made on %s"%(c.decide_on)
        print "\t\tpass: %s\n\t\tfail: %s"%(c.pass_ratio)
        print "\t\tlearning rate %s"%(c.learning_rate)
        print "\t\tIs_leaf %s"%(c.is_leaf)
        print
        for c2 in c.children:
            print "\t\t\tdecision made on %s"%(c2.decide_on)
            print "\t\t\tpass: %s\n\t\t\tfail: %s"%(c2.pass_ratio)
            print "\t\t\tlearning rate %s"%(c2.learning_rate)
            print "\t\t\tIs_leaf %s"%(c2.is_leaf)
            print
