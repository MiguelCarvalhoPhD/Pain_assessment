
#--------------------
#importing

from ..base import OverSamplingSimplex
from ..base import coalesce, coalesce_dict
from ..base._simplexsampling import *
from sklearn.metrics import fbeta_score, make_scorer, classification_report, precision_score, recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.linalg import circulant
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pygad
from functools import partial
import scipy



__all__ = ["Genetic_BETA_resampling_framework","Genetic_BETA_resampling_framework_v2","pmf_local_neighbourhood","pmf_prowsyn",
           "generate_samples_local_neighboorhood","generate_samples_prowsyn"]



#---------------------------------------
# framework

"""

PMF functions guidelines:

    - input: obligatory: instance, X, y. not obligatory: other relavant variables
    - output: group number (from 0 to num_groups) for each sample in X_min, other relevant variables, where each row is X_min.

"""    

def pmf_local_neighbourhood(instance,X,y,to_remove,n_neighbors=5):

    """
        compute NN counts for sample and group them based on said information
    
    """

    if to_remove:
        min_label = instance.maj_label
        maj_label = instance.min_label
    else:
        min_label = instance.min_label
        maj_label = instance.maj_label

    #minority class 
    X_min = X[y==min_label]

    #-------------------------------
    # finding NN counts
    n_neighbors = min([X_min.shape[0], n_neighbors])

    knn = NearestNeighbors(n_neighbors=n_neighbors,n_jobs=-1) # Find the single closest neighbor
    knn.fit(X)  # Train the model on data

    _, indexes = knn.kneighbors(X_min,n_neighbors)

    #--------------------------------------------
    #finding sample typology
    NN_counts = np.sum(y[indexes[:,1:n_neighbors]] == maj_label,axis=1).astype(int)

    #--------------------------------------------
    #finding minority NN of minority samples

    knn.fit(X_min)
    _,indexes = knn.kneighbors(X_min,n_neighbors)
    indexes = indexes[:,1:]

    #print(f"Levels NN: {np.bincount(NN_counts.astype(int))}")


    return NN_counts, [indexes]

def generate_samples_local_neighboorhood(instance,X,y,pmf,other_vars,sample_group,n_to_sample):

    """
    
    Generate the desired amount of samples on a certain instance hardness
    
    """

    #------------------------------------------
    #generate samples


    #determine relevant samples adn generate sampling probability
    candidate_samples_condition = pmf==sample_group

    #---------------------------------
    #sample generation
    
    #assign equal probability to all samples
    r = np.zeros((len(candidate_samples_condition),))
    r[candidate_samples_condition] = 1
    r = r / np.sum(r)

    #define dim
    n_dim = instance.n_dim
    n_dim = np.min([n_dim, other_vars[0].shape[1]])

    #sample generation
    simplices = instance.simplices(X=X[y==instance.min_label],indices=other_vars[0],base_weights=r,n_to_sample=n_to_sample,n_dim=n_dim)
    samples = random_samples_from_simplices(X=X[y==instance.min_label], simplices=simplices, random_state=instance.random_state,vertex_weights=None,X_vertices=None)     

    #print(f"Pmf: {pmf} | simplices: {simplices}")
    #sort via typology of last simplice (first is always the same)
    sort_idx = np.argsort(pmf[simplices[:,1]])

    if instance.beta < 1:
        samples = samples[sort_idx]

        #----------------------------------------
        #save generate samples 

    #print(f'Generated {samples.shape[0]} samples with local')

    return samples    

def pmf_prowsyn(instance,X,y,to_remove,n_levels=4,n_neighbors=5):

    """
    Determine the proximity level for each minority sample directly.

    The function returns an array of shape (n_minority_samples,)
    where each element is the proximity level (an integer) for that minority sample.

    Args:
        X (np.array): All training vectors.
        y (np.array): All target labels.
        fold (int): Fold index used for internal tracking.

    Returns:
        np.array: An array with the proximity level for each minority sample.
    """
    # check whether to remove
    if to_remove:
        min_label = instance.maj_label
        maj_label = instance.min_label
    else:
        min_label = instance.min_label
        maj_label = instance.maj_label

    # Get global indices of minority samples and store for later use
    minority_indices = np.where(y == min_label)[0]
    majority_indices = np.where(y == maj_label)[0]

    # Extract features of minority samples (using self.X for consistency) and majority samples
    X_min = X[minority_indices]
    X_maj = X[majority_indices]
    
    n_min = len(minority_indices)
    # Output array: one element per minority sample (indexed 0 .. n_min-1)
    levels_arr = np.zeros((n_min,))
    
    # Use relative indices for the minority samples (0,1,...,n_min-1)
    P = np.arange(n_min)
    sorted_list = []

    for lvl in range(n_levels):

        if len(P) == 0:
            break
        
        #define number of neighbors
        n_neighbors = np.min([len(P),5])

        # Choose number of neighbors based on available samples and class stats
        nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        nn.fit(X_min[P])
        inds = nn.kneighbors(X_maj, return_distance=False)

        # Count how many times each candidate in P was selected
        unique, counts = np.unique(inds.flatten(), return_counts=True)
        selected = P[unique]

        #print(f"Number of sekected samples at {lvl} is {len(selected)}")

        #save order of counts for outlier removal 

        sorted_counts  = np.argsort(counts*(-1))
        sorted_indexes = unique[sorted_counts]

        sorted_list.append(sorted_counts)

        # array_scores = np.linspace(lvl+0.1,lvl+0.9,len(unique))[sorted_counts]

        # Directly assign the current level to these minority samples
        levels_arr[selected] = lvl

        # Remove selected samples from P
        P = np.setdiff1d(P, selected, assume_unique=True)

    # Any remaining minority samples get the next level
    if len(P) > 0:
        levels_arr[P] = lvl + 1
        sorted_list.append(np.arange(len(P)))

    if lvl < 3:
        num_groups = np.min([int(n_min/3),5])

        for k in range(num_groups):
            idx = np.arange(k,n_min,num_groups)
            levels_arr[idx] = k


    #discretize into 4 sets if needed
    # if lvl <= 3:

    #     #generate sample groups
    #     n_bins = 4
    #     quantiles = np.linspace(0,1,n_bins+1)
    #     bin_edges = np.quantile(levels_arr,quantiles)
    #     levels_arr = np.digitize(levels_arr,bin_edges,right=True)
    #     levels_arr[levels_arr==0] = 1

    #print(f"Levels prowsyn: {np.bincount(levels_arr.astype(int))}")

    return levels_arr, sorted_list#[sorted_indexes_1]

def generate_samples_prowsyn(instance,X,y,pmf_1,pmf_2,sample_group,n_to_sample):

    """
    
        generate samples via prowsyn
    
    """

    #determine relevant samples adn generate sampling probability
    candidate_samples_condition = pmf_2==sample_group
    cluster_vectors = X[y==instance.min_label][candidate_samples_condition]

    #print(f"Cluster vector shape: {cluster_vectors.shape} of {sample_group}")


    #define dim
    n_dim = instance.n_dim

    #create samples
    simplices = instance.simplices(X=cluster_vectors,indices=circulant(np.arange(len(cluster_vectors))),n_to_sample=n_to_sample,n_dim=n_dim) 
    samples   = random_samples_from_simplices(cluster_vectors,simplices=simplices,random_state=instance.random_state)

    # #sorting
    # sorted_indexes = np.argsort(np.mean(pmf_1[candidate_samples_condition][simplices],axis=1))
    # samples = samples[sorted_indexes]


    return samples

class Genetic_BETA_resampling_framework_v2(OverSamplingSimplex):

    def __init__(
        self,
        proportion=1.0,
        *,
        nn_params=None,
        ss_params=None,
        random_state=None,
        func1_pmf=None,
        func2_pmf=None,
        func1_generate_samples=None,
        func2_generate_samples=None,
        func1_toremove=False,
        func2_toremove=False,
        outlier_removal=True,
        Beta=1,
        n_splits=5,
        **_kwargs
    ):
        """
        Constructor of the SMOTE object

        Args:
            arg 1:  ...
            
            arg 2: ...
            
        """

        ss_params_default = {
            "n_dim": 2,
            "simplex_sampling": "random",
            "within_simplex_sampling": "random",
            "gaussian_component": None,
        }
        ss_params = coalesce_dict(ss_params, ss_params_default)

        OverSamplingSimplex.__init__(self, **ss_params, random_state=random_state)

        #relevant checks
        self.check_greater_or_equal(proportion, "proportion", 0)

        #saving init variables
        self.proportion      = proportion
        self.nn_params       = coalesce(nn_params, {})
        self.beta            = Beta
        self.n_splits        = n_splits
        self.func1_toremove  = func1_toremove
        self.func2_toremove  = func2_toremove
        self.func1_pmf       = func1_pmf
        self.func2_pmf       = func2_pmf
        self.func1_generate  = func1_generate_samples
        self.func2_generate  = func2_generate_samples
        self.outlier_removal = outlier_removal

    def create_dictionary_saving(self,to_remove):

        """
        
            Create a datastructe to save sample groups
        
        """

        sample_groups = {}
        required_variables = {}
        samples = {}

        #iterate over folds
        for fold in range(-1,self.n_splits):

            sample_groups[fold] = {}
            required_variables[fold] = {}

            if not to_remove:
                samples[fold] = {}
            

        return sample_groups, required_variables, samples

    def create_saving_datastructure(self):

        """
        
        Create a data format to save generated samples during optimization

        """

        #-----------------------------
        # create datastructure for data saving

        self.pmf_1, self.vars_pmf_1, self.pmf_1_samples = self.create_dictionary_saving(self.func1_toremove)
        self.pmf_2, self.vars_pmf_2, self.pmf_2_samples = self.create_dictionary_saving(self.func2_toremove)


    def generate_pmfs(self,X,y,fold):

        """
        
            Determine sample groups

        
        """
        self.pmf_1[fold], self.vars_pmf_1[fold] = self.func1_pmf(self,X,y,self.func1_toremove)
        self.pmf_2[fold], self.vars_pmf_2[fold] = self.func2_pmf(self,X,y,self.func2_toremove)

    def check_number_genes(self):


        """
        
        Check number of genes required to optimized a certain dataset


        
        """

        uniques, counts = np.unique(self.pmf_1[-1].astype(int),return_counts=True)
        self.number_genes_pmf1, self.genes_pmf1_counts = uniques[counts>0], counts[counts>0]
        print(f"Genes 1: {self.number_genes_pmf1} ! counts gene 1: {self.genes_pmf1_counts}")

        uniques, counts = np.unique(self.pmf_2[-1].astype(int),return_counts=True)
        self.number_genes_pmf2, self.genes_pmf2_counts = uniques[counts>2], counts[counts>2]


    def generate_samples(self,X,y,n_to_sample,sample_group,pmf_number,fold):

        """
            Generate N samples from sample group 
        
        """

        if pmf_number == 1:


            self.pmf_1_samples[fold][sample_group] = self.func1_generate(self,X,y,self.pmf_1[fold],self.vars_pmf_1[fold],sample_group,n_to_sample)
        
        elif pmf_number == 2:

            self.pmf_2_samples[fold][sample_group] = self.func2_generate(self,X,y,self.pmf_1[fold],self.pmf_2[fold],sample_group,n_to_sample)

    
    def make_all_samples_fold(self,X,y,fold):

        """
        
            make all the required samples
        
        """
        total_number_samples = int(self.class_stats[self.maj_label]*self.proportion - self.class_stats[self.min_label])

        #generate for first pmf if required
        if not self.func1_toremove:

            unique, counts = np.unique(self.pmf_1[fold].astype(int),return_counts=True)

            for k,group in enumerate(unique):  

                    if counts[k] > 0:           

                        self.generate_samples(X,y,total_number_samples,group,1,fold)

        #generate for second pmf if required
        if not self.func2_toremove:

            unique, counts = np.unique(self.pmf_2[fold].astype(int),return_counts=True)

            for k,group in enumerate(unique):  

                    if counts[k] > 2:           

                        self.generate_samples(X,y,total_number_samples,group,2,fold)

    
    def sampling(self,X,y,distribution,fold):


        """

            Generate all the required synthetic data

        """
        #------------------------------------
        #get distribution values

        dist_pmf1  = distribution[:len(self.number_genes_pmf1)]
        dist_pmf2  = distribution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)]
        
        if self.outlier_removal:
            dist_ratio      = distribution[-2]
            percent_removal = distribution[-1]
        else:
            dist_ratio = np.abs(distribution[-1])

        ########################################################################
        # error handling for different groups in main vs original sample group

        samples_groups_in_fold_pmf1, counts_pmf1 = np.unique(self.pmf_1[fold].astype(int),return_counts=True)

        #print(f"PRIOR -- Dist: {dist_pmf1} | counts: {counts_pmf1} | available groups main: {self.number_genes_pmf1} | available groups: {samples_groups_in_fold_pmf1}")

        group_counts = dict(zip(samples_groups_in_fold_pmf1,counts_pmf1))
        indexes      = np.array([group_counts.get(group, 0) > 0 for group in self.number_genes_pmf1])
        dist_pmf1[~indexes] = 0

        #print(f"Indexes: {indexes}")
        #print(f"POST -- Dist: {dist_pmf1} | counts: {counts_pmf1} | available groups main: {self.number_genes_pmf1} | available groups: {samples_groups_in_fold_pmf1}")

        samples_groups_in_fold_pmf2, counts_pmf2 = np.unique(self.pmf_2[fold].astype(int),return_counts=True)
        
        group_counts = dict(zip(samples_groups_in_fold_pmf2,counts_pmf2))
        indexes      = np.array([group_counts.get(group, 0) > 2 for group in self.number_genes_pmf2])
        dist_pmf2[~indexes] = 0

        counts_pmf2 = np.array([group_counts.get(group,0) for group in self.number_genes_pmf2])
        
        #print(f"POST -- Dist: {dist_pmf2} | counts: {counts_pmf2} | available groups main: {self.number_genes_pmf2} | available groups: {samples_groups_in_fold_pmf2}")
        # print(f"Counts PMF_2: {counts_pmf2}")

        #error handling if there are no samples, return original data

        ##################################################################


        #check total number of required samples
        if self.func2_toremove:
            #total_main_dist   = np.sum(y == self.maj_label)*self.proportion*(1-dist_ratio) - np.sum(y == self.min_label)
            to_remove_samples = np.sum(y == self.maj_label)*dist_ratio

            #check whether it is possible to remove such samples, if not, adjust
            samples_to_remove_per_group = to_remove_samples*dist_pmf2
 
            condition = samples_to_remove_per_group > counts_pmf2
            samples_to_remove_per_group[condition] = counts_pmf2[condition]

            #update required amount of samples to make
            to_remove_samples = np.sum(samples_to_remove_per_group)
            total_main_dist   = (np.sum(y == self.maj_label) - to_remove_samples)*self.proportion - np.sum(y ==self.min_label)

        #in case both pmf are to generate samples
        else:
            total_number_samples = np.sum(y == self.maj_label)*self.proportion - np.sum(y == self.min_label)
            total_main_dist      = total_number_samples*dist_ratio
            total_secundary_dist = total_number_samples*(1-dist_ratio)

        #instanciate new syntetic dataset
        temp_data      = X[y==self.min_label]
        temp_y         = y[y==self.min_label]

        #-------------------------------
        # outlier removal procedure

        if self.outlier_removal:
            number_to_remove  = int(len(self.vars_pmf_2[fold][0])*percent_removal) 
            indexes_to_remove = self.vars_pmf_2[fold][0][-number_to_remove:]
            new_indexes       = np.setdiff1d(np.arange(len(temp_y)),indexes_to_remove)
            temp_y            = temp_y[new_indexes]
            temp_data         = temp_data[new_indexes]


            #forcing behaviour
            # if percent_removal > 0.4:
            #     dist_pmf1[-1] = 0
            #     dist_pmf2[0]  = 0

        ###########################

        if np.sum(dist_pmf2) == 0 or np.sum(dist_pmf1) == 0:
            return X, y 

        #
        #print(f"probabilities after outlier removal: {dist_pmf1} | {dist_pmf2}")

        #normalize distributions
        dist_pmf1 = np.abs(dist_pmf1) / np.sum(dist_pmf1)
        dist_pmf2 = np.abs(dist_pmf2) / np.sum(dist_pmf2)

        #-----------------------------------
        #main dist sample generation

        #print(f"total: {total_number_samples} | main: {total_main_dist} | secu: {total_secundary_dist} | manual: {np.bincount(y.astype(int))}")
        added_samples = 0

        #iterate over distribution and select/generate samples
        for k,main_dist_value in enumerate(dist_pmf1):

            #print(f"Fold: {fold} | Main dist: {dist_pmf1} | Main dist value at k: {main_dist_value} | {self.number_genes_pmf1[k]}")
            
            #avoid sampling if no samples exist
            if main_dist_value != 0:

                required_samples_main = np.abs(np.ceil(main_dist_value*total_main_dist).astype(np.int16))

                #print(f"Required samples main; {required_samples_main}")

                #added_samples += len(self.pmf_1_samples[fold][self.number_genes_pmf1[k]][:required_samples_main,:])
                temp_data             = np.vstack((temp_data,self.pmf_1_samples[fold][self.number_genes_pmf1[k]][:required_samples_main,:]))
                temp_y                = np.hstack((temp_y,np.ones((required_samples_main,))*self.min_label))
                
        #--------------------------------
        #secundary dist sample generation /deletion
        if self.func2_toremove:
            
            #create set to save indexes of samples to be remove
            total_sample_indexes = np.empty((0,))

            #get majority samples in original indexes
            converter            = np.where(y==self.maj_label)[0]

            for k, to_remove_group in enumerate(samples_to_remove_per_group):
                
                #get samples in indexes of sample group 
                indexes_selected = self.pmf_2[fold] == self.number_genes_pmf2[k]

                if np.sum(indexes_selected) == 0 or to_remove_group == 0:
                    continue

                #print(f"Samples to remove in {k} group: {to_remove_group}")


                #get indexes in original indexes

                try:
                    indexes_selected = converter[indexes_selected][self.vars_pmf_2[fold][k]][:int(to_remove_group)]

                    #save selected samples
                    total_sample_indexes  = np.hstack((total_sample_indexes,indexes_selected))
                except:
                    print(f"{converter[indexes_selected].shape} | {len([self.vars_pmf_2[fold][k]])} | {int(to_remove_group)}")



            #actually remove samples
            indexes_to_keep = np.setdiff1d(converter,total_sample_indexes)
            temp_data = np.vstack((temp_data,X[indexes_to_keep,:]))
            temp_y    = np.hstack((temp_y,np.ones((
                                        len(indexes_to_keep),
                                        ))*self.maj_label
                                    ))

        else:

            for k, secundary_dist_value in enumerate(dist_pmf2):  

                #print(f"Fold: {fold} | Main dist: {secundary_dist_value} | {self.number_genes_pmf2[k]}")

                if secundary_dist_value !=0:    

                    required_samples_sec = np.ceil(secundary_dist_value*total_secundary_dist).astype(np.int16)
                    #added_samples += len(self.pmf_2_samples[fold][self.number_genes_pmf2[k]][:required_samples_sec,:])
                    temp_data            = np.vstack((temp_data,self.pmf_2_samples[fold][self.number_genes_pmf2[k]][:required_samples_sec,:]))
                    temp_y               = np.hstack((temp_y,np.ones((required_samples_sec,))*self.min_label))
            
            #add samples majority class, meaning no oversampling
            temp_data = np.vstack((temp_data,X[y==self.maj_label]))
            temp_y    = np.hstack((temp_y,y[y==self.maj_label]))

        #print(f"Added {added_samples} samples")

        return temp_data, temp_y
    
    
    def fitness_function(self,ga_instance,solution,solution_idx):

        """
            fitness function compution given a certain distribution        
        """
        #print(f"Main dist: {solution[:len(self.number_genes_pmf1)]} | secundary sol: {solution[len(self.number_genes_pmf1):-2]} at {solution_idx}")

        model_performance = []
        model = RandomForestClassifier(n_estimators=10,random_state=1)
        #model = LinearSVC(random_state=1)
        #model = AdaBoostClassifier(n_estimators=10,random_state=1)
        #model = LGBMClassifier(random_state=1)
        #model = XGBClassifier()
        #model = LogisticRegression(random_state=1)
        #model = GaussianNB()
        #model = DecisionTreeClassifier(max_depth=5,random_state=1)


        for fold, (train_idx, test_idx) in enumerate(self.kf.split(self.X,self.y)):

            #print(f"{fold} - SOlution inside fold: {solution} | 2: {solution.copy()}")

            #make training/testing dataset
            train_data, test_data = self.X[train_idx], self.X[test_idx]
            train_y   , test_y    = self.y[train_idx], self.y[test_idx] 

            #print(f"Solution: {solution} | {train_data.shape} and {train_y.shape} | counts: {np.bincount(train_y.astype(int))}")

            synthetic_X, synthetic_y  = self.sampling(train_data,train_y,solution.copy(),fold)

            #print(f"Synthetic data shape: {synthetic_X.shape} and {synthetic_y.shape} | counts: {np.bincount(synthetic_y.astype(int))} ")

            #test model performance
            try:
                model.fit(synthetic_X,synthetic_y)
                model_performance.append(fbeta_score(test_y,model.predict(test_data),beta=self.beta))
            except:
                model_performance.append(0)

        return np.mean(model_performance)
        
    def make_final_samples(self,X,y,distribution):

        """
        
        Make final samples by sampling each possible fold
        
        """

        #------------------------------------
        #get distribution values

        dist_pmf1  = distribution[:len(self.number_genes_pmf1)]
        dist_pmf2  = distribution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)]
        
        if self.outlier_removal:
            dist_ratio      = distribution[-2]
            percent_removal = distribution[-1]
        else:
            dist_ratio = np.abs(distribution[-1])

        ########################################################################

        _, counts_pmf2 = np.unique(self.pmf_2[-1].astype(int),return_counts=True)
        counts_pmf2 = counts_pmf2[counts_pmf2 > 2]

        #error handling if there are no samples, return original data

        if np.sum(dist_pmf2) == 0 or np.sum(dist_pmf1) == 0:
            return X, y 

        ##################################################################        

        #check total number of required samples
        if self.func2_toremove:
            #total_main_dist   = np.sum(y == self.maj_label)*self.proportion*(1-dist_ratio) - np.sum(y == self.min_label)
            to_remove_samples = np.sum(y == self.maj_label)*dist_ratio

            #check whether it is possible to remove such samples, if not, adjust
            samples_to_remove_per_group = to_remove_samples*dist_pmf2
 
            condition = samples_to_remove_per_group > counts_pmf2
            samples_to_remove_per_group[condition] = counts_pmf2[condition]

            #update required amount of samples to make
            to_remove_samples = np.sum(samples_to_remove_per_group)
            total_main_dist   = (np.sum(y == self.maj_label) - to_remove_samples)*self.proportion - np.sum(y ==self.min_label)

        #in case both pmf are to generate samples
        else:
            total_number_samples = np.sum(y == self.maj_label)*self.proportion - np.sum(y == self.min_label)
            total_main_dist      = total_number_samples*dist_ratio
            total_secundary_dist = total_number_samples*(1-dist_ratio)

        #instanciate new syntetic dataset
        temp_data      = X[y==self.min_label]
        temp_y         = y[y==self.min_label]

        #-------------------------------
        # outlier removal procedure

        if self.outlier_removal:
            number_to_remove  = int(len(self.vars_pmf_2[-1][0])*percent_removal) 
            indexes_to_remove = self.vars_pmf_2[-1][0][-number_to_remove:]
            new_indexes       = np.setdiff1d(np.arange(len(temp_y)),indexes_to_remove)
            temp_y            = temp_y[new_indexes]
            temp_data         = temp_data[new_indexes]

            #forcing behaviour
            # if percent_removal > 0.4:
            #     dist_pmf1[-1] = 0
            #     dist_pmf2[0]  = 0

        ###########################

        #normalize distributions
        dist_pmf1 = np.abs(dist_pmf1) / np.sum(dist_pmf1)
        dist_pmf2 = np.abs(dist_pmf2) / np.sum(dist_pmf2)

        #-----------------------------------
        #main dist sample generation
        counts_per_fold = np.array([
            [len(self.pmf_1_samples[fold].get(sample_group, [])) for sample_group in self.number_genes_pmf1]
            for fold in range(self.n_splits)
        ])
        #iterate over distribution and select/generate samples
        for k,main_dist_value in enumerate(dist_pmf1):

            available_folds = np.where(counts_per_fold[:,k]>0)[0]
            
            #avoid sampling if no samples exist
            if main_dist_value != 0:

                required_samples_main = np.abs(np.ceil(main_dist_value*total_main_dist).astype(np.int16))

                for fold in available_folds:

                    idx = np.arange(fold,required_samples_main,len(available_folds))

                    temp_data             = np.vstack((temp_data,self.pmf_1_samples[fold][self.number_genes_pmf1[k]][idx,:]))
                    temp_y                = np.hstack((temp_y,np.ones((len(idx),))*self.min_label))
                
        #--------------------------------
        #secundary dist sample generation /deletion
        if self.func2_toremove:
            
            #create set to save indexes of samples to be remove
            total_sample_indexes = np.empty((0,))

            #get majority samples in original indexes
            converter            = np.where(y==self.maj_label)[0]

            for k, to_remove_group in enumerate(samples_to_remove_per_group):
                
                #get samples in indexes of sample group 
                indexes_selected = self.pmf_2[-1] == self.number_genes_pmf2[k]

                #print(f"Samples to remove in {k} group: {to_remove_group}")
                
                #if samples are not present in fold
                if np.sum(indexes_selected) == 0 or to_remove_group == 0:
                    continue

                #get indexes in original indexes
                try:
                    indexes_selected = converter[indexes_selected][self.vars_pmf_2[-1][k]][:int(to_remove_group)]

                    #save selected samples
                    total_sample_indexes  = np.hstack((total_sample_indexes,indexes_selected))
                except:
                    continue
                    #print(f"{converter[indexes_selected].shape} | {len([self.vars_pmf_2[fold][k]])} | {int(to_remove_group)}")


            #actually remove samples
            indexes_to_keep = np.setdiff1d(converter,total_sample_indexes)
            temp_data = np.vstack((temp_data,X[indexes_to_keep,:]))
            temp_y    = np.hstack((temp_y,np.ones((
                                        len(indexes_to_keep),
                                        ))*self.maj_label
                                    ))

        else:
            counts_per_fold = np.array([
                [len(self.pmf_2_samples[fold].get(sample_group, [])) for sample_group in self.number_genes_pmf2]
                for fold in range(self.n_splits)
            ])
            for k, secundary_dist_value in enumerate(dist_pmf2):    

                available_folds = np.where(counts_per_fold[:,k]>0)[0]

                if secundary_dist_value !=0:    

                    required_samples_sec = np.ceil(secundary_dist_value*total_secundary_dist).astype(np.int16)

                    for fold in available_folds:

                        idx = np.arange(fold,required_samples_sec,len(available_folds))

                        temp_data            = np.vstack((temp_data,self.pmf_2_samples[fold][self.number_genes_pmf2[k]][idx,:]))
                        temp_y               = np.hstack((temp_y,np.ones((len(idx),))*self.min_label))
            
            #add samples majority class, meaning no oversampling
            temp_data = np.vstack((temp_data,X[y==self.maj_label]))
            temp_y    = np.hstack((temp_y,y[y==self.maj_label]))

        return temp_data, temp_y
    

    def resample(self,X,y):

        """
            does the optimization
        
        """

        #----------------------
        # init vars

        self.X = X
        self.y = y
        self.class_label_statistics(self.y)
        self.create_saving_datastructure()
        self.kf    = StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=1)
        self.model = RandomForestClassifier(n_estimators=10,max_depth=5,random_state=1) 

        #---------------------
        # getting pmfs and synthetic samples for whole data

        self.generate_pmfs(X,y,-1)
        self.check_number_genes()
        self.make_all_samples_fold(X,y,-1)

        #---------------------
        # getting pmfs and synthetic data for each fold

        for fold, (train_idx, _) in enumerate(self.kf.split(self.X,self.y)):

            print(f"Conducting analysis on fold: {fold}")
            
            #divide into train and test split
            temp_data, temp_y = self.X[train_idx], self.y[train_idx]

            #(1)
            self.generate_pmfs(temp_data,temp_y,fold)
            #(2)
            self.make_all_samples_fold(temp_data,temp_y,fold)    

        #------------------------
        # init vars for optimization

        #define gene space

        #20 > 2000
        #10 > 1000

        gene_space = [np.linspace(0.01,0.5,20) for _ in range(len(self.number_genes_pmf1)+len(self.number_genes_pmf2))] #removal
        gene_space.append(np.linspace(0.01,0.99,20)) #dist ratio

        #increase gene space if outlier removal is needed
        if self.outlier_removal:
            gene_space.append(np.linspace(0,0.5,10)) #outlier removal

        #other vars
        num_genes = len(gene_space)
        fitness_function = self.fitness_function
        num_generations = 300
        sol_per_pop = 25
        num_parents_mating = int(sol_per_pop/2)
        parent_selection_type = "tournament"
        crossover_type = "single_point"
        mutation_type = "swap"
        keep_elistism = 4
        crossover_probability = 0.75
        mutation_probability = 0.25
        K_tornament = 4
        stop_criteria = ["reach_1"]

        #------------------------
        # run optimization

        ga_instance = pygad.GA(num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_function,
                sol_per_pop=sol_per_pop,
                num_genes=num_genes,
                K_tournament=K_tornament,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                parent_selection_type=parent_selection_type,
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                gene_space=gene_space,
                parallel_processing=16,
                keep_elitism=keep_elistism,
                stop_criteria=stop_criteria,
                save_solutions=True)
        
        ga_instance.run()

        #----------------------------------------
        # get results and generate final dataset

        #get solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        final_data, final_y = self.sampling(X,y,solution,-1)


        print(solution)
        print("*"*50)
        print(f"Initial counts: {np.bincount(y.astype(int))}")
        print(f"FInal data shape: {final_data.shape} and {final_y.shape} | counts: {np.bincount(final_y.astype(int))}")

        return [solution,self.number_genes_pmf1,self.number_genes_pmf2], final_data, final_y


class Genetic_BETA_resampling_framework(OverSamplingSimplex):

    def __init__(
        self,
        proportion=1.0,
        *,
        nn_params=None,
        ss_params=None,
        random_state=None,
        func1_pmf=None,
        func2_pmf=None,
        func1_generate_samples=None,
        func2_generate_samples=None,
        func1_toremove=False,
        func2_toremove=False,
        outlier_removal=True,
        Beta=1,
        n_splits=5,
        **_kwargs
    ):
        """
        Constructor of the SMOTE object

        Args:
            arg 1:  ...
            
            arg 2: ...
            
        """

        ss_params_default = {
            "n_dim": 2,
            "simplex_sampling": "random",
            "within_simplex_sampling": "random",
            "gaussian_component": None,
        }
        ss_params = coalesce_dict(ss_params, ss_params_default)

        OverSamplingSimplex.__init__(self, **ss_params, random_state=random_state)

        #relevant checks
        self.check_greater_or_equal(proportion, "proportion", 0)

        #saving init variables
        self.proportion      = proportion
        self.nn_params       = coalesce(nn_params, {})
        self.beta            = Beta
        self.n_splits        = n_splits
        self.func1_toremove  = func1_toremove
        self.func2_toremove  = func2_toremove
        self.func1_pmf       = func1_pmf
        self.func2_pmf       = func2_pmf
        self.func1_generate  = func1_generate_samples
        self.func2_generate  = func2_generate_samples
        self.outlier_removal = outlier_removal

    def create_dictionary_saving(self,to_remove):

        """
        
            Create a datastructe to save sample groups
        
        """

        sample_groups = {}
        required_variables = {}
        samples = {}

        #iterate over folds
        for fold in range(-1,self.n_splits):

            sample_groups[fold] = {}
            required_variables[fold] = {}

            if not to_remove:
                samples[fold] = {}
            

        return sample_groups, required_variables, samples

    def create_saving_datastructure(self):

        """
        
        Create a data format to save generated samples during optimization

        """

        #-----------------------------
        # create datastructure for data saving

        self.pmf_1, self.vars_pmf_1, self.pmf_1_samples = self.create_dictionary_saving(self.func1_toremove)
        self.pmf_2, self.vars_pmf_2, self.pmf_2_samples = self.create_dictionary_saving(self.func2_toremove)


    def generate_pmfs(self,X,y,fold):

        """
        
            Determine sample groups

        
        """
        self.pmf_1[fold], self.vars_pmf_1[fold] = self.func1_pmf(self,X,y,self.func1_toremove)
        self.pmf_2[fold], self.vars_pmf_2[fold] = self.func2_pmf(self,X,y,self.func2_toremove)

    def check_number_genes(self):


        """
        
        Check number of genes required to optimized a certain dataset


        
        """

        uniques, counts = np.unique(self.pmf_1[-1].astype(int),return_counts=True)
        self.number_genes_pmf1, self.genes_pmf1_counts = uniques[counts>0], counts[counts>0]
        #print(f"Genes 1: {self.number_genes_pmf1} ! counts gene 1: {self.genes_pmf1_counts}")

        uniques, counts = np.unique(self.pmf_2[-1].astype(int),return_counts=True)
        self.number_genes_pmf2, self.genes_pmf2_counts = uniques[counts>2], counts[counts>2]


    def generate_samples(self,X,y,n_to_sample,sample_group,pmf_number,fold):

        """
            Generate N samples from sample group 
        
        """

        if pmf_number == 1:


            self.pmf_1_samples[fold][sample_group] = self.func1_generate(self,X,y,self.pmf_1[fold],self.vars_pmf_1[fold],sample_group,n_to_sample)
        
        elif pmf_number == 2:

            self.pmf_2_samples[fold][sample_group] = self.func2_generate(self,X,y,self.pmf_1[fold],self.pmf_2[fold],sample_group,n_to_sample)

    
    def make_all_samples_fold(self,X,y,fold):

        """
        
            make all the required samples
        
        """
        total_number_samples = int(self.class_stats[self.maj_label]*self.proportion - self.class_stats[self.min_label])

        #generate for first pmf if required
        if not self.func1_toremove:

            unique, counts = np.unique(self.pmf_1[fold].astype(int),return_counts=True)

            for k,group in enumerate(unique):  

                    if counts[k] >0:           

                        self.generate_samples(X,y,total_number_samples,group,1,fold)

        #generate for second pmf if required
        if not self.func2_toremove:

            unique, counts = np.unique(self.pmf_2[fold].astype(int),return_counts=True)

            for k,group in enumerate(unique):  

                    if counts[k] >2:           

                        self.generate_samples(X,y,total_number_samples,group,2,fold)

    
    def sampling(self,X,y,distribution,fold):


        """

            Generate all the required synthetic data

        """
        #------------------------------------
        #get distribution values

        dist_pmf1  = distribution[:len(self.number_genes_pmf1)]
        dist_pmf2  = distribution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)]
        
        if self.outlier_removal:
            dist_ratio      = distribution[-2]
            percent_removal = distribution[-1]
        else:
            dist_ratio = np.abs(distribution[-1])

        ########################################################################
        # error handling for different groups in main vs original sample group

        samples_groups_in_fold_pmf1, counts_pmf1 = np.unique(self.pmf_1[fold].astype(int),return_counts=True)

        #print(f"PRIOR -- Dist: {dist_pmf1} | counts: {counts_pmf1} | available groups main: {self.number_genes_pmf1} | available groups: {samples_groups_in_fold_pmf1}")

        group_counts = dict(zip(samples_groups_in_fold_pmf1,counts_pmf1))
        indexes      = np.array([group_counts.get(group, 0) > 0 for group in self.number_genes_pmf1])
        dist_pmf1[~indexes] = 0

        #print(f"Indexes: {indexes}")
        #print(f"POST -- Dist: {dist_pmf1} | counts: {counts_pmf1} | available groups main: {self.number_genes_pmf1} | available groups: {samples_groups_in_fold_pmf1}")

        samples_groups_in_fold_pmf2, counts_pmf2 = np.unique(self.pmf_2[fold].astype(int),return_counts=True)
        
        group_counts = dict(zip(samples_groups_in_fold_pmf2,counts_pmf2))
        indexes      = np.array([group_counts.get(group, 0) > 2 for group in self.number_genes_pmf2])
        dist_pmf2[~indexes] = 0

        counts_pmf2 = np.array([group_counts.get(group,0) for group in self.number_genes_pmf2])
        
        #print(f"POST -- Dist: {dist_pmf2} | counts: {counts_pmf2} | available groups main: {self.number_genes_pmf2} | available groups: {samples_groups_in_fold_pmf2}")
        # print(f"Counts PMF_2: {counts_pmf2}")

        #error handling if there are no samples, return original data

        ##################################################################


        #check total number of required samples
        if self.func2_toremove:
            #total_main_dist   = np.sum(y == self.maj_label)*self.proportion*(1-dist_ratio) - np.sum(y == self.min_label)
            to_remove_samples = np.sum(y == self.maj_label)*dist_ratio

            #check whether it is possible to remove such samples, if not, adjust
            samples_to_remove_per_group = to_remove_samples*dist_pmf2
 
            condition = samples_to_remove_per_group > counts_pmf2
            samples_to_remove_per_group[condition] = counts_pmf2[condition]

            #update required amount of samples to make
            to_remove_samples = np.sum(samples_to_remove_per_group)
            total_main_dist   = (np.sum(y == self.maj_label) - to_remove_samples)*self.proportion - np.sum(y ==self.min_label)

        #in case both pmf are to generate samples
        else:
            total_number_samples = np.sum(y == self.maj_label)*self.proportion - np.sum(y == self.min_label)
            total_main_dist      = total_number_samples*dist_ratio
            total_secundary_dist = total_number_samples*(1-dist_ratio)

        #instanciate new syntetic dataset
        temp_data      = X[y==self.min_label]
        temp_y         = y[y==self.min_label]

        #-------------------------------
        # outlier removal procedure

        if self.outlier_removal:
            number_to_remove  = int(len(self.vars_pmf_2[fold][0])*percent_removal) 
            indexes_to_remove = self.vars_pmf_2[fold][0][-number_to_remove:]
            new_indexes       = np.setdiff1d(np.arange(len(temp_y)),indexes_to_remove)
            temp_y            = temp_y[new_indexes]
            temp_data         = temp_data[new_indexes]


            #forcing behaviour
            # if percent_removal > 0.4:
            #     dist_pmf1[-1] = 0
            #     dist_pmf2[0]  = 0

        ###########################

        if np.sum(dist_pmf2) == 0 or np.sum(dist_pmf1) == 0:
            return X, y 

        #
        #print(f"probabilities after outlier removal: {dist_pmf1} | {dist_pmf2}")

        #normalize distributions
        dist_pmf1 = np.abs(dist_pmf1) / np.sum(dist_pmf1)
        dist_pmf2 = np.abs(dist_pmf2) / np.sum(dist_pmf2)

        #-----------------------------------
        #main dist sample generation

        #print(f"total: {total_number_samples} | main: {total_main_dist} | secu: {total_secundary_dist} | manual: {np.bincount(y.astype(int))}")
        added_samples = 0

        #iterate over distribution and select/generate samples
        for k,main_dist_value in enumerate(dist_pmf1):

            #print(f"Fold: {fold} | Main dist: {dist_pmf1} | Main dist value at k: {main_dist_value} | {self.number_genes_pmf1[k]}")
            
            #avoid sampling if no samples exist
            if main_dist_value != 0:

                required_samples_main = np.abs(np.ceil(main_dist_value*total_main_dist).astype(np.int16))

                #print(f"Required samples main; {required_samples_main}")

                #added_samples += len(self.pmf_1_samples[fold][self.number_genes_pmf1[k]][:required_samples_main,:])
                temp_data             = np.vstack((temp_data,self.pmf_1_samples[fold][self.number_genes_pmf1[k]][:required_samples_main,:]))
                temp_y                = np.hstack((temp_y,np.ones((required_samples_main,))*self.min_label))
                
        #--------------------------------
        #secundary dist sample generation /deletion
        if self.func2_toremove:
            
            #create set to save indexes of samples to be remove
            total_sample_indexes = np.empty((0,))

            #get majority samples in original indexes
            converter            = np.where(y==self.maj_label)[0]

            for k, to_remove_group in enumerate(samples_to_remove_per_group):
                
                #get samples in indexes of sample group 
                indexes_selected = self.pmf_2[fold] == self.number_genes_pmf2[k]

                if np.sum(indexes_selected) == 0 or to_remove_group == 0:
                    continue

                #print(f"Samples to remove in {k} group: {to_remove_group}")


                #get indexes in original indexes

                try:
                    indexes_selected = converter[indexes_selected][self.vars_pmf_2[fold][k]][:int(to_remove_group)]

                    #save selected samples
                    total_sample_indexes  = np.hstack((total_sample_indexes,indexes_selected))
                except:
                    print(f"{converter[indexes_selected].shape} | {len([self.vars_pmf_2[fold][k]])} | {int(to_remove_group)}")



            #actually remove samples
            indexes_to_keep = np.setdiff1d(converter,total_sample_indexes)
            temp_data = np.vstack((temp_data,X[indexes_to_keep,:]))
            temp_y    = np.hstack((temp_y,np.ones((
                                        len(indexes_to_keep),
                                        ))*self.maj_label
                                    ))

        else:

            for k, secundary_dist_value in enumerate(dist_pmf2):  

                #print(f"Fold: {fold} | Main dist: {secundary_dist_value} | {self.number_genes_pmf2[k]}")

                if secundary_dist_value !=0:    

                    required_samples_sec = np.ceil(secundary_dist_value*total_secundary_dist).astype(np.int16)
                    #added_samples += len(self.pmf_2_samples[fold][self.number_genes_pmf2[k]][:required_samples_sec,:])
                    temp_data            = np.vstack((temp_data,self.pmf_2_samples[fold][self.number_genes_pmf2[k]][:required_samples_sec,:]))
                    temp_y               = np.hstack((temp_y,np.ones((required_samples_sec,))*self.min_label))
            
            #add samples majority class, meaning no oversampling
            temp_data = np.vstack((temp_data,X[y==self.maj_label]))
            temp_y    = np.hstack((temp_y,y[y==self.maj_label]))

        #print(f"Added {added_samples} samples")

        return temp_data, temp_y
    
    
    def fitness_function(self,ga_instance,solution,solution_idx):

        """
            fitness function compution given a certain distribution        
        """
        #print(f"Main dist: {solution[:len(self.number_genes_pmf1)]} | secundary sol: {solution[len(self.number_genes_pmf1):-2]} at {solution_idx}")

        model_performance = []
        #model = RandomForestClassifier(n_estimators=10,max_depth=5,random_state=1)
        model = GaussianNB()


        for fold, (train_idx, test_idx) in enumerate(self.kf.split(self.X,self.y)):

            #print(f"{fold} - SOlution inside fold: {solution} | 2: {solution.copy()}")

            #make training/testing dataset
            train_data, test_data = self.X[train_idx], self.X[test_idx]
            train_y   , test_y    = self.y[train_idx], self.y[test_idx] 

            #print(f"Solution: {solution} | {train_data.shape} and {train_y.shape} | counts: {np.bincount(train_y.astype(int))}")

            synthetic_X, synthetic_y  = self.sampling(train_data,train_y,solution.copy(),fold)

            #print(f"Synthetic data shape: {synthetic_X.shape} and {synthetic_y.shape} | counts: {np.bincount(synthetic_y.astype(int))} ")

            #test model performance
            model.fit(synthetic_X,synthetic_y)
            model_performance.append(fbeta_score(test_y,model.predict(test_data),beta=self.beta))

        return np.mean(model_performance)
        
    def make_final_samples(self,X,y,distribution):

        """
        
        Make final samples by sampling each possible fold
        
        """

        #------------------------------------
        #get distribution values

        dist_pmf1  = distribution[:len(self.number_genes_pmf1)]
        dist_pmf2  = distribution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)]
        
        if self.outlier_removal:
            dist_ratio      = distribution[-2]
            percent_removal = distribution[-1]
        else:
            dist_ratio = np.abs(distribution[-1])

        ########################################################################

        _, counts_pmf2 = np.unique(self.pmf_2[-1].astype(int),return_counts=True)
        counts_pmf2 = counts_pmf2[counts_pmf2 > 2]

        #error handling if there are no samples, return original data

        if np.sum(dist_pmf2) == 0 or np.sum(dist_pmf1) == 0:
            return X, y 

        ##################################################################        

        #check total number of required samples
        if self.func2_toremove:
            #total_main_dist   = np.sum(y == self.maj_label)*self.proportion*(1-dist_ratio) - np.sum(y == self.min_label)
            to_remove_samples = np.sum(y == self.maj_label)*dist_ratio

            #check whether it is possible to remove such samples, if not, adjust
            samples_to_remove_per_group = to_remove_samples*dist_pmf2
 
            condition = samples_to_remove_per_group > counts_pmf2
            samples_to_remove_per_group[condition] = counts_pmf2[condition]

            #update required amount of samples to make
            to_remove_samples = np.sum(samples_to_remove_per_group)
            total_main_dist   = (np.sum(y == self.maj_label) - to_remove_samples)*self.proportion - np.sum(y ==self.min_label)

        #in case both pmf are to generate samples
        else:
            total_number_samples = np.sum(y == self.maj_label)*self.proportion - np.sum(y == self.min_label)
            total_main_dist      = total_number_samples*dist_ratio
            total_secundary_dist = total_number_samples*(1-dist_ratio)

        #instanciate new syntetic dataset
        temp_data      = X[y==self.min_label]
        temp_y         = y[y==self.min_label]

        #-------------------------------
        # outlier removal procedure

        if self.outlier_removal:
            number_to_remove  = int(len(self.vars_pmf_2[-1][0])*percent_removal) 
            indexes_to_remove = self.vars_pmf_2[-1][0][-number_to_remove:]
            new_indexes       = np.setdiff1d(np.arange(len(temp_y)),indexes_to_remove)
            temp_y            = temp_y[new_indexes]
            temp_data         = temp_data[new_indexes]

            #forcing behaviour
            # if percent_removal > 0.4:
            #     dist_pmf1[-1] = 0
            #     dist_pmf2[0]  = 0

        ###########################

        #normalize distributions
        dist_pmf1 = np.abs(dist_pmf1) / np.sum(dist_pmf1)
        dist_pmf2 = np.abs(dist_pmf2) / np.sum(dist_pmf2)

        #-----------------------------------
        #main dist sample generation
        counts_per_fold = np.array([
            [len(self.pmf_1_samples[fold].get(sample_group, [])) for sample_group in self.number_genes_pmf1]
            for fold in range(self.n_splits)
        ])
        #iterate over distribution and select/generate samples
        for k,main_dist_value in enumerate(dist_pmf1):

            available_folds = np.where(counts_per_fold[:,k]>0)[0]
            
            #avoid sampling if no samples exist
            if main_dist_value != 0:

                required_samples_main = np.abs(np.ceil(main_dist_value*total_main_dist).astype(np.int16))

                for fold in available_folds:

                    idx = np.arange(fold,required_samples_main,len(available_folds))

                    temp_data             = np.vstack((temp_data,self.pmf_1_samples[fold][self.number_genes_pmf1[k]][idx,:]))
                    temp_y                = np.hstack((temp_y,np.ones((len(idx),))*self.min_label))
                
        #--------------------------------
        #secundary dist sample generation /deletion
        if self.func2_toremove:
            
            #create set to save indexes of samples to be remove
            total_sample_indexes = np.empty((0,))

            #get majority samples in original indexes
            converter            = np.where(y==self.maj_label)[0]

            for k, to_remove_group in enumerate(samples_to_remove_per_group):
                
                #get samples in indexes of sample group 
                indexes_selected = self.pmf_2[-1] == self.number_genes_pmf2[k]

                #print(f"Samples to remove in {k} group: {to_remove_group}")
                
                #if samples are not present in fold
                if np.sum(indexes_selected) == 0 or to_remove_group == 0:
                    continue

                #get indexes in original indexes
                try:
                    indexes_selected = converter[indexes_selected][self.vars_pmf_2[-1][k]][:int(to_remove_group)]

                    #save selected samples
                    total_sample_indexes  = np.hstack((total_sample_indexes,indexes_selected))
                except:
                    print(f"{converter[indexes_selected].shape} | {len([self.vars_pmf_2[fold][k]])} | {int(to_remove_group)}")


            #actually remove samples
            indexes_to_keep = np.setdiff1d(converter,total_sample_indexes)
            temp_data = np.vstack((temp_data,X[indexes_to_keep,:]))
            temp_y    = np.hstack((temp_y,np.ones((
                                        len(indexes_to_keep),
                                        ))*self.maj_label
                                    ))

        else:
            counts_per_fold = np.array([
                [len(self.pmf_2_samples[fold].get(sample_group, [])) for sample_group in self.number_genes_pmf2]
                for fold in range(self.n_splits)
            ])
            for k, secundary_dist_value in enumerate(dist_pmf2):    

                available_folds = np.where(counts_per_fold[:,k]>0)[0]

                if secundary_dist_value !=0:    

                    required_samples_sec = np.ceil(secundary_dist_value*total_secundary_dist).astype(np.int16)

                    for fold in available_folds:

                        idx = np.arange(fold,required_samples_sec,len(available_folds))

                        temp_data            = np.vstack((temp_data,self.pmf_2_samples[fold][self.number_genes_pmf2[k]][idx,:]))
                        temp_y               = np.hstack((temp_y,np.ones((len(idx),))*self.min_label))
            
            #add samples majority class, meaning no oversampling
            temp_data = np.vstack((temp_data,X[y==self.maj_label]))
            temp_y    = np.hstack((temp_y,y[y==self.maj_label]))

        return temp_data, temp_y
    

    def resample(self,X,y):

        """
            does the optimization
        
        """

        #----------------------
        # init vars

        self.X = X
        self.y = y
        self.class_label_statistics(self.y)
        self.create_saving_datastructure()
        self.kf    = StratifiedKFold(n_splits=self.n_splits,shuffle=True,random_state=1)
        self.model = RandomForestClassifier(n_estimators=10,max_depth=5,random_state=1) 

        #---------------------
        # getting pmfs and synthetic samples for whole data

        self.generate_pmfs(X,y,-1)
        self.check_number_genes()
        self.make_all_samples_fold(X,y,-1)

        #---------------------
        # getting pmfs and synthetic data for each fold

        for fold, (train_idx, _) in enumerate(self.kf.split(self.X,self.y)):

            #print(f"Conducting analysis on fold: {fold}")
            
            #divide into train and test split
            temp_data, temp_y = self.X[train_idx], self.y[train_idx]

            #(1)
            self.generate_pmfs(temp_data,temp_y,fold)
            #(2)
            self.make_all_samples_fold(temp_data,temp_y,fold)    

        #------------------------
        # init vars for optimization

        #define gene space

        #20 > 2000
        #10 > 1000

        gene_space = [np.linspace(0.01,0.5,20) for _ in range(len(self.number_genes_pmf1)+len(self.number_genes_pmf2))] #removal
        gene_space.append(np.linspace(0.01,0.99,20)) #dist ratio

        #increase gene space if outlier removal is needed
        if self.outlier_removal:
            gene_space.append(np.linspace(0,0.5,10)) #outlier removal

        #other vars
        num_genes = len(gene_space)
        fitness_function = self.fitness_function
        num_generations = 300
        sol_per_pop = 25
        num_parents_mating = int(sol_per_pop/2)
        parent_selection_type = "tournament"
        crossover_type = "single_point"
        mutation_type = "swap"
        keep_elistism = 4
        crossover_probability = 0.75
        mutation_probability = 0.25
        K_tornament = 4
        stop_criteria = ["reach_1"]

        #------------------------
        # run optimization

        ga_instance = pygad.GA(num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_function,
                sol_per_pop=sol_per_pop,
                num_genes=num_genes,
                K_tournament=K_tornament,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
                parent_selection_type=parent_selection_type,
                crossover_type=crossover_type,
                mutation_type=mutation_type,
                gene_space=gene_space,
                parallel_processing=16,
                keep_elitism=keep_elistism,
                stop_criteria=stop_criteria,
                save_solutions=True)
        
        ga_instance.run()

        #----------------------------------------
        # get results and generate final dataset

        #get solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        final_data, final_y = self.sampling(X,y,solution,-1)

        PMF_1 =   solution[:len(self.number_genes_pmf1)] / np.sum(solution[:len(self.number_genes_pmf1)])
        PMF_2 =   solution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)] / np.sum(solution[len(self.number_genes_pmf1):len(self.number_genes_pmf1)+len(self.number_genes_pmf2)])

        print(f"Fittest Solution: PMF_1 - {PMF_1} | PMF_2 - {PMF_2}\nDist ratio: {solution[-1]}")
        print(f"FInal data shape: {final_data.shape} and {final_y.shape} | counts: {np.bincount(final_y.astype(int))}")
        print("*"*50)

        return [solution,self.number_genes_pmf1,self.number_genes_pmf2], final_data, final_y
    
