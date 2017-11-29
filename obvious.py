import random
import math

def euclidean_distance_square(a,b):
    # validate
    if len(a) != len(b):
        raise Exception('len(a) != len(b)')
        return

    # counting distance
    distance_square = 0
    for i in range(0,len(a)):
        distance_square += (a[i] - b[i])**2
    
    return math.sqrt(distance_square)

def build_obvious(  centroids = [[0,0],[10,10]],
                     amount_instance_each_cluster=[5,10],
                     range_to_centroid = 2,
                     range_type = 'euclidean',
                     amount_of_outlier = 0,
                     outlier_border_ratio = 5):
    """
    centroids : centroid-centroidnya. Panjangnya harus sama dengan amount_instance_each_cluster
    amount_instance_each_cluster : jumlah data yang ingin dibuat tiap cluster
    range_to_centroid : jarak yang ingin dibuat tiap centroid
    range_type : 
        square_area => untuk setiap atribut (attx), attx-range< attx < attx+range
        euclidean => pake euclidean buat ngitung
    amount_of_outlier : jumlah outlier
    outlier_border_ratio : susah jelasinnya. intinya, makin besar, makin jauh. 
                *tenang aja, ga sejauh itu kok*
    """
        
    # validating range_type
    if range_type not in ['square_area', 'euclidean']:
        raise Exception("Range Type not recognize (only 'square_area' or 'euclidean')")
    
    # validating attribute of centroid. All must be same.
    ammount_of_attribute = len(centroids[0])
    #print(ammount_of_attribute)
    for centroid in centroids:
        if len(centroid) != ammount_of_attribute:
            raise Exception("Number attribute of all centroid not same")
            return
    # validating outlier properties
    if amount_of_outlier > 0 and outlier_border_ratio < 1:
        raise Exception("If amount_of_outlier, outlier_border_ratio can't < 1")
        return        
    if amount_of_outlier > 0 and range_to_centroid == 0:
        raise Exception("range_to_centroid to low.")
    
    if len(centroids) != len(amount_instance_each_cluster):
        raise Exception("len(centroids) != len(amount_instance_each_cluster)")        
    
    # creating all
    instances_all = []
    for idx_current_centroid in range(0,len(amount_instance_each_cluster)):
        instances_current_centroid = []
        while len(instances_current_centroid) < amount_instance_each_cluster[idx_current_centroid]:
            instance = []
            for idx_att in range(0,ammount_of_attribute):
                attx = centroids[idx_current_centroid][idx_att]
                rand_attx = random.randint(attx-range_to_centroid, attx+range_to_centroid)
                instance.append(rand_attx)
            if (range_type == 'euclidean') and euclidean_distance_square(instance, centroids[idx_current_centroid]) > range_to_centroid:
                pass
            else:                                
                instances_current_centroid.append(instance)
        instances_all.extend(instances_current_centroid)
        
    # creating outlier    
    outlier_list = []
    border = range_to_centroid*outlier_border_ratio

    outlier_border_ratio = 0 
    while len(outlier_list) < amount_of_outlier:
        for i in range(0,amount_of_outlier):

            # creating outlier one by one
            outlier = []
            for idx_att in range(0,ammount_of_attribute):
                attx = centroids[idx_current_centroid][idx_att]
                rand_attx = random.randint(attx-border, attx+border)
                outlier.append(rand_attx)                

            # check if it far enough to centroid
            valid = True
            for idx_current_centroid in range(0,len(centroids)):
                _range_to_outlier =euclidean_distance_square(outlier, centroids[idx_current_centroid])
                if _range_to_outlier < border:
                    valid = False
                    break
            if valid:
                outlier_list.append(outlier)
    instances_all.extend(outlier_list)
            
    return instances_all 