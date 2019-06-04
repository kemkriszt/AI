from sklearn import preprocessing
def scale(data,method):
	"""Scale the feature across data points given the method

    Parameters
    ----------
    data : list

    method : int
       The scaling method
	   1 => standardization
	   2 => min max
	   3 => robust scaling

    Returns
    -------
    scaled_data : list
        List containing the scaled data
    """
	scaled_data = data
	if method == 1:
		scaled_data = preprocessing.scale(data)
	elif method == 2:
		scaled_data = preprocessing.minmax_scale(data)
	elif method == 3:
		scaled_data = preprocessing.robust_scale(data)
	
	return data