import numpy as np

def convolve2d(values, filtre, mode="full"):

    added_values = np.zeros((filtre.shape[0]-1, values.shape[1]))
    values = np.concatenate((added_values, values, added_values), axis=0)
    
    added_values = np.zeros((values.shape[0], filtre.shape[1]-1))
    values = np.concatenate((added_values, values, added_values), axis=1)


    size_image = values.shape
    size_filtre = filtre.shape

    values = np.sum((
        values[
            size_0:size_image[0]-size_filtre[0] + 1 + size_0,
            size_1:size_image[1]-size_filtre[1] + 1 + size_1
        ] * filtre[size_0, size_1] 
        for size_0 in range(size_filtre[0]) 
        for size_1 in range(size_filtre[1])
        ), axis=0)

    if mode == "valid":
        values = values[filtre.shape[0]-1:-filtre.shape[0]+1, filtre.shape[1]-1:-filtre.shape[1]+1]
    
    return values

def convolve3d(values, filtre, mode="full"):

# b = np.repeat(f, a.size).reshape(f.shape + a.shape)
# b = np.tile(a, f.shape) * np.repeat(np.repeat(f, a.shape[1], axis=1), a.shape[0], axis=0)
# c = b*a

    added_values = np.zeros((filtre.shape[0]-1, values.shape[1], values.shape[2]))
    values = np.concatenate((added_values, values, added_values), axis=0)
    
    added_values = np.zeros((values.shape[0], filtre.shape[1]-1, values.shape[2]))
    values = np.concatenate((added_values, values, added_values), axis=1)

    added_values = np.zeros((values.shape[0], values.shape[1], filtre.shape[2]-1))
    values = np.concatenate((added_values, values, added_values), axis=2)

    size_image = values.shape
    size_filtre = filtre.shape

    values = np.sum((
        values[
            size_0:size_image[0]-size_filtre[0] + 1 + size_0,
            size_1:size_image[1]-size_filtre[1] + 1 + size_1,
            size_2:size_image[2]-size_filtre[2] + 1 + size_2
        ] * filtre[size_0, size_1, size_2] 
        for size_0 in range(size_filtre[0]) 
        for size_1 in range(size_filtre[1])
        for size_2 in range(size_filtre[2])
        ), axis=0)

    if mode == "valid":
        values = values[
            filtre.shape[0]-1:values.shape[0]-size_filtre[0] + 1, 
            filtre.shape[1]-1:values.shape[1]-size_filtre[1] + 1, 
            filtre.shape[2]-1:values.shape[2]-size_filtre[2] + 1
        ]
    
    return values

def multi_convolve3d(values, filtre, mode="full"):
    """Filtre should have one dim more than values

    Args:
        values (ndarray): _description_
        filtre (ndarray): _description_
    """
    added_values = np.zeros((filtre.shape[1]-1, values.shape[1], values.shape[2]))
    values = np.concatenate((added_values, values, added_values), axis=0)
    
    added_values = np.zeros((values.shape[0], filtre.shape[2]-1, values.shape[2]))
    values = np.concatenate((added_values, values, added_values), axis=1)

    added_values = np.zeros((values.shape[0], values.shape[1], filtre.shape[3]-1))
    values = np.concatenate((added_values, values, added_values), axis=2)

    size_image = values.shape
    size_filtre = filtre.shape

    values = np.array([
        np.sum((
            values[
                size_0:size_image[0]-size_filtre[1] + 1 + size_0,
                size_1:size_image[1]-size_filtre[2] + 1 + size_1,
                size_2:size_image[2]-size_filtre[3] + 1 + size_2
            ] * filtre[index_filter, size_0, size_1, size_2] 
            for size_0 in range(size_filtre[1]) 
            for size_1 in range(size_filtre[2])
            for size_2 in range(size_filtre[3])
        ), axis=0)
        for index_filter in range(size_filtre[0])])

    if mode == "valid":
        values = values[
            :,
            filtre.shape[1]-1:values.shape[1]-size_filtre[1] + 1, 
            filtre.shape[2]-1:values.shape[2]-size_filtre[2] + 1, 
            filtre.shape[3]-1:values.shape[3]-size_filtre[3] + 1
        ]
    
    return values





