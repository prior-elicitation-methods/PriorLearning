import pickle
import os

def save_as_pkl(variable, path_to_file): 
    """
    Helper functions to save a file as pickle.

    Parameters
    ----------
    variable : any
        file that needs to be saved.
    path_to_file : str
        path indicating the file location.

    Returns
    -------
    None.

    """
    # if directory does not exists, create it
    os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    # save file to location as pickle
    with open(path_to_file, 'wb') as df_file:
        pickle.dump(variable, file = df_file) 