import numpy as np

list_attributes_session_to_edit = ['dF_F', 'name_roi', 'S_noisy','S','activations','kept_cells','roi_mask',
                                   'rate_f','rate_n','selectivity','rate_f_discrete','rate_n_discrete',
                                   'selectivity_discrete', 'rate_f_continuous','rate_n_continuous',
                                   'selectivity_continuous','selectivity_manu','selectivity_manu_div',
                                   'selectivity_manu_nodiv','rois','rate_vector']

list_attributes_laps_to_edit = ['dF_F', 'S','event_length','activations','event_count',
                                'raster','rate_maps','s_maps','rate_vector','rois']


def selectPlaceCells(sess_arg, dict_place_cells, mode = 'any'):
    """
    Edit sessions so the attributes defined in list_attributes_session_to_edit and list_attributes_laps_to_edit
    only contains the place cells, according to dict_place_cells.
    Mode allows to select specifically cells which are place cells in 'familiar', 'novel', 'both' or 'any' laps.
    """
    
    #Throwing exception if place cells were not computed
    if sess_arg.session_name not in dict_place_cells.keys():
        raise Exception('Place cells were not computed for this session.')
    
    #Defining boolean vector of valid cells.
    d = dict_place_cells[sess_arg.session_name]
    if mode == 'familiar':
        kept_cells = d['place_cells_F']
    elif mode == 'novel':
        kept_cells = d['place_cells_N']
    elif mode == 'any':
        kept_cells = d['place_cells_F'] | d['place_cells_N']
    elif mode == 'both':
        kept_cells = d['place_cells_F'] & d['place_cells_F']
    else:
        raise Exception("Unvalid mode. Value should be 'familiar', 'novel', 'any' or 'both'")
        
    selectCellSubsetSession(sess_arg, kept_cells)  
    return


def selectCellSubsetSession(sess_arg, kept_cells):
    """
    Edit the atributes listed in list_attributes_session_to_edit from the session to keep 
    only a subset of cells defined by the boolean vector kept_cells.
    Throw an exception if attribute dimensions do not match the expected values defined in n_roi.
    """
    
    #Checking consistency in nroi
    if sess_arg.n_roi != sess_arg.n_roi_all:
        raise Exception('Attribute n_roi does not match n_roi_all')
        
    #First handling the session attributes
    for at in list_attributes_session_to_edit:        
        cur_at = getattr(sess_arg, at)
        str_error = 'Attribute ' + at + ' size does not match n_roi.'   
        
        #List case
        if type(cur_at) == list:            
            #Checking consistency with nroi
            if len(cur_at) != sess_arg.n_roi:                
                raise Exception(str_error)                
            sub_at = [e for i,e in enumerate(cur_at) if kept_cells[i]]            
        
        #Array case
        else:
            #Specific case, kept_cells length might not match n_roi. The number of True values doees.
            if at == 'kept_cells':
                if np.sum(sess_arg.kept_cells) != sess_arg.n_roi:
                    raise Exception(str_error)
                sub_at = sess_arg.kept_cells[sess_arg.kept_cells][kept_cells]  
                
            #Default case
            else:
                #Checking consistency with nroi
                if cur_at.shape[0] != sess_arg.n_roi:
                    raise Exception(str_error)

                if len(cur_at.shape) == 1:
                    sub_at = cur_at[kept_cells]
                elif len(cur_at.shape) == 2:
                    sub_at = cur_at[kept_cells,:]
                
        #Setting new attribute value
        setattr(sess_arg, at, sub_at)

    #Now updating the attributes from the lap object, individual and familiar laps will be edited automatically by reference
    for l in sess_arg.laps:
        selectCellSubsetLaps(l, kept_cells)
    
    #Finally editing the number of rois
    new_n_roi = np.sum(kept_cells)
    setattr(sess_arg, 'n_roi', new_n_roi)
    setattr(sess_arg, 'n_roi_all', new_n_roi)
    
    return


def selectCellSubsetLaps(lap_arg, kept_cells):
    """
    Edit the attributes listed in list_attributes_laps_to_edit from an invidual lap to keep 
    only a subset of cells defined by the boolean vector kept_cells.
    Throw an exception if attribute dimensions do not match the expected values defined in n_roi.
    """  
    
    #First handling the session attributes
    for at in list_attributes_laps_to_edit:        
        cur_at = getattr(lap_arg, at)
        str_error = 'Attribute ' + at + ' does not have the expected size.'
        
        #List case
        if type(cur_at) == list:            
            #Checking consistency with nroi
            if len(cur_at) != lap_arg.n_roi:                
                raise Exception(str_error)                
            sub_at = [e for i,e in enumerate(cur_at) if kept_cells[i]]
        
        #Array case
        else:            
            #1-dimensional array
            if len(cur_at.shape) == 1:                
                #Checking consistency with nroi
                if cur_at.shape[0] != lap_arg.n_roi:
                    raise Exception(str_error)                
                sub_at = cur_at[kept_cells]
                
            #2-dimensional array
            else:                
                #Specific attributes, raster has the number of cells as second dimension
                if at == 'raster':                    
                    #Checking consistency with nroi
                    if cur_at.shape[1] != lap_arg.n_roi:
                        raise Exception(str_error)
                    sub_at = cur_at[:,kept_cells]                      
                #Default case
                else:                
                    #Checking consistency with nroi
                    if cur_at.shape[0] != lap_arg.n_roi:
                        raise Exception(str_error)
                    sub_at = cur_at[kept_cells,:]    
    
        #Setting new attribute value
        setattr(lap_arg, at, sub_at)
        
    #Editing the number of roi
    new_n_roi = np.sum(kept_cells)
    setattr(lap_arg, 'n_roi', new_n_roi)
    
    return
