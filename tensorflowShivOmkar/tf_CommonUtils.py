def reduce_concat(x, sep=""):
    import functools
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)

def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)

# paste0 = functools.partial(paste, sep="")

#Description: Get PCA transformed data.
# Input: provide non scaled data and count of PCA component
def GetPCA(train, n_components=None, bScale = False, fileImageToSave = "pca.png"):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Scale the incoming data
    if bScale:
        train = ScaleAndCenter_NumericOnly(train)

    # create instance of PCA object
    pca = PCA(n_components=n_components)
    # Fit the model with X and apply the dimensionality reduction on X.
    train = pca.fit_transform(train)

    #Cumulative Variance explains
    cumVarExplained = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    n_components = len(cumVarExplained)

    #Plot the cumulative explained variance as a function of the number of components
    plt.subplots(figsize=(13, 9))
    plt.plot(range(1,n_components+1,1), cumVarExplained, 'bo-')  #
    plt.ylabel('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(1, n_components+1, 1))
    plt.title("PCA: Number of Features vs Variance (%)")
    #plt.ylim([0.0, 1.1])
    #plt.yticks(np.arange(0.0, 1.1, 0.10))
    #plt.show()
    plt.tight_layout()  # To avoid overlap of subtitles
    print("PCA: Number of Features vs Variance (%) is saved to " + fileImageToSave)
    plt.savefig(fileImageToSave, bbox_inches='tight')
    plt.close()
    plt.gcf().clear()

    print("PCA: Number of Features vs Variance (%) CSV is saved to pca.csv")
    df = pd.DataFrame({"CumVarExplained" : cumVarExplained, "Components": range(1,n_components+1,1)})
    df.to_csv('pca.csv', index=False)
    del(df)
    # Prepapre for return
    train = pd.DataFrame(train, columns= paste(["PC"] * n_components, np.arange(1, n_components+1, 1), sep=''))  #   # ('string\n' * 4)[:-1]

    return(train.iloc[:, 0:n_components])
    # end of GetPCA

#%% Utility file and folder related
# location= file_save_model
def remove_nonempty_folder(location):
    import shutil, stat
    fileList = os.listdir(location)

    # Clean the folder by deleting the files and folder inside
    for fileName in fileList:
        fullpath=os.path.join(location, fileName)
        if os.path.isfile(fullpath):
            os.chmod(fullpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.remove(os.path.join(location, fileName))
        elif os.path.islink(fullpath):
            os.unlink(fullpath)
        elif os.path.isdir(fullpath):
            if len(os.listdir(fullpath)) > 0:
                remove_nonempty_folder(fullpath)
            if os.path.exists(os.path.join(location, fileName)):
                shutil.rmtree(os.path.join(location, fileName))

    # Remove the folder. Check is there to avoid the iteration from above
    if os.path.exists(location):
        shutil.rmtree(location)

    return
# end of remove_nonempty_folder

def save_in_image_data_generator_format(folder_name, train_data, train_labels, eval_data, eval_labels):
    # Sanity test and clening old files        
    if os.path.exists(folder_name):
        print('Folder cleaned',folder_name)
        remove_nonempty_folder(folder_name)
        
    #Few constants
    folder_name_train = os.path.join(folder_name, 'train')
    folder_name_eval = os.path.join(folder_name, 'eval')
    im_wh = train_data.shape[1]
    
    #Create folder and subfolder
    print('Folder created',folder_name)
    os.mkdir(folder_name)
    print('Folder created',folder_name_train)
    os.mkdir(folder_name_train)
    print('Folder created',folder_name_eval)
    os.mkdir(folder_name_eval)
    
    #Create class folder for train
    for class_name in np.unique(np.array(train_labels)):
        folder_name_class = os.path.join(folder_name_train, str(class_name))
        print('Folder created',folder_name_class)
        os.mkdir(folder_name_class)
        del(folder_name_class)
    
    #Create class folder for eval
    for class_name in np.unique(np.array(eval_labels)):
        folder_name_class = os.path.join(folder_name_eval, str(class_name))
        print('Folder created',folder_name_class)
        os.mkdir(folder_name_class)
        del(folder_name_class)
    
    #Save train images
    for img_count in range(train_data.shape[0]):
        file_name = os.path.join(folder_name_train, str(train_labels[img_count]), str(img_count) + '.png')
        print('Saving ', file_name)
        image_data = train_data[img_count].reshape((im_wh, im_wh, 1))
        tf.compat.v2.keras.preprocessing.image.save_img(file_name, image_data, scale = False)
        del(image_data, file_name)
        
    #Save eval images
    for img_count in range(eval_data.shape[0]):
        file_name = os.path.join(folder_name_eval, str(eval_labels[img_count]), str(img_count) + '.png')
        print('Saving ', file_name)
        image_data = eval_data[img_count].reshape((im_wh, im_wh, 1))
        tf.compat.v2.keras.preprocessing.image.save_img(file_name, image_data, scale = False)
        del(image_data, file_name)
    
    del(folder_name_train, folder_name_eval, im_wh)
    return
#end of save_in_image_data_generator_format
    
#Dummy Encoding of categorical data, scale and center numerical data
#def Encoding(train, strResponse, scale_and_center = False, fileTrain = "train_EncodedScaled.csv", fileTest = "test_EncodedScaled.csv"):
def Encoding(train, strResponse, scale_and_center = False, fileTrain = "train_EncodedScaled.csv"):
    from sklearn import preprocessing

    # get all numeric features
    listNumericFeatures = train.select_dtypes(include=[np.number]).columns.values

    # get all categorical features
    listCategoricalFeatures = []
    for columnName in train.columns:
        if hasattr(train[columnName], 'cat'):
            listCategoricalFeatures.append(columnName)

    # Remove Response variable
    if hasattr(train[strResponse], 'cat'):
        listCategoricalFeatures = list(set(listCategoricalFeatures) - set([strResponse]))
    elif np.issubdtype(train[strResponse].dtype, np.number):
        listNumericFeatures = list(set(listNumericFeatures) - set([strResponse]))
        
    #Default
    df = train
    # If scale and center (in 0-1) is true
    if (scale_and_center and len(listNumericFeatures) > 0):
        print("Scaling and Creating Dummy features")
        # get the same sclaer for both train and test
        min_max_scaler = preprocessing.MinMaxScaler()

        # do for train
        df = pd.DataFrame(min_max_scaler.fit_transform(train[listNumericFeatures]), index=train.index, columns=listNumericFeatures)

        if(len(listCategoricalFeatures) > 0):
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), df, train[strResponse]], axis = 1, join = "outer") # cbind
        else:
            df = pd.concat([df, train[strResponse]], axis = 1, join = "outer") # cbind

        print("Encoded train file is saved to " + fileTrain)
        df.to_csv(fileTrain, index=False)  #min_max_scaler.scale_  min_max_scaler.min_
        
    elif(len(listCategoricalFeatures) > 0):  # Make sure there are at least one categorical value
        print("Creating Dummy features only")
        # Non scaled version of above
        if len(listNumericFeatures) > 0:
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), train[listNumericFeatures], train[strResponse]], axis = 1, join = "outer") # cbind
        else:
            df = pd.concat([pd.get_dummies(train[listCategoricalFeatures]), train[strResponse]], axis = 1, join = "outer") # cbind

        print("Encoded train file is saved to " + fileTrain)
        df.to_csv(fileTrain, index=False)  #   min_max_scaler.scale_        min_max_scaler.min_
        

    return(df)
    # Encoding end

#Dummy Encoding of categorical data, scale and center numerical data
def ScaleAndCenter_NumericOnly(train, strResponse = None):
    from sklearn import preprocessing

    # get all numeric features
    listNumericFeatures = train.select_dtypes(include=[np.number]).columns.values

    # Remove Response variable
    if strResponse != None:
        if np.issubdtype(train[strResponse].dtype, np.number):
            listNumericFeatures = list(set(listNumericFeatures) - set([strResponse]))

    listRemainingFeatures = list(set(train.columns) - set(listNumericFeatures))

    # If scale and center (in 0-1) is true
    if (len(listNumericFeatures) > 0):
        print("Scaling features")
        # get the same sclaer for both train and test
        min_max_scaler = preprocessing.MinMaxScaler()

        # do for train
        df = pd.DataFrame(min_max_scaler.fit_transform(train[listNumericFeatures]), index=train.index, columns=listNumericFeatures)

        if(len(listRemainingFeatures) > 0):
            df = pd.concat([df, train[listRemainingFeatures]], axis = 1, join = "outer") # cbind
    del(listNumericFeatures, listRemainingFeatures)

    return(df)
    # ScaleAndCenter_NumericOnly end
