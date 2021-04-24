import pandas as pd

def extract_val_data(data="features/features_Kmeans_metadata_and_label_onehot.csv", write=False):
    """" read features and names of validation images and extract the vaidation data
    """
    # read data and names of validation images
    all_data = pd.read_csv(data)
    val_names = pd.read_csv("features/groundtruth_val.csv")     # hardcoded because we use always the same validation images
    # find all rows given in val_names and delete them in all_data
    val_data = all_data.loc[all_data["image"].isin(val_names["image"].values)]
    all_data = pd.concat([all_data, val_data]).drop_duplicates(keep=False)

    # sort val_data like in val_names (needed for submission)
    # looks a bit hacky but works
    val_data = val_data.set_index("image")
    val_data = val_data.reindex(index=val_names["image"])
    val_data = val_data.reset_index()
    if write:
        # save to file so we don't have to do this all the time
        all_data.to_csv("features/features_train.csv", index=False)
        val_data.to_csv("features/features_val.csv", index=False)

    # return the data
    print("extracted datasets of size: ")
    print(all_data.shape, val_data.shape)
    return all_data, val_data

if __name__ == "__main__":
    extract_val_data(write=True)