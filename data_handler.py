import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import blob_dog

import image_features as features


class DataHandler:
    """
    Manages the data loading, feature extraction, and caching pipeline for
    histopathological image analysis.

    Attributes:
        X_raw (numpy.ndarray): The 2D array of extracted features for all images.
        Y (numpy.ndarray): The 1D array of corresponding class labels.
        dataset_path (str): The full file path where the .npz file is saved/loaded.
    """

    def __init__(self, data_csv_path, images_dir, output_dir='./', feature_list=None, extract_features=False,
                 num_samples=10000):
        """
        Initializes the DataHandler and automatically determines whether to load
        existing data or extract features from scratch.

        Args:
            data_csv_path (str): Path to the CSV file containing image IDs and labels.
            images_dir (str): Path to the directory containing the raw .tif images.
            output_dir (str, optional): Directory where the .npz dataset will be saved. Defaults to './'.
            feature_list (list of str, optional): List of specific feature sets to extract
                (e.g., ['dog', 'color', 'glcm']). Defaults to all available features.
            extract_features (bool, optional): If True, forces the extraction of features from
                scratch, overwriting any existing saved data. Defaults to False.
            num_samples (int, optional): The maximum number of images to process. Defaults to 10000.
        """
        self.data_csv_path = data_csv_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.num_samples = num_samples

        # Default to all features if none are provided
        self.feature_list = feature_list if feature_list is not None else ['dog', 'color', 'glcm', 'lbp', 'lbglcm',
                                                                           'glrlm', 'sfta']

        self.labels_df = pd.read_csv(self.data_csv_path)
        self.training_files = sorted(os.listdir(self.images_dir))

        self.X_raw = None
        self.Y = None

        # Build the dynamic file path based on the requested features
        os.makedirs(self.output_dir, exist_ok=True)
        features_suffix = "_".join(self.feature_list)
        self.dataset_path = os.path.join(self.output_dir, f'extracted_features_{features_suffix}.npz')


        if extract_features:
            print("Flag 'extract_features' is True. Forcing extraction from images")
            self.build_dataset()
        else:
            if os.path.exists(self.dataset_path):
                print(f"Flag 'extract_features' is False. Loading existing data from {self.dataset_path}")
                self.load_dataset()
            else:
                print(
                    f"Warning: 'extract_features' is False, but {self.dataset_path} does not exist. Overriding flag and building dataset")
                self.build_dataset()

    def build_dataset(self):
        """
        Iterates over the image directory, extracts the specified mathematical
        features, concatenates them into a single dataset, and saves the result
        to a compressed .npz file.

        The features extracted are strictly determined by the `feature_list`
        provided during class initialization.
        """
        X, Y = [], []
        num_samples_to_process = min(self.num_samples, len(self.training_files))

        for i in tqdm(range(num_samples_to_process), desc="Extracting features"):
            file_name = self.training_files[i]
            img_id = file_name.rsplit(".", 1)[0]
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]

            image_path = os.path.join(self.images_dir, file_name)
            img_bgr = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            features_to_concat = []

            if 'dog' in self.feature_list:
                features_to_concat.append(features.extract_advanced_stats(img_gray,
                                                                          blob_dog(img_gray, min_sigma=2, max_sigma=15,
                                                                                   threshold=0.1)))
            if 'color' in self.feature_list:
                features_to_concat.append(features.extract_color_stats(img_bgr))
            if 'glcm' in self.feature_list:
                features_to_concat.append(features.extract_glcm_features(img_gray))
            if 'lbp' in self.feature_list:
                features_to_concat.append(features.extract_lbp_features(img_gray))
            if 'lbglcm' in self.feature_list:
                features_to_concat.append(features.extract_lbglcm_features(img_gray))
            if 'glrlm' in self.feature_list:
                features_to_concat.append(features.extract_glrlm_features(img_gray))
            if 'sfta' in self.feature_list:
                features_to_concat.append(features.extract_sfta_features(img_gray, n_levels=3))

            combined_fd = np.concatenate(features_to_concat)

            X.append(combined_fd)
            Y.append(label)

        self.X_raw = np.array(X)
        self.Y = np.array(Y)
        print(f"\nRaw X shape: {self.X_raw.shape}")
        print(f"Final Y shape: {self.Y.shape}")

        # Save automatically to the predefined path
        np.savez_compressed(self.dataset_path, X=self.X_raw, Y=self.Y)
        print(f"Dataset successfully saved to: {self.dataset_path}")

    def load_dataset(self):
        """
        Bypasses the image extraction phase and loads pre-extracted features
        directly into memory from the saved .npz file.

        Raises:
            FileNotFoundError: If the .npz file cannot be found at the expected path.
        """
        data = np.load(self.dataset_path)
        self.X_raw = data['X']
        self.Y = data['Y']
        print(f"Loaded X shape: {self.X_raw.shape}")
        print(f"Loaded Y shape: {self.Y.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract and save features from H&E images.')
    parser.add_argument('--csv', type=str, default='/kaggle/input/histopathologic-cancer-detection/train_labels.csv',
                        help='Path to labels CSV')
    parser.add_argument('--images', type=str, default='/kaggle/input/histopathologic-cancer-detection/train',
                        help='Path to image directory')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save output arrays')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to process')
    parser.add_argument('--features', nargs='+', default=['dog', 'color', 'glcm', 'lbp', 'lbglcm', 'glrlm', 'sfta'],
                        help='List of features to extract')
    args = parser.parse_args()

    print(f"Starting Standalone Extraction for features: {args.features}")

    handler = DataHandler(args.csv, args.images, output_dir=args.output_dir, feature_list=args.features,
                          extract_features=True, num_samples=args.samples)