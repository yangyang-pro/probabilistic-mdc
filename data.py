import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import torch

from ast import literal_eval
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io.image import read_image


class IMGDataset(Dataset):
    def __init__(self, img_dir, img_files, img_labels=None):
        self.img_dir = img_dir
        self.img_files = img_files
        self.img_labels = torch.LongTensor(img_labels) if img_labels is not None else None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = read_image(img_path)
        rescaled_img = self.transform(img)
        if self.img_labels is not None:
            return rescaled_img, self.img_labels[idx]
        else:
            return rescaled_img


class IMGDataLoader:
    def __init__(self, img_dir, xml_dir, target_domains):
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        xml_files = sorted(os.listdir(xml_dir))
        img_files = []
        all_labels = []
        for i, xml_file in enumerate(xml_files):
            xml_path = os.path.join(xml_dir, xml_file)
            xml_tree = ET.parse(xml_path)
            root = xml_tree.getroot()
            labels = np.zeros(len(target_domains), dtype=np.int32)
            is_valid_sample = True
            for object in root.findall('object'):
                label_name = object.find('name').text
                for j, target in enumerate(target_domains):
                    target_domain = target_domains[target]
                    if label_name in target_domain:
                        label_idx = target_domain.index(label_name) + 1
                        if labels[j] != 0 and labels[j] != label_idx:
                            is_valid_sample = False
                            break
                        labels[j] = label_idx
                if not is_valid_sample:
                    break
            if not is_valid_sample:
                continue
            img_name = root.find('filename').text
            img_files.append(img_name)
            all_labels.append(labels)
        self.labels = np.array(all_labels)
        self.img_files = np.array(img_files)
        self.label_domains = {str(i): np.arange(len(target_domains[target]) + 1)
                              for i, target in enumerate(target_domains)}

    def get_slices(self, node, parents, configuration):
        node_idx = int(node)
        parent_indices = [int(parent) for parent in parents]
        indices = np.logical_and.reduce(self.labels[:, parent_indices] == list(configuration), axis=-1)
        return self.img_files[indices], self.labels[indices][:, node_idx]

    def create_local_data_loaders(self, local_files, local_labels, batch_size, shuffle=True):
        local_img_dataset = IMGDataset(img_dir=self.img_dir, img_files=local_files, img_labels=local_labels)
        local_data_loader = DataLoader(dataset=local_img_dataset, batch_size=batch_size, shuffle=shuffle)
        return local_data_loader

    def create_img_datasets(self):
        img_dataset = IMGDataset(img_dir=self.img_dir, img_files=self.img_files, img_labels=self.labels)
        return img_dataset


class TabularDataLoader:
    def __init__(self,
                 csv_path=None, arff_path=None, n_labels=None,
                 mat_path=None):
        if csv_path is not None and arff_path is not None and n_labels is not None:
            print('Read data from the arff file ...')
            csv_file = open(csv_path, 'r')
            arff_file = open(arff_path, 'r')
            arff_lines = arff_file.readlines()
            continuous_feature_names = []
            discrete_attr_names = []
            discrete_attr_domains = {}
            for line in arff_lines:
                if '@attribute' in line or '@ATTRIBUTE' in line:
                    attributes = line.split()
                    if attributes[-1] == 'numeric':
                        continuous_feature_names.append(attributes[-2])
                    else:
                        discrete_attr_names.append(attributes[-2])
                        discrete_attr_domains[attributes[-2]] = literal_eval(attributes[-1])
                elif '@data' in line or '@DATA' in line:
                    break
            print('Continuous features:', continuous_feature_names)
            discrete_feature_names = discrete_attr_names[0:-n_labels]
            label_names = discrete_attr_names[-n_labels:]
            print('Discrete features:', discrete_feature_names)
            print('Labels:', label_names)
            df = pd.read_csv(csv_file)

            self.continuous_features = df.loc[:, continuous_feature_names].values
            self.discrete_features = df.loc[:, discrete_feature_names].values
            self.discrete_feature_names = [chr(i + 97) for i in range(len(discrete_feature_names))]
            self.discrete_feature_domains = {k: discrete_attr_domains[n] for k, n in zip(self.discrete_feature_names,
                                                                                         discrete_feature_names)}
            self.labels = df.loc[:, label_names].values
            self.label_names = [str(i) for i in range(len(label_names))]
            self.label_domains = {k: discrete_attr_domains[n] for k, n in zip(self.label_names, label_names)}
        elif mat_path is not None:
            print('Read data from the mat file ...')
            mat_data = loadmat(mat_path)
            orig_data = mat_data['data']['orig'][0][0]
            norm_data = mat_data['data']['norm'][0][0]
            labels = mat_data['target']
            # matlab's index starts from 1
            continuous_feature_indices = mat_data['data_type']['c'][0][0][0] - 1
            non_ordinal_feature_indices = mat_data['data_type']['d_wo_o'][0][0] - 1
            ordinal_feature_indices = mat_data['data_type']['d_w_o'][0][0] - 1
            binary_feature_indices = mat_data['data_type']['b'][0][0] - 1
            discrete_feature_indices = np.concatenate((non_ordinal_feature_indices, ordinal_feature_indices,
                                                       binary_feature_indices), axis=None)

            self.labels = labels - 1
            self.label_names = [str(i) for i in range(labels.shape[1])]

            label_domains = {}
            for i in range(labels.shape[1]):
                label_domains[str(i)] = np.unique(labels[:, i]) - 1
            self.label_domains = label_domains

            if discrete_feature_indices.shape[0] != 0:
                self.continuous_features = norm_data[:, continuous_feature_indices]
                self.discrete_features = orig_data[:, discrete_feature_indices]
                self.discrete_feature_names = [chr(i + 97) for i in range(len(discrete_feature_indices))]
                discrete_feature_domains = {}
                for i, feature_idx in enumerate(discrete_feature_indices):
                    discrete_feature_domains[chr(i + 97)] = np.unique(orig_data[:, feature_idx])
                self.discrete_feature_domains = discrete_feature_domains
            else:
                self.continuous_features = norm_data
                self.discrete_feature_names = None
                self.discrete_feature_domains = None
        else:
            self.continuous_features = None
            self.discrete_features = None
            self.discrete_feature_names = None
            self.discrete_feature_domains = None
            self.labels = None
            self.label_names = None
            self.label_domains = None

    def split_train_test(self, test_size=0.25):
        train_data_loader = TabularDataLoader()
        test_data_loader = TabularDataLoader()
        indices = np.arange(len(self.continuous_features))
        train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=True, random_state=0)
        train_data_loader.continuous_features = self.continuous_features[train_indices]
        train_data_loader.labels = self.labels[train_indices]
        train_data_loader.label_names = self.label_names
        train_data_loader.label_domains = self.label_domains
        test_data_loader.continuous_features = self.continuous_features[test_indices]
        test_data_loader.labels = self.labels[test_indices]
        test_data_loader.label_names = self.label_names
        test_data_loader.label_domains = self.label_domains

        if self.discrete_feature_domains:
            train_data_loader.discrete_features = self.discrete_features[train_indices]
            train_data_loader.discrete_feature_names = self.discrete_feature_names
            train_data_loader.discrete_feature_domains = self.discrete_feature_domains
            test_data_loader.discrete_features = self.discrete_features[test_indices]
            test_data_loader.discrete_feature_names = self.discrete_feature_names
            test_data_loader.discrete_feature_domains = self.discrete_feature_domains
        else:
            train_data_loader.discrete_features = None
            train_data_loader.discrete_feature_names = None
            train_data_loader.discrete_feature_domains = None
            test_data_loader.discrete_features = None
            test_data_loader.discrete_feature_names = None
            test_data_loader.discrete_feature_domains = None
        return train_data_loader, test_data_loader

    def get_slices(self, node, parents, configuration):
        if self.discrete_feature_domains:
            return self.__get_slices(node, parents, configuration)
        else:
            return self.__get_slices_no_discrete_features(node, parents, configuration)

    def __get_slices(self, node, parents, configuration):
        node_name_idx = self.label_names.index(node)
        label_name_indices = []
        label_configuration = []
        discrete_feature_name_indices = []
        discrete_feature_configuration = []
        for i, parent in enumerate(parents):
            if parent in self.label_names:
                label_name_indices.append(self.label_names.index(parent))
                label_configuration.append(configuration[i])
            if parent in self.discrete_feature_names:
                discrete_feature_name_indices.append(self.discrete_feature_names.index(parent))
                discrete_feature_configuration.append(configuration[i])
        instance_label_indices = np.logical_and.reduce(self.labels[:, label_name_indices] == label_configuration,
                                                       axis=-1)
        instance_discrete_feature_indices = np.logical_and.reduce(self.discrete_features[:,
                                                                  discrete_feature_name_indices] ==
                                                                  discrete_feature_configuration,
                                                                  axis=-1)
        indices = np.logical_and(instance_label_indices, instance_discrete_feature_indices)
        return self.continuous_features[indices], self.labels[indices][:, node_name_idx]

    def __get_slices_no_discrete_features(self, node, parents, configuration):
        node_name_idx = self.label_names.index(node)
        label_name_indices = []
        label_configuration = []
        for i, parent in enumerate(parents):
            label_name_indices.append(self.label_names.index(parent))
            label_configuration.append(configuration[i])
        indices = np.logical_and.reduce(self.labels[:, label_name_indices] == label_configuration, axis=-1)
        return self.continuous_features[indices], self.labels[indices][:, node_name_idx]

    def create_sub_data_loader(self, indices):
        data_loader = TabularDataLoader()
        data_loader.continuous_features = self.continuous_features[indices]
        data_loader.discrete_features = self.discrete_features[indices] if self.discrete_feature_domains else None
        data_loader.discrete_feature_names = self.discrete_feature_names
        data_loader.discrete_feature_domains = self.discrete_feature_domains
        data_loader.labels = self.labels[indices]
        data_loader.label_names = self.label_names
        data_loader.label_domains = self.label_domains
        return data_loader
