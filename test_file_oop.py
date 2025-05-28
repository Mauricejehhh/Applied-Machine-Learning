import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from project_name.models.classification_base_model import CNNClassifier
from project_name.data.dataset_loader import TT100KSignDataset


class DatasetPreparer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.annos_path = os.path.join(data_root,
                                       'annotations_all.json')
        self.filtered_path = os.path.join(data_root,
                                          'filtered_annotations.json')
        self.ids_path = os.path.join(data_root,
                                     'train',
                                     'ids.txt')

    def prepare(self):
        if not os.path.exists(self.filtered_path):
            print('Creating a new .json file for training ids.')
            with open(self.ids_path, 'r') as f:
                ids = set(line.strip() for line in f)

            with open(self.annos_path, 'r') as f:
                annos = json.load(f)

            filtered_imgs = {
                img_id: img_data for img_id,
                img_data in annos['imgs'].items() if img_id in ids
            }

            train_annotations = {
                'types': annos['types'],
                'imgs': filtered_imgs
            }

            with open(self.filtered_path, 'w') as f:
                json.dump(train_annotations,
                          f,
                          indent=4)

        return self.filtered_path


class DataModule:
    def __init__(self, annotation_path, data_root, batch_size=32):
        self.annotation_path = annotation_path
        self.data_root = data_root
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def setup(self):
        dataset = TT100KSignDataset(self.annotation_path,
                                    self.data_root,
                                    self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset,
                                            [train_size, val_size])
        return (
            DataLoader(train_data,
                       batch_size=self.batch_size,
                       shuffle=True),
            DataLoader(val_data,
                       batch_size=self.batch_size,
                       shuffle=True),
            len(dataset.annotations['types'])
        )


class Trainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, d_loader, ep_idx):
        self.model.train()
        running_loss = 0.0

        for i, (inputs,
                labels) in enumerate(tqdm(d_loader,
                                          desc=f"Epoch {ep_idx+1} Training")):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:
                print(f'\nBatch {i}: Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

    def validate(self, data_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

        avg_loss = running_loss / len(data_loader)
        print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss


class TrainingPipeline:
    def __init__(self, data_root, model_save_path, epochs=1):
        self.data_root = data_root
        self.model_save_path = model_save_path
        self.epochs = epochs
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def run(self):
        preparer = DatasetPreparer(self.data_root)
        annotation_path = preparer.prepare()

        data_module = DataModule(annotation_path, self.data_root)
        train_loader, val_loader, num_classes = data_module.setup()

        model = CNNClassifier(num_classes)
        trainer = Trainer(model, self.device)

        for epoch in range(self.epochs):
            trainer.train(train_loader, epoch)
            trainer.validate(val_loader)

        torch.save(model.state_dict(), self.model_save_path)
        print(f'Model saved to: {self.model_save_path}')


if __name__ == "__main__":
    root_path = os.path.join(os.getcwd(), 'data_storage', 'tt100k_2021')
    model_path = os.path.join(os.getcwd(), 'models', 'model.pth')
    pipeline = TrainingPipeline(root_path, model_path, epochs=1)
    pipeline.run()
