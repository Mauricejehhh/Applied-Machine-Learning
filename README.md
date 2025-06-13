# Applied Machine Learning: TT100K Traffic Sign Detector & Classifier
This project implements a machine learning pipeline using Convolutional Neural Networks (CNNs) to detect and classify traffic signs in images from the TT100K dataset. The system comprises two main components: a bounding box regression model for localizing signs in full images, and a classification model that recognizes individual sign types cropped from those images. Additionally, it includes a FastAPI web server for interactive predictions and multiple scripts for training, evaluating, and visualizing both models.

Key features include:

    A localization model using a ResNet-50 backbone.

    A lightweight CNN classifier capable of recognizing 232 traffic sign types.

    A FastAPI interface for real-time inference on uploaded images.

    Python scripts to train and evaluate both classification and localization pipelines.

## The Dataset and Preprocessing
The **TT100K (Tsinghua-Tencent 100K)** dataset is a large-scale traffic sign dataset intended for research in computer vision, specifically traffic sign classification and localization. This project includes a structured and modular pipeline for managing the **TT100K data**, using a combination of preprocessing, custom PyTorch Dataset classes, and visualization utilities. Together, these components streamline the process of loading, transforming, and visualizing the dataset for both object detection and classification tasks.

### Obtaining the Dataset
To obtain the **TT100K dataset**, navigate to the official webpage: https://cg.cs.tsinghua.edu.cn/traffic-sign/. On the site, locate the 2021 dataset version and follow the download instructions. The dataset is typically provided as a compressed archive containing the image files and an annotation JSON file (commonly named *annotations_all.json*). Once downloaded, extract the contents into a working directory. Ensure that the JSON annotation file and the images maintain the original structure, as the data loading scripts rely on consistent file paths.

### Annotation handling
Annotation management is handled through the *annotations.py* script, which includes the **check_annotations()** function. This function plays a critical role in validating and preparing the dataset. It first checks for the existence of the main annotation file. If the required split files for training/validation and testing are not found, the script automatically generates them using stratified sampling via *train_test_split()* from the scikit-learn library. The default split ratio allocates 70% of the data for training, 15% for validation, and 15% for testing. To ensure reproducibility and correctness, the function also checks the consistency of image counts after the split and raises an error if any discrepancy is detected. If the split files already exist and match the expected proportions, they are reused, thereby saving time and avoiding redundant processing.

### Creating Custom Dataset Classes
The data loading logic is implemented in *dataloader.py*, where several custom PyTorch Dataset classes are defined, each tailored for specific tasks. The **TT100KDataset** class is designed for object localization. It loads full-resolution images and extracts their corresponding bounding boxes from the annotations. These bounding boxes are normalized relative to the image dimensions and returned along with class labels in a dictionary format. The class also supports transformations like resizing and normalization, applied on-the-fly to each image. This setup allows users to train localization models such as bounding box regressors directly from the raw data without requiring external preprocessing steps.

In contrast, the **TT100KSignDataset** class is optimized for classification. Rather than loading entire images, it parses the annotation data to crop individual traffic signs from their parent images. Each sample consists of a cropped image patch and its associated class label. This approach is particularly well-suited for training classifiers that need focused, background-free representations of traffic signs. The class handles image cropping and label assignment internally, enabling efficient and seamless integration into any PyTorch-based training pipeline.

Additionally, the **TT100KFRCNNDataset** class is included to support detection frameworks like Faster R-CNN. This variant returns the full image along with bounding boxes in absolute pixel coordinates and their respective class labels. It is compatible with object detection models that require exact positional information rather than normalized bounding boxes.

### Visualization Tools
Lastly, an convenient component of the project is the *dataset_visualizer.py* module, which provides tools for visual inspection of the dataset. This module leverages matplotlib to render images annotated with bounding boxes and class labels. Such visualizations are invaluable for verifying annotation correctness, understanding class distributions, and debugging issues related to data parsing. The ability to see samples with overlaid annotations greatly aids in quality assurance and can also be used to communicate insights about the dataset during presentations or collaborative work.

## The Classification Model
### The Model's Architecture
At the core of this image classification model lies the CNNClassifier, a custom convolutional neural network designed for multi-class classification tasks involving image data. The model takes input images of shape (3, 64, 64), which are RGB images resized to 64x64 pixels during preprocessing. The architecture is composed of two main building blocks: a feature extractor (convolutional layers) and a classifier (fully connected layers).

The feature extractor starts with a convolutional layer that uses 32 filters of size 3x3, followed by a ReLU activation function to introduce non-linearity, and then a max pooling layer with a kernel size of 2 to reduce spatial dimensions while preserving the most salient features. This is followed by a second convolutional layer with 64 filters, again followed by a ReLU activation and a second max pooling layer. These layers allow the network to progressively learn and abstract hierarchical features, from edges and textures in the early layers to more complex shapes and patterns in deeper layers. The two pooling operations reduce the image resolution, ultimately transforming the (64x64) input into a smaller spatial representation while increasing the depth of feature maps.

After feature extraction, the output is flattened into a one-dimensional tensor and passed through the classifier. The classifier starts with a fully connected linear layer that projects the flattened features (with size 64 × 14 × 14, assuming valid padding is not used) into a 128-dimensional hidden space. A ReLU activation introduces non-linearity, allowing the model to learn complex decision boundaries. Finally, the output is passed through a second linear layer that maps the 128 hidden features to a number of outputs equal to the number of target classes. This final layer produces raw logits, which represent unnormalized scores for each class and are typically passed through a softmax function during evaluation or loss computation.

Overall, this architecture is intentionally kept compact to support fast training and experimentation while remaining expressive enough for a wide range of classification tasks. Inspired by the CNN structure in the official PyTorch CIFAR-10 tutorial, the model balances simplicity and effectiveness.

### Transformation of the Dataset
To effectively train the model, the **TT100KSignDataset** class is used to load and manage the dataset, which includes road sign images alongside corresponding annotations. The preprocessing pipeline applied to each image ensures that all data inputs are uniform and suitable for the CNN architecture. Specifically, each image is first converted to grayscale and then replicated across three channels to meet the input format expectations of convolutional networks pre-trained on RGB images. Following that, the images are resized to a fixed dimension of 64x64 pixels, converted to PyTorch tensors, and normalized with a mean and standard deviation of 0.5 for each channel. This normalization step facilitates more stable and efficient training by centering the data.

In parallel, the **DatasetPreparer** class handles the preparation and validation of the annotation files. It uses a utility function called *check_annotations* to ensure the dataset contains valid and usable entries, returning a path to a filtered annotations file. This quality control step is crucial to prevent training disruptions and maintain high data quality.

Data loading and splitting are managed by the **DataModule** class, which integrates seamlessly with the model and training logic. It supports configurable hyperparameters such as batch size and number of folds for cross-validation, defaulting to five folds. After preprocessing, it applies *sklearn.model_selection.KFold* to generate reproducible fold splits, helping in thorough and consistent model evaluation across different subsets of data.

### Training Pipeline
The training pipeline orchestrates the entire classification process, from dataset preparation to model training, validation, and saving, ensuring a modular and reproducible workflow throughout. It starts with the **DatasetPreparer**, which verifies and cleans the raw annotation data. This preprocessing step is crucial for maintaining consistency and integrity in the input data, reducing the likelihood of training errors due to corrupted or misaligned labels. Once the annotations are validated, the pipeline uses the **DataModule** class to load and transform the images. The transformation steps: 'grayscale conversion, channel replication, resizing, tensor conversion, and normalization', standardize the image data, preparing it for model ingestion.

A central feature of this pipeline is the implementation of k-fold cross-validation using scikit-learn’s **KFold** class. This strategy divides the dataset into several subsets or "folds," enabling the model to be trained on different splits and validated on the remaining data. It not only ensures that every data point is used for both training and validation but also provides a reliable estimate of the model’s ability to generalize. The pipeline wraps each data split into a *PyTorch DataLoader*, which handles batching and shuffling to optimize GPU utilization and convergence.

Each training fold initializes a fresh instance of the **CNNClassifier**. The model is trained using the **Trainer** class, which controls the learning loop over a configurable number of epochs (typically starting with 1 for rapid testing). During each epoch, the model performs forward passes to generate predictions, calculates the loss using the *CrossEntropyLoss* function, and updates weights via backpropagation using the Adam optimizer. The validation phase follows each epoch, reporting key metrics such as accuracy and average loss. Training and validation metrics are logged and retained for post-training analysis, including convergence checks and overfitting detection.

One advanced feature is the averaging of model weights across folds. Once all folds are trained, their model states are aggregated to create an ensemble-like averaged model. This step harnesses the strengths of each individual model, smoothing out irregularities and reducing the risk of overfitting to any particular subset. The resulting model, often more stable and robust, is saved to disk and can be reloaded for future inference. Optionally, models trained on individual folds can also be saved for comparison or ensemble use.

To support interpretability and debugging, the pipeline includes optional plotting tools such as loss curves and confusion matrices. These visual diagnostics provide a deeper understanding of how the model learns and where it may struggle, particularly at the class level. Though not always enabled by default, they are highly recommended for detailed evaluations.

Altogether, the training pipeline encapsulates practices of deep learning workflows, data preprocessing, k-fold cross-validation, metric logging, model checkpointing, and performance visualization.

### Model Evaluation
Once training is complete, the model’s performance is assessed using a dedicated evaluation script. This script loads the final trained model and evaluates it on a held-out test set that was not used during training or validation. The test data undergoes the same preprocessing pipeline to ensure consistency. Evaluation metrics include cross-entropy loss and overall accuracy, offering a quantitative measure of the model’s generalization capabilities.

To contextualize performance, the script calculates a baseline accuracy, typically the inverse of the number of classes, representing the performance of a random guesser. If the trained model significantly exceeds this baseline, it demonstrates successful learning. Additionally, the script optionally generates a confusion matrix and classification report, providing class-level insights such as precision, recall, and F1-score. These tools help identify which classes are frequently misclassified, guiding future improvements to the model or dataset.

### Implementation Notes and Recommendations
The classification system is implemented with a modular architecture that enforces separation of concerns. Each component (from dataset loading and preprocessing to model training and evaluation), serves a distinct purpose, making the codebase easier to maintain, test, and extend. For example, substituting a different model architecture simply requires replacing the CNNClassifier class, while supporting a new dataset may only involve extending the dataset and transformation logic.

The classification model centers around the CNNClassifier, which includes two convolutional layers with ReLU activations and max pooling, followed by a pair of fully connected layers. This structure maps input images to 232 traffic sign categories with high efficiency. The TrainingPipeline class coordinates data ingestion, preprocessing, model training, and evaluation, supporting multi-fold validation and automated logging. After training, the *evaluate_classification_model.py* script evaluates model accuracy on unseen data and generates visual reports, ensuring the model generalizes well and enabling in-depth performance review.

## The Localization Model
### The Model's Architecture
The localization model is built around a bounding box regression framework that predicts four normalized coordinates (xmin, ymin, xmax, ymax), representing the bounding box around objects within an image. Central to the model is a pretrained **ResNet-50 backbone**, which has proven highly effective in computer vision tasks. To adapt the backbone for bounding box prediction, the final average pooling and the fully connected layers are removed. 

Moreover, the **ResNet-50 backbone** is frozen during training, meaning its weights are not updated. This approach leverages transfer learning, allowing the model to benefit from the broad, general-purpose features learned from large datasets like ImageNet, while preventing overfitting on smaller target datasets and reducing computational costs. Following feature extraction, the model applies an *Adaptive Average Pooling* layer to aggregate the spatial dimensions of the feature maps into a fixed-size representation. This pooled output is then flattened and passed through a fully connected linear layer, which predicts the four bounding box coordinates. A Sigmoid activation constrains these outputs within the [0, 1] range, normalizing the coordinates relative to the input image size. This design ensures scale invariance and facilitates consistent interpretation across varying image resolutions. Overall, this modular architecture balances deep feature extraction with a lightweight regression head tailored for accurate bounding box prediction.

### Transformation of the Dataset
To ensure effective training and evaluation, the dataset undergoes a series of transformations. Each image is first resized to a standard resolution of 224×224 pixels to match the input requirements of the ResNet-50 backbone. Next, the images are converted into *PyTorch tensors* and normalized using mean and standard deviation values derived from the ImageNet dataset ([0.485, 0.456, 0.406] for mean and [0.229, 0.224, 0.225] for standard deviation). This normalization aligns the pixel value distribution with the pretrained model’s expectations, which stabilizes and accelerates learning. An inverse normalization function is also defined, enabling the reconstruction of visually interpretable images during debugging and visualization, thus allowing predicted bounding boxes to be overlaid on original images for qualitative inspection.

### Training Pipeline 
The training pipeline employs a robust **K-Fold cross-validation** strategy, splitting the dataset into multiple folds to iteratively train and validate the model. This ensures that every data subset is used for validation at least once, providing a comprehensive assessment of the model's generalization capabilities while reducing variance in performance estimates. Before training, each image undergoes preprocessing: resizing to 224×224 pixels to fit the ResNet-50 input size, conversion to tensor format, and normalization using ImageNet mean and standard deviation values. This normalization aligns the dataset’s pixel value distribution with what the pretrained backbone expects, promoting stable and efficient training. An inverse normalization transform is also implemented to reconstruct original images for debugging and visualization purposes.

For each fold, training and validation datasets are loaded using PyTorch’s **DataLoader** with a custom collate function to handle batch data properly. A new instance of the bounding box regression model is initialized and moved to the available device (GPU if possible). The model is trained using the *Smooth L1 loss* function, which is well-suited for regression tasks involving bounding boxes because it is less sensitive to outliers compared to the *Mean Squared Error loss*. The Adam optimizer, configured with a learning rate of 0.001, facilitates efficient convergence by adapting the learning rate during training.

The training loop involves forward passing batches of images through the model, calculating the *Smooth L1 loss* against ground truth bounding boxes, backpropagating errors, and updating the regression head’s weights while keeping the backbone fixed. Training and validation losses are tracked and printed regularly to monitor progress and detect potential issues like overfitting. Upon completing each fold, the trained model is saved to disk. These saved models are later combined into an ensemble by averaging their predictions during inference, which reduces variance and enhances prediction stability.

In addition to quantitative evaluation, the pipeline includes visualization tools that overlay predicted and ground truth bounding boxes on sample images. This visual inspection helps qualitatively assess the model’s localization accuracy and identify common success or failure cases. Loss curves are also plotted for each fold, offering insight into the training dynamics across epochs. Together, these elements create a comprehensive, efficient, and interpretable training framework for bounding box regression.


### Model Evaluation
Following training, the ensemble model is evaluated on a held-out test set. Each predicted bounding box is compared to the ground truth using the *Intersection over Union (IoU)* metric, a standard measure in object detection that quantifies the overlap between predicted and actual boxes. For context, a random baseline is computed by generating boxes with random plausible coordinates. This comparison ensures that the model’s performance reflects meaningful learning rather than chance.

The evaluation results demonstrate the localization model’s effectiveness, consistently outperforming the random baseline and confirming its ability to learn spatial features relevant to object localization. The final mean IoU score provides a single scalar summary of model accuracy across the test set.

### Implementation Notes and Recommendations
In short, the combination of the frozen ResNet-50 backbone with a lightweight regression head forms a robust and efficient localization model architecture. The cross-validation training framework, along with ensemble averaging, enhances reliability and generalization. Visualization tools and statistical evaluation methods provide comprehensive insight into model performance, supporting thorough analysis and interpretation of results.

## The R-CNN Model 
### The Model's Architecture
The core of the object detection system is the Faster R-CNN architecture, which is a two-stage detector combining both region proposal and classification tasks efficiently. The backbone of the model is ResNet-50, a deep convolutional neural network known for its residual connections that help alleviate the vanishing gradient problem and allow training of very deep networks. This backbone extracts rich, hierarchical feature representations from input images. To enhance detection at multiple scales, especially important for objects of varying sizes such as road signs, the backbone is augmented with a *Feature Pyramid Network* (FPN). The FPN constructs a multi-scale feature pyramid by combining low-level, high-resolution features with high-level, semantically rich features, enabling the detector to recognize both small and large objects effectively.

Following feature extraction, the model employs a *Region Proposal Network* (RPN) that scans the feature maps to generate candidate object bounding boxes called proposals. These proposals are then fed into the second stage of the detector where *RoI* (Region of Interest) pooling extracts fixed-size feature representations for each proposal. These features are passed to fully connected layers that perform classification and bounding box regression, refining both the class labels and the localization of the proposals.

To tailor the model for the specific TT100K road sign detection task, the original classification head of Faster R-CNN is replaced with a custom FastRCNNPredictor that matches the number of classes in the dataset, including the background class. This replacement allows the model to leverage pretrained features learned from large-scale datasets like COCO, while adapting the final layers to recognize the unique categories present in the road sign dataset. This architecture balance between pretrained backbone and a task-specific prediction head facilitates faster convergence and improved generalization, especially when the available training data is limited.

### Transformation of the Dataset
To ensure effective learning, the dataset undergoes a transformation process that standardizes input images and applies augmentations to improve model generalization. The base transformation pipeline converts images to PyTorch tensors. During training, additional augmentations are applied, including random horizontal flips, slight color jitter (altering brightness, contrast, and saturation), and small random rotations. These augmentations expose the model to diverse viewing conditions, enhancing its robustness to real-world variations.

For evaluation and inference, only the conversion to tensors is applied to ensure consistency and avoid introducing randomness that could distort predictions. This clear distinction between training and evaluation transformations helps maintain the integrity of validation metrics.

### Training Pipeline
The training process for the Faster R-CNN model is designed to ensure robust and reliable learning through a comprehensive *K-Fold cross-validation* pproach. This method divides the dataset into multiple folds, typically five, allowing the model to be trained and validated on different subsets of the data. For each fold, a fresh instance of the Faster R-CNN model with a **ResNet-50** backbone and a custom prediction head is initialized. This model is trained using the Adam optimizer with a carefully selected learning rate, updating only those parameters that require gradients, primarily the classification and bounding box regression heads.

The training pipeline applies a series of data augmentations such as random horizontal flips, color jitter, and small rotations during the training phase. These augmentations enhance the model’s ability to generalize by exposing it to varied visual conditions. Each epoch involves iterating over batches of images and their corresponding annotations, computing the losses that include classification loss, box regression loss, and other components, which are then summed to obtain the total training loss.

To manage potential issues like GPU out-of-memory errors, the training loop includes error handling that skips problematic batches and clears the CUDA cache, maintaining training stability. After each epoch, the model is evaluated on a held-out validation set for that fold, calculating the average validation loss to monitor overfitting and performance improvements. Throughout training, loss metrics for both training and validation phases are logged and plotted, providing valuable visual feedback on learning progress.

At the end of training each fold, the model state is saved. Once all folds are trained, their weights are averaged to produce a final, more generalized model that smooths out fold-to-fold variations. This averaging approach, combined with the careful training setup and augmentations, helps the model achieve better overall performance and robustness when deployed in real-world scenarios.

### Model Evaluation
Evaluation is performed on a fixed subset of the dataset using the averaged model. The evaluation script loads the model with the correct architecture and trained weights, sets it to evaluation mode, and applies it to images in batches. Predictions and ground truths are compared using two metrics: *classification accuracy* and *mean Area of Intersection* (AoI), a variant of *Intersection over Union* (IoU).

Classification accuracy measures how often the predicted class matches the ground truth, given that the predicted and ground truth boxes overlap sufficiently (IoU > 0.5). Mean AoI quantifies the spatial alignment between predicted and ground truth bounding boxes, indicating how well the model localizes objects. These metrics provide insight into both the model’s detection and localization capabilities.

### Implementation Notes and Recommendations
The implementation of the Faster R-CNN pipeline incorporates several practical considerations to enhance training stability, performance, and generalization. The pretrained ResNet-50 backbone is kept unfrozen, allowing the entire network to be fine-tuned to the specific characteristics of the target dataset. However, in scenarios with limited data or training time, selectively freezing early convolutional layers can accelerate training and help reduce the risk of overfitting. To address occasional GPU memory overflows, which can occur with large or high-resolution images, the training loop includes safeguards that skip problematic batches and explicitly clear the CUDA cache as needed. This ensures training continues smoothly without manual intervention.

Data augmentation is applied with care, as overly aggressive transformations can distort object geometry or semantics and negatively affect detection performance. To improve generalization, model averaging is applied across multiple cross-validation folds. This technique combines the weights of independently trained models, assuming the same architecture and training configuration, to produce a more stable and robust ensemble.

Due to the high memory requirements of Faster R-CNN, especially with large input images and complex architectures, the batch size is limited to a small number such as four to remain within GPU constraints. Although small batches may introduce noise into gradient estimates, they are often necessary for such tasks. For evaluation, a subset of up to 300 images is used to balance computational efficiency and effectiveness. However, a more thorough assessment should include evaluation on the full validation set or an external test set. Overall, this pipeline represents a carefully designed approach to training and evaluating Faster R-CNN for road sign detection, incorporating best practices such as cross-validation, model checkpointing, and comprehensive metric tracking.

## Fast API
The main.py file is the entry point to an interactive web server built using FastAPI, which provides a simple interface for testing the trained models. This server exposes an endpoint that allows users to upload images from the TT100K dataset and receive predictions for both the bounding box location of traffic signs (localization) and their class labels (classification).

When the server starts, it loads two pretrained models:

    A localization model (based on a frozen ResNet-50) that outputs the coordinates of a predicted bounding box.

    A classification model that predicts the class label of a cropped traffic sign image.

For inference, the server performs two separate tasks:

    It passes the uploaded image through the localization model to predict the most prominent bounding box in the scene.

    It then uses ground-truth bounding boxes from TT100K annotations (not the predicted one) to crop out individual signs and run them through the classification model, returning predicted class labels.

This separation ensures the classification model gets clean, labeled input, while the localization model demonstrates its ability to generalize on unseen images. The API returns both outputs as a JSON response, and includes automatic Swagger documentation via FastAPI at /docs.

To start the API, first make sure to have installed: uvicorn, fastapi[standard], python-multipart, torch, torchvision and pillow. On top of that, the dataset folder 'tt100k' needs to be put under 'data/'. To launch the API, make sure to be in the main working directory (Applied-Machine-Learning/) and in the terminal run the command 'uvicorn main:app'. In the terminal, it will output an IP address, such as 'http://127.0.0.1:8000/', which you can either 'ctrl + click' or paste in a webbrowser. Automatically, it will redirect to the Swagger UI with the automated documentation. 

Here, you find a short description about the API, as well as the documentation for two functions: 'get / root' and 'post / predict'. 'get / root' is only a function to immediately redirect the FastAPI to its Swagger UI / Documentation. This is to give quick and easy access to the API documentation and interactive interface instead of manually having to switch to the documentation each time. 

The 'post / predict' function is the interactive interface where we can upload an image. For now, as we are using two individual base models, please upload an image of the actual dataset. The function will use our trained localization model to predict its bounding boxes. After that, it will take the ground truth bounding boxes from the dataset and use the trained classification model to predict the type of traffic sign. 

The response / result will show below, with the first bounding box showing the pixels (xmin, ymin, xmax, ymax) of the original image (as our baseline model only localizes the first bounding box). Below that, it will show all predictions of traffic sign types from all bounding boxes from the ground truths of the image in the dataset. The prediction (e.g. p150) corresponds to the traffic sign in the dataset 'marks' file.


## Requirements
The code is structered to run smoothly on python version 3.12.3, all other dependencies and libraries are listed in the requirements.txt file with the specific versions used. This ensures a consistent environment for training, evaluation, and deployment.
To run the models or specific files, just write in your terminal 'python -m folder_name.file_name' and it should run smoothly.

## 📁 Project Structure
```bash
├───data_storage # Stores the dataset, does get showcased in the repository
├───models  # Stores the trained models as .pth
├───road_sign_detection
│   ├───data
│       ├───annotations.py
│       ├───dataset_loader.py
│       └───dataset_visualizer.py
│   ├───evaluation
│       ├───evaluation_of_classification.py
│       ├───evaluation_of_localization.py
│   ├───models
│       ├───classification_base_model.py
│       └───localization_base_model.py
│   └───training
│       ├───train_classification_model.py
│       ├───train_localization_model.py
│       ├───train_faster_r_cnn_model.py
├───test
│   ├───test_data
│       ├───test_annotations.py
│       └───test_dataset_loader.py
│   └───test_models
│       ├───test_classification_base_model.py
│       └───test_localization_base_model.py
├───.gitignore
├───.gitattributes
├───.pre-commit-config.yaml
├───main.py
├───demo.py
├───requirements.txt
├───README.md
```
***Made by: Maurice Theo Meijer (s5480604), David van Wuijkhuijse(s5592968), Emily Heugen (s5587042), Yannick van Maanen (s5579082)***
        ***Group 27***