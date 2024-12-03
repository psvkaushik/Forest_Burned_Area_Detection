from datasets import load_dataset
from datasets import concatenate_datasets, DatasetDict
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
import torch.nn.functional as F  # Correct import for interpolate
from sklearn.metrics import f1_score, accuracy_score, classification_report
from Prithvi import MaskedAutoencoderViT

dataset = load_dataset("DarthReca/california_burned_areas")
def combine_datasets(dataset_dict):
    datasets = [dataset_dict[key] for key in dataset_dict.keys()]
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset
combined_dataset = combine_datasets(dataset)


# Efficient resize and normalization for multi-channel data
def custom_transform(image, target_size=(224, 224), normalize=True):
    # Convert to tensor and normalize
    image = torch.tensor(image, dtype=torch.float32)  # Assuming input is NumPy array
    if normalize:
        image = (image - image.mean()) / image.std()  # Normalize globally

    # Resize using PyTorch's interpolate function (efficient for multi-channel data)
    image = image.unsqueeze(0)  # Add batch dimension for interpolation
    image = interpolate(image, size=target_size, mode="bilinear", align_corners=False)
    return image.squeeze(0)  # Remove batch dimension

# Preprocess function for batched datasets
def preprocess_data(batch):
    images = [
        custom_transform(image, target_size=(224, 224)) for image in batch["post_fire"]
    ]
    masks = [
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        for mask in batch["mask"]
    ]
    return {
        "images": torch.stack(images),  # Shape: (batch_size, C, H, W)
        "masks": torch.stack(masks)     # Shape: (batch_size, 1, H, W)
    }

# from tqdm import tqdm

# for i, processed_batch in tqdm(enumerate(dataset), total=len(dataset)):
#     _ = preprocess_data(processed_batch)
    
# test_batch = combined_dataset[:10]  # First 10 samples from the first split
# preprocessed = preprocess_data(test_batch)
# print(preprocessed["images"].shape, preprocessed["masks"].shape)


# Apply preprocessing to the dataset
processed_dataset = combined_dataset.map(preprocess_data, batched=True, batch_size=16)  # Adjust batch_size as needed
# Initialize the model
model = MaskedAutoencoderViT(img_size=224, patch_size=16, in_chans=3)

# Function to process predictions into binary format
def process_predictions(pred, threshold=0.5):
    """
    Convert raw predictions to binary masks based on the threshold.
    Args:
        pred (torch.Tensor): Predicted values (logits or probabilities) of shape (B, C, H, W).
        threshold (float): Threshold to binarize predictions.
    Returns:
        torch.Tensor: Binary predictions.
    """
    pred = torch.sigmoid(pred)  # Convert logits to probabilities
    return (pred > threshold).long()

# Metrics storage
all_labels = []
all_preds = []

# Loop through the dataset
for batch in processed_dataset:
    # Convert lists to tensors
    images = torch.tensor(batch["images"], dtype=torch.float32)  # Convert to tensor
    masks = torch.tensor(batch["masks"], dtype=torch.long)  # Convert to tensor
    
    # Debugging input tensor size
    print(f"Original image tensor size: {images.size()}")
    print(f"Original mask tensor size: {masks.size()}")
    
    # Ensure images have correct dimensions (B, C, H, W)
    images = images.unsqueeze(0)
    projected_images = torch.nn.Conv2d(512, 3, kernel_size=1)(images)  # Reduce channels to 3
    projected_images = F.interpolate(
        projected_images,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )
    print(f"Processed image tensor size: {projected_images.size()}")  # Debugging size
    
    projected_images=projected_images.unsqueeze(2)

    # Ensure masks have the correct shape (B, C, H, W)
    reshaped_masks = masks.squeeze(-1)  # Remove last dimension (W=1)
    reshaped_masks=reshaped_masks.unsqueeze(0)
    reshaped_masks = F.interpolate(
        reshaped_masks.float(),  # Convert to float for interpolation
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).long()  # Convert back to long after interpolation
    print(f"Reshaped mask tensor size: {reshaped_masks.size()}")  # Debugging size

    # Forward pass
    loss, pred, mask = model(projected_images)

    # Process predictions
    pred_binary = process_predictions(pred)  # (B, C, H, W)

    # Flatten predictions and masks for metric calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    mask_flat = reshaped_masks.view(-1).cpu().numpy()

    # Append for overall metrics
    all_preds.extend(pred_flat)
    all_labels.extend(mask_flat)

    # Print batch loss
    print(f"Loss: {loss.item()}")

# Compute metrics after all batches
f1 = f1_score(all_labels, all_preds, average="binary")
accuracy = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
