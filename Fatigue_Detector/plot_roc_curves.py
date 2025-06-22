import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from data import WiderFaceDetection, preproc, cfg_mnet, cfg_slim, cfg_rfb
from data.wider_face import detection_collate
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_model(network, weight_path):
    """Load trained model"""
    if network == "mobile0.25":
        cfg = cfg_mnet
        model = RetinaFace(cfg=cfg)
    elif network == "slim":
        cfg = cfg_slim
        model = Slim(cfg=cfg)
    elif network == "RFB":
        cfg = cfg_rfb
        model = RFB(cfg=cfg)
    else:
        raise ValueError("Unsupported network type")
    
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model, cfg

def get_predictions(model, dataloader):
    """Get predictions and ground truth matching train.py processing"""
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        # Add progress bar
        for images, targets in tqdm(dataloader, desc="Processing batches", unit="batch"):
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
            
            # Forward pass
            out = model(images)
            
            # Get confidence scores from classification output
            # out[0] shape: [batch_size, num_priors, 2]
            # We take the face class probability (index 1)
            conf_scores = torch.sigmoid(out[0][:, :, 1])
            
            # Get ground truth labels from targets
            batch_scores = []
            batch_labels = []
            for i, target in enumerate(targets):
                # Each target contains [x1,y1,x2,y2,landmarks...,class]
                # Class is at last position (1 for face, 0 for background)
                valid_indices = target[:, -1] != -1  # Only consider valid annotations
                batch_labels.append(target[valid_indices, -1].cpu().numpy())
                
                # Match scores to valid annotations
                num_valid = valid_indices.sum().item()
                if num_valid > 0:
                    batch_scores.append(conf_scores[i, :num_valid].cpu().numpy())
            
            all_scores.append(np.concatenate(batch_scores))
            all_labels.append(np.concatenate(batch_labels))
    
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    
    # Print label statistics
    print(f"Label statistics - Positive: {(labels == 1).sum()}, Negative: {(labels == 0).sum()}")
    
    if (labels == 0).sum() == 0:
        print("Warning: No negative samples found in labels")
    
    return scores, labels

def plot_roc_curves():
    """Plot ROC curves for three models"""
    # Define model paths
    model_paths = {
        "mobile0.25": "weights/mobilenet0.25_Final.pth",
        "slim": "weights/slim_Final.pth",
        "RFB": "weights/RFB_Final.pth"
    }
    
    # Load training dataset (since validation labels are not available)
    train_dataset = WiderFaceDetection(
        "./data/widerface/widerface/train/label.txt",
        preproc(cfg_mnet['image_size'], (104, 117, 123))
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4,
        collate_fn=detection_collate
    )
    
    plt.figure(figsize=(10, 8))
    
    # Evaluate each model
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"Model {name} not found at {path}")
            continue
            
        print(f"Evaluating {name} model...")
        model, _ = load_model(name, path)
        scores, labels = get_predictions(model, train_loader)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Format plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save and show plot
    plt.savefig('roc_curves.png')
    plt.show()

if __name__ == "__main__":
    plot_roc_curves()
