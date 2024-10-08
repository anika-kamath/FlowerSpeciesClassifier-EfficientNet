from model_load import load_checkpoint
from image_process import process_image
from map_labels import get_label_mapping
import torch.nn.functional
import torch

import argparse

def predict(image_path, checkpoint_path, category_names, topk=5, gpu=False):

    if gpu and not torch.cuda.is_available():
        print("GPU is not available. Using CPU for prediction.")
        gpu = False

    model, _, _ = load_checkpoint(checkpoint_path, gpu)

    if gpu:
        model.cuda()

    img_tensor = torch.tensor(process_image(image_path))
    img_tensor = img_tensor.unsqueeze(0) # add batch dim
    if gpu:
        img_tensor = img_tensor.cuda()
    
    model.eval()
        
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # Get top K predictions
        top_k_probs, top_k_classes = probabilities.topk(topk, dim=1)
    
    # Convert results to lists and adjust class indices (if needed)
    top_probs = top_k_probs.squeeze().tolist()
    top_classes = top_k_classes.squeeze().tolist()
    if top_classes[0] != 0:  # Check if class indexing starts from 0
      top_classes = [c for c in top_classes]  # Adjust if necessary


    predicted_class = ""
    # max_val = -1

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    cls_names = []

    for cl in top_classes:
        cls_names.append(int(idx_to_class[cl]))

    print(cls_names)
    

    print("Top classes are =>", cls_names)
    print("Top probabilities are =>", top_probs)

    print("Predicted class is =>", get_label_mapping(category_names)[predicted_class])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using the pre-trained model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint.")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predicted classes.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the category names.")
    parser.add_argument("--gpu", action="store_true", help="Set to Use GPU.")

    args = parser.parse_args()
    
    predict(
        image_path=args.image_path, 
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        gpu=args.gpu,
        category_names=args.category_names 
    )

    

