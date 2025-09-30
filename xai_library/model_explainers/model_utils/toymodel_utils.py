import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
import os

def load_and_configure_model(model_name, weight_path, weight_file, num_classes, device):
    # Get model from torchvision library
    model = torchvision.models.get_model(model_name,
                                     weights=None,
			                         weights_backbone=None,
			                         num_classes=num_classes)
    # Load the trained model
    checkpoint = torch.load(os.path.join(weight_path, weight_file))
    model.load_state_dict(checkpoint)
    model.to(device)
    return model

def preprocess_image(image_path, image_file, device, resize_height=320, resize_width=320):
    # Define the transformation for inputs
    transform = T.Compose([
        T.Resize((resize_height, resize_width)),
        T.ToTensor(),
    ])
    # Load image
    image = Image.open(os.path.join(image_path, image_file))
    # Apply the transformation to the image and add an extra dimension (for batch)
    image = transform(image).unsqueeze(0)
    # Move the image to the device
    image = image.to(device)
    return image

def perform_inference_and_visualize(model, image, threshold=0.5, show_image=True, save_image=True, output_file="test_image_output.png"):
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        prediction = model(image)
    # Get best detection
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    # Find the index of the bounding box with the highest score over the threshold
    max_score_index = torch.argmax(scores)
    max_score = scores[max_score_index]

    if max_score > threshold:
        max_score_box = boxes[max_score_index]
        # Draw the bounding box on the image
        transform_to_PIL = T.ToPILImage()
        img = transform_to_PIL(image[0])
        draw = ImageDraw.Draw(img)
        draw.rectangle([(max_score_box[0], max_score_box[1]), (max_score_box[2], max_score_box[3])], outline="white", width=3)
        # Show the image if requested
        if show_image:
            img.show()
        # Save the image if requested
        if save_image:
            img.save(output_file)
    else:
        print("There is no detection with score > threshold=", threshold)