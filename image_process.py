from PIL import Image
from torchvision import transforms

def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # Open the image
    # img = Image.open(image_path)

    # Convert to RGB mode if necessary
    # if img.mode != 'RGB':
    #   img = img.convert('RGB')

    # Define a sequence of image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)

    # Convert the tensor to a NumPy array
    img_array = img_tensor.numpy()

    return img_array