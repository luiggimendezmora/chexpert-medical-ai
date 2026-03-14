import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import torchvision.transforms as transforms

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

def visualize_gradcam(image_path, class_idx, model, device, save_path=None):
    class_name = PATHOLOGIES[class_idx]

    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    target_layers = [model.features[-1]]
    targets: list = [ClassifierOutputTarget(class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(
        img_array.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original X-Ray", fontsize=13)
    axes[0].axis('off')
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM — {class_name}", fontsize=13)
    axes[1].axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()  