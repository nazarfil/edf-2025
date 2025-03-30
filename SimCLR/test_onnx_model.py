import argparse
import onnx
import onnxruntime as ort
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(num_images=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_images, shuffle=False)
    images, labels = next(iter(loader))
    return images, labels

def test_onnx_model(onnx_model_path, num_images=10):
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name

    images, labels = load_cifar10(num_images)
    images_np = images.cpu().numpy().astype(np.float32)
    
    outputs = sess.run(None, {input_name: images_np})[0]
    
    print("Embeddings shape:", outputs.shape)
    for i, emb in enumerate(outputs):
        norm = np.linalg.norm(emb)
        print(f"Image {i} - label: {labels[i].item()} - embedding norm: {norm:.4f}")
    
    if outputs.shape[0] >= 2:
        emb1 = outputs[0] / (np.linalg.norm(outputs[0]) + 1e-8)
        emb2 = outputs[1] / (np.linalg.norm(outputs[1]) + 1e-8)
        cosine_sim = np.dot(emb1, emb2)
        print(f"Cosine similarity entre image 0 et 1: {cosine_sim:.4f}")
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test ONNX model on CIFAR10 images")
    parser.add_argument("onnx_model_path", help="Chemin vers le fichier ONNX")
    parser.add_argument("--num_images", type=int, default=10, help="Nombre d'images à tester")
    args = parser.parse_args()
    
    test_onnx_model(args.onnx_model_path, args.num_images)
