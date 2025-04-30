import torch
from PIL import Image
from torchvision import transforms
from models.rgb2thermal import RGB2Thermal
import os

# Modeli yükle
model = RGB2Thermal()
model.load_state_dict(torch.load("rgb2thermal.pth"))
model.eval()

# Ön işleme transformasyonu
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Görüntü boyutunu modele uygun hale getir
    transforms.ToTensor()  # Görüntüyü tensor'a çevir
])

# RGB fotoğraflarının bulunduğu dizin (kendi RGB fotoğraflarınızı buraya koyun)
input_dir = "path/to/your/rgb_images"
# Termal çıktılar için çıktı dizini
output_dir = "generated_thermal_images"
os.makedirs(output_dir, exist_ok=True)

# RGB fotoğraflarını al ve modelden termal görüntü üret
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    
    # Fotoğrafı aç
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)  # Tensor formatına çevir
    
    # Modelden termal görüntüyü üret
    with torch.no_grad():  # Hesaplama sırasında gradyanlara gerek yok
        output = model(input_tensor)[0][0]  # Modelin çıktısını al (grayscale)

    # Çıktıyı PIL formatına dönüştür
    output_img = transforms.ToPILImage()(output)
    
    # Çıktıyı kaydet
    output_img.save(os.path.join(output_dir, img_name))

print("Termal görüntüler başarıyla üretildi!")
