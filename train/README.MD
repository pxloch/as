## YAPI
```
rgb2thermal/
├── dataset/
│   ├── train/
│   │   ├── rgb/
│   │   │   ├── frame_0001.png
│   │   │   └── ...
│   │   └── thermal/
│   │       ├── frame_0001.png
│   │       └── ...
│   └── val/
│       ├── rgb/
│       └── thermal/
├── models/
│   └── rgb2thermal.py
├── train.py
├── inference.py
├── utils.py
├── requirements.txt
└── README.md
```

`python train.py` Modeli eğit rgb2thermal.pth çıktısı olucak
`python inference.py` kendi datasetini termale çevir


Çok iyi soru — proje yapısındaki `dataset/` klasörlerine ne koyacağınız kritik. Sana adım adım ve örnekli olarak açıklayayım:

---

## ✅ 1. `dataset/train/rgb/`

Bu klasöre, elinizdeki videodan çıkartılmış **renkli (RGB)** görüntüleri koyacaksınız.

### Örnek:
```
dataset/train/rgb/
├── frame_0001.png
├── frame_0002.png
├── ...
```

📌 Bu görüntüler, **termal karşılığı olan** renkli kareler olmalı (yani eşleşmiş veri).

---

## ✅ 2. `dataset/train/thermal/`

Bu klasöre, aynı anın çekilmiş **grayscale termal görüntülerini** koyacaksınız.

⚠️ İsimler, `rgb/` klasöründekilerle birebir aynı olmalı!

### Örnek:
```
dataset/train/thermal/
├── frame_0001.png
├── frame_0002.png
├── ...
```

📌 Her `frame_000X.png` dosyasının hem `rgb/` hem de `thermal/` versiyonu olmalı.

---

## ✅ 3. `dataset/val/rgb/` ve `dataset/val/thermal/`

Bu klasörler, modelin doğrulama (validation) sırasında görmediği veriyle test edebilmeniz için var. Aynı yapıyı küçük bir subset ile burada da kurabilirsiniz.

Eğer elinizde yeterince veri yoksa, bu klasörler boş da olabilir veya `train/` verisinin %10'unu buraya ayırabilirsiniz.

---

### ✅ Örnek Dosya Eşleşmesi

| `dataset/train/rgb/frame_0010.png` | eşleşir | `dataset/train/thermal/frame_0010.png` |
|------------------------------------|---------|----------------------------------------|
| `dataset/train/rgb/frame_0011.png` | eşleşir | `dataset/train/thermal/frame_0011.png` |

---

## 📌 Ekstra İpuçları

- Dosya adlarını **senkronize etmek** çok önemli.
- 1 karede sorun varsa (eksik ya da bozuk), dataloader hata verir.
- Görsellerin boyutu (resolution) önemli değil, çünkü `transforms.Resize((256, 256))` zaten eğitim sırasında hepsini eşitliyor.

---

İstersen sana `ffmpeg` ile bu RGB ve termal videolardan senkron kareleri nasıl çıkaracağını da gösterebilirim. Yardımcı olmamı ister misin?
