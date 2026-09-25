# AU-AIR Veri Seti ile İHA Görüntülerinde CNN Mimarileri Kullanarak Nesne Sınıflandırma

Bu proje, AU-AIR veri setindeki insansız hava aracı (İHA/Drone) görüntüleri üzerinde yer alan nesneleri, özel olarak tasarlanmış bir Evrişimli Sinir Ağı (DinamikCNN) ve aktarmalı öğrenme (transfer learning) tabanlı önceden eğitilmiş ResNet18 mimarisi ile sınıflandırmayı amaçlamaktadır. 

**AU-AIR Veri Seti Görselleri:** https://drive.google.com/open?id=1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ

**AU-AIR Veri Seti Etiketleri:** https://drive.google.com/open?id=1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA

## 🏗️ Proje Mimarisi ve İş Akışı

Sistem temel olarak dört aşamadan oluşmaktadır:

1. **Veri Ön İşleme (Kırpma ve Yeniden Boyutlandırma):** `annotations.json` dosyasındaki koordinatlar kullanılarak her bir nesne resimden kırpılmış ve 224x224 piksel boyutuna yeniden ölçeklendirilmiştir. 
2. **Arka Plan (Background) Sınıfının Eklenmesi:** Modelin boş yolları veya binaları nesne sanmasını engellemek için, nesne içermeyen bölgelerden rastgele kesitler alınarak "Background" adında yeni bir sınıf veri setine dahil edilmiştir. Zaman tasarrufu sağlamak adına tüm kırpılan görseller eğitim öncesi diske kaydedilmiştir.
3. **Model Mimarisi - Yöntem 1 (Dinamik CNN):** PyTorch kullanılarak sıfırdan geliştirilen bu mimari; girdi tensörünü (3 x 224 x 224) 3 ardışık bloktan (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d) geçirerek kanal sayısını sırasıyla 32, 64 ve 128'e genişletir. Çıkan 100.352 boyutlu öznitelik vektörü Dropout (0.35) katmanından geçirilerek tam bağlantılı (Linear) sınıflandırıcıya beslenir.
4. **Model Mimarisi - Yöntem 2 (ResNet18 ile Transfer Learning):** `torchvision.models` kütüphanesinden ImageNet verisiyle önceden eğitilmiş ResNet18 modeli çağırılmıştır. Sadece ağın sonundaki tam bağlantılı katman (fc), 9 sınıf (8 nesne + 1 arka plan) çıkışı verecek şekilde yeniden uyarlanmıştır.

## 🧠 Optimizasyon Stratejileri ve Çözümler

*   **Sınıf Dengesizliği Çözümü:** Veri setindeki "Car" (Araba) sınıfının %68.9'luk ezici çoğunluğa sahip olması nedeniyle, sınıfların frekansları sayılarak ters karekök formülüyle ağırlıklandırılmış ve bu ağırlıklar doğrudan `CrossEntropyLoss` kayıp fonksiyonuna entegre edilerek azınlık sınıfların etkisi artırılmıştır.
*   **Aşırı Öğrenmeyi (Overfitting) Engelleme:** DinamikCNN modelinde her evrişim katmanı sonrasında uygulanan `BatchNorm2d` ve sınıflandırıcı öncesindeki `Dropout` katmanları ile ağın ezber yapması engellenerek genelleme yeteneği artırılmıştır.

## 📊 Veri Seti Dağılımı

Modelin aynı drone karesindeki nesneleri hem eğitim hem de test süreçlerinde görerek ezberlemesini önlemek amacıyla resim bazlı bölünme uygulanmıştır:
*   **Eğitim (Train):** 22.976 resim (nesne)
*   **Geçerleme (Validation):** 4.923 resim (nesne)
*   **Test:** 4.924 resim (nesne)

## 🏆 Deneysel Sonuçlar ve Performans Kıyası

Her iki model için de hiperparametre arama uzayı (Batch Size, Öğrenme Oranı, Patience, Dropout) kurularak en iyi konfigürasyonlar tespit edilmiş ve `.pt` formatında kaydedilmiştir.

*   **Dinamik CNN Başarısı:** Piksellerin 2 boyutlu uzamsal hiyerarşisini (receptive field) koruyarak kendi özniteliklerini sıfırdan öğrenen bu model, nihai test setinde **%86.00** genel doğruluk (accuracy) elde ederek projenin önceki aşamasındaki MLP mimarisini geride bırakmıştır.
*   **En İyi Model (ResNet18):** ImageNet üzerindeki görsel tecrübesini aktarmalı öğrenme (transfer learning) ile sürece dahil eden ResNet18 modeli, **%91.00** ile tüm projeler arasındaki en yüksek test doğruluğuna ulaşmıştır. Azınlık sınıflarının (Motorbike, Bicycle) özelliklerini daha iyi ayırt etmiş ve Macro F1-Skorunu 0.58'den 0.73'e yükseltmiştir.

## ⚙️ Kurulum ve Çalıştırma

**Gereksinimler:**
Projenin çalışması için `numpy`, `Pillow`, `scikit-learn`, `torch` (PyTorch) ve `torchvision` kütüphanelerinin yüklü olması gerekmektedir.

1. Depoyu bilgisayarınıza klonlayın.
2. Gerekli bağımlılıkları sisteminize kurun.
3. DinamikCNN mimarisini eğitmek için `projectFinal.py`, ResNet18 mimarisini eğitmek için `projectFinal2.py` dosyasını çalıştırın.
4. **Çıkarım (Inference):** Model eğitimini beklemeden sonuçları görmek isterseniz, `Proje_demo.ipynb` (veya `Proje_demo2.ipynb`) dosyasını Jupyter Notebook üzerinden açarak, tek bir görüntü üzerinden anlık sınıf tahmini ve softmax güven skoru üretebilirsiniz.

> **⚠️ Önemli Not (Veri Boyutu):**
> Görüntü boyutlandırma sonucunda oluşan kırpılmış resimler ve ağırlık modelleri (`.pt`), boyut sınırları nedeniyle bu depoya yüklenmemiştir. Eğitim dosyaları çalıştırıldığında bu veriler bilgisayarınızda lokal olarak otomatik üretilip kaydedilecektir.
> İndireceğiniz dosyaların isimleri koddakinden farklı olabilir. O isimlere göre kodu güncellemeyi unutmayın.
