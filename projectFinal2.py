import os, numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json, random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

with open("C:/Users/USER/OneDrive/Belgeler/auair2019annotations/annotations.json", "r") as file:
    tümEtiketler = json.load(file)

etiketHaritası = {}
for e in tümEtiketler['annotations']:
    resimAd = e['image_name'] 
    bbox = e.get('bbox',[])   

    if len(bbox) > 0:
        etiketHaritası[resimAd] = bbox

imageKlasör = "C:/Users/USER/images"     
tümResimler = os.listdir(imageKlasör)

def yolDöndürme(resimListesi):
    görseller = []

    for isim in resimListesi:
        yol = os.path.join(imageKlasör, isim)
        mevcutBboxlar = etiketHaritası.get(isim, [])

        for box in mevcutBboxlar:
            görseller.append({
                'yol': yol,
                'class': box['class'],
                'crop': (int(box['left']), int(box['top']), int(box['left']) + int(box['width']), int(box['top']) + int(box['height']))
            })    

        if random.random() <= 0.5:
            try:
                with Image.open(yol) as img:
                    genislik, yukseklik = img.size
                
                if genislik >= 100 and yukseklik >= 100:
                    randLeft = random.randint(0, genislik - 100)
                    randTop = random.randint(0, yukseklik - 100)
                    
                    görseller.append({
                        'yol': yol,
                        'class': 'Background', 
                        'crop': (randLeft, randTop, randLeft + 100, randTop + 100) 
                    })
            except Exception:
                continue  

    return görseller

kirpilmisResimKlasoru = "C:/Users/USER/Kırpılmış Resimler"
os.makedirs(kirpilmisResimKlasoru, exist_ok=True)

def gorselleriKirp(Liste, mod):
    yeniListe = []
    print(f"{mod} görselleri kırpılıyor...")

    for i, item in enumerate(Liste):
        kayitAdi = f"{mod}_{i}.jpg"
        kayitYolu = os.path.join(kirpilmisResimKlasoru, kayitAdi)

        if not os.path.exists(kayitYolu):
            try:
                with Image.open(item['yol']) as img:
                    img = img.convert('RGB')
                    crop = item['crop']
                    if crop[2] <= crop[0] or crop[3] <= crop[1]:
                        cropped = img.resize((224,224))
                    else:
                        cropped = img.crop(crop).resize((224,224))
                    cropped.save(kayitYolu, quality=90)
            except Exception:
                continue

        yeniListe.append({
            'yol': kayitYolu,
            'class': item['class']
        })
    return yeniListe    
        
class AUAIRDataset(Dataset):
    def __init__(self, girdi, etiket):
        self.girdi = girdi
        self.etiket = etiket

    def __len__(self):
        return len(self.girdi)

    def __getitem__(self, idx):
        item = self.girdi[idx]
        try:
            with Image.open(item['yol']) as img:
                
                arr = np.array(img, dtype=np.float32) / 255.0
                tensorİmg = torch.from_numpy(arr.transpose((2, 0, 1)))
                etiket = torch.tensor(self.etiket[str(item['class'])], dtype=torch.long)
                return tensorİmg, etiket
            
        except Exception:
            return self.__getitem__(0)
    
def modelEgit(konfigürasyon, train, val, agırlık, sinifSayisi):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--> MODELİN ÇALIŞTIĞI CİHAZ: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  

    inFeatures = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(konfigürasyon['dropout']),
        nn.Linear(inFeatures, sinifSayisi)
    )
    model = model.to(device)

    kriter = nn.CrossEntropyLoss(weight=agırlık.to(device))
    optimizer = optim.Adam(model.parameters(), lr=konfigürasyon['lr'])

    patience = konfigürasyon['patience']
    pcounter = 0
    enİyiValLoss = float('inf')
    enİyiModelAgırlıkları = None
    enİyiAccuracy = 0.0

    for epoch in range(10):

        model.train()
        trainLoss = 0.0

        for batchx, batchy in train:
            batchx, batchy = batchx.to(device), batchy.to(device)

            optimizer.zero_grad()
            output = model(batchx)

            loss = kriter(output, batchy)
            loss.backward()
            optimizer.step()

            trainLoss = trainLoss + loss.item() * batchx.size(0)

        trainLoss = trainLoss/len(train.dataset)

        model.eval()
        valLoss = 0.0
        tahminler = []
        etiketler = []

        with torch.no_grad():
            for batchx, batchy in val:
                batchx, batchy = batchx.to(device), batchy.to(device)
                output = model(batchx)
                loss = kriter(output, batchy)
                valLoss = valLoss + loss.item() * batchx.size(0)

                deger, aitSınıf = torch.max(output, 1)
                tahminler.extend(aitSınıf.cpu().numpy())
                etiketler.extend(batchy.cpu().numpy())

        valLoss = valLoss/len(val.dataset)
        valAccuracy = accuracy_score(etiketler, tahminler)

        print(f"Epoch {epoch+1:02d} -> Train Loss: {trainLoss:.4f} , Validation Loss: {valLoss:.4f} , Validation Accuracy: {valAccuracy:.4f}")

        if valLoss < enİyiValLoss:
            enİyiValLoss = valLoss
            enİyiAccuracy = valAccuracy
            enİyiModelAgırlıkları = model.state_dict()
            pcounter = 0
        else:
            pcounter +=1
            if pcounter >= patience:
                print(f"Early stopping tetiklendi! Epoch {epoch+1} de eğitim durduruldu")
                break                

    return enİyiModelAgırlıkları, enİyiAccuracy

if __name__ == '__main__':

    nesneliResimler = []
    for n in tümResimler:
       if n in etiketHaritası:
          nesneliResimler.append(n)

    print(f"toplam {len(tümResimler)} resimden {len(nesneliResimler)} tanesinde nesne bulundu")   

    random.seed(3)
    random.shuffle(nesneliResimler)

    toplamResimSayısı = len(nesneliResimler)
    trainDataSınır = int(toplamResimSayısı * 0.7)
    validationDataSınır = int(toplamResimSayısı * 0.85)

    trainResimleri = nesneliResimler[:trainDataSınır]
    validationResimleri = nesneliResimler[trainDataSınır:validationDataSınır]
    testResimleri = nesneliResimler[validationDataSınır:]

    print(f"Resim dağılımı -> Train: {len(trainResimleri)}, Validation: {len(validationResimleri)}, Test: {len(testResimleri)}")

    trainResim = yolDöndürme(trainResimleri)
    valResim = yolDöndürme(validationResimleri)
    testResim = yolDöndürme(testResimleri)

    trainResim = gorselleriKirp(trainResim, "train")
    valResim = gorselleriKirp(valResim, "val")
    testResim = gorselleriKirp(testResim, "test")

    sınıflar = sorted(list(set([str(k['class']) for k in trainResim])))
    sınıfIndex = {}

    for index, sınıf in enumerate(sınıflar):
        sınıfIndex[sınıf] = index

    sınıfSayıları = Counter([sınıfIndex[str(k['class'])] for k in trainResim])
    toplam = len(trainResim)

    agirlik = [np.sqrt(toplam/sınıfSayıları[i]) for i in range(len(sınıflar))]
    agirlikTensor = torch.FloatTensor(agirlik).to('cuda' if torch.cuda.is_available() else 'cpu')

    trainDataset = AUAIRDataset(trainResim, sınıfIndex)
    valDataset = AUAIRDataset(valResim, sınıfIndex)
    testDataset = AUAIRDataset(testResim, sınıfIndex)

    hiperparametreler = [
        #{'batchsize': 32, 'lr': 0.0001, 'patience': 5, 'dropout': 0.3},
        #{'batchsize': 64, 'lr': 0.0001, 'patience': 5, 'dropout': 0.2},

        {'batchsize': 64, 'lr': 0.00008, 'patience': 5, 'dropout': 0.15},
        {'batchsize': 64, 'lr': 0.0001, 'patience': 5, 'dropout': 0.10}
    ]

    enİyiSkor = -1
    enİyiKonfigürasyon = None
    enİyiModelAgırlıkları = None

    print("--- Resnet18 Hiperparametre Optimizasyon Süreci Başlıyor ---")
    for index, konfigürasyon in enumerate(hiperparametreler):
        print(f"\nKonfigürasyon {index+1} eğitiliyor: {konfigürasyon}")

        trainLoad = DataLoader(trainDataset, batch_size=konfigürasyon['batchsize'], shuffle=True, num_workers=4)
        valLoad = DataLoader(valDataset, batch_size=konfigürasyon['batchsize'], shuffle=False, num_workers=4)

        weights, valAccuracy = modelEgit(konfigürasyon, trainLoad, valLoad, agirlikTensor, len(sınıflar))
        print(f"Validation Doğruluğu: {valAccuracy:.4f}")

        if valAccuracy > enİyiSkor:
            enİyiSkor = valAccuracy
            enİyiKonfigürasyon = konfigürasyon
            enİyiModelAgırlıkları = weights

    print("\n--- Optimizasyon Tamamlandı: En İyi Model Test Ediliyor ---")
    print(f"En İyi Konfigürasyon: {enİyiKonfigürasyon}")

    torch.save(enİyiModelAgırlıkları, "EnİyiResnet18Model.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finalModel = models.resnet18(weights=None)
    inFeatures = finalModel.fc.in_features
    finalModel.fc = nn.Sequential(
        nn.Dropout(enİyiKonfigürasyon['dropout']),
        nn.Linear(inFeatures, len(sınıflar))
    )
    finalModel = finalModel.to(device)

    finalModel.load_state_dict(torch.load("EnİyiResnet18Model.pt"))
    finalModel.eval()

    testLoad = DataLoader(testDataset, batch_size=enİyiKonfigürasyon['batchsize'], shuffle=False, num_workers=4)
    tahminler = []
    testEtiketler = []
        
    with torch.no_grad():
        for batchx, batchy in testLoad:
            batchx = batchx.to(device)
            output = finalModel(batchx)
            deger, aitSınıf = torch.max(output, 1)
            tahminler.extend(aitSınıf.cpu().numpy())
            testEtiketler.extend(batchy.cpu().numpy())

    orijinalİsimler = ['human','car','truck','van','motorbike','bicycle','bus','trailer','Background']

    print("\n --- Resnet18 Modelinin Test Seti Sınıflandırma Raporu ---")
    print(classification_report(testEtiketler, tahminler, target_names=orijinalİsimler))

    print("--- Resnet18 Confusion Matrix ---")
    print(confusion_matrix(testEtiketler, tahminler))