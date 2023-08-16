Klasifikacija audio fajlova iz UrbanSounds8k dataseta primenom Resnet18 arhitekture.

## Opis
Sistem se sastoji iz sledecih skripti:
- `settings.py`
    - U ovoj skripti se vrsi kreiranje foldera  MFCC i Mel (u svakom od njih se kreiraju folderi fold_i (i je u opsegu [1,10] )) u kojima ce se cuvati PNG grafici treinranja, modeli i istorija treniranja (history). 
- `AudioPrep.py`
    - Skripta za preprocesiranje audio fajlova i ekstrakciju deskriptora (moze se birati izmedju MFCC i MelSpectrogram-a).
- `SoundDS.py`
    - Skripta za pomocnim funkcijama za ucitavanje audio dataseta, i ekstrakciju deskriptora iz njegovih audio fajlova.
- `data.py`
    - Skripta za ucitavanje UrbanSounds8k audio dataseta, i ekstrakciju deskriptora iz njega.    
- `model.py`
    - Skripta za kreiranje Resnet18 modela.
- `train.py`
    - Skripta za treniranje i evaluaciju modela pokrenutim nad UrbanSounds8k dataset-om.

## Koriscenje
Prvo je potrebno skinuti UrbanSounds8k dataset koji sadrzi foldere:
    - /fold1
    - /fold2
    - /fold3
    - /fold4
    - /fold5
    - /fold6
    - /fold7
    - /fold8
    - /fold9
    - /fold10
    - /UrbanSound8k.csv
Struktura direktorijuma se treba postaviti na sledeci nacin:
- ./
    - /fold1
    - /fold2
    - /fold3
    - /fold4
    - /fold5
    - /fold6
    - /fold7
    - /fold8
    - /fold9
    - /fold10
    - /PycharmProject
    - /UrbanSound8k.csv