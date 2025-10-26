# Opinnäytetyö: Avoimen lähdekoodin kielimallien tarkkuus ja luotettavuus STT-käännöksissä eri ympäristöissä.

## Yleiskatsaus
Tämä arkisto sisältää koodin ja konfiguraatiot opinnäytetyön käytännön osuuteen, jossa mitataan ennalta koulutettujen avoimen lähdekoodin kielimallien suorituskykyä **puheentunnistuksen (ASR)** ja **konekäännöksen (MT)** tehtävissä.

Tätä arkistoa käytetään opinnäytetyön aikana kerätyn audio-aineiston ja kääntämistä varten luotujen käännöstekstien prosessointiin ja tulosten automaattiseen arviointiin.

Erityishuomio on kiinnitetty mallien **kvantisointiin** rajoitetun VRAM-muistin (ASUS RTX 2060 6BG) ympäristössä.

---

## Laitteisto ja vaatimukset

### Laitteisto (Testiympäristö)

Laitteisto koostuu kierrätetyistä ja varastosta löytyvistä komponenteista, eli yhtään uutta komponenttia ei ole tätä työtä varten hankittu. Kielimallien suorituskyky paranisi uudemmilla ja tehokkaammilla komponenteilla.

| Komponentti           | Tiedot | Huomioitavaa |
|:----------------------| :--- | :--- |
| **GPU**               | ASUS RTX 2060 DUAL OC EVO **6GB** | Vaatii suurempien mallien **4-bittisen (INT4) kvantisoinnin** |
| **CPU**               | Intel Core-i5-660 | |
| **RAM**               | Kingston 8 GB Fury Beast DDR4 | |
| **Käyttöjärjestelmä** | Linux (Ubuntu/LXQt etäkäyttöön) | `bitsandbytes` vaatii Linux-ympäristön |

### Ohjelmistovaatimukset

Projekti käyttää **Python 3.12**-versiota. Kaikki tarvittavat kirjastot on lueteltu tiedostossa `requirements.txt`.

**Kvantisoinnin avainkirjastot:**
* `accelerate`
* `bitsandbytes`

**Asennus:**
1. **Kloonaa arkisto:**
    ```bash
    git clone [https://github.com/PanTheRandomness/asr-mt-thesis.git](https://github.com/PanTheRandomness/asr-mt-thesis.git)
    cd asr-mt-thesis
    ```
2. **Luo ja aktivoi virtuaaliympäristö**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3. **Asenna riippuvuudet**
    ```bash
    pip install -r requirements.txt
    ```
---
## Käyttöohjeet

### 1. Mallien ajo
Käytä kansiossa `asr/` ja `mt/` olevia skriptejä mallin suorittamiseksi.

**Esimerkki (ASR - Whisper Large v2):**
```bash
python asr/whisper_medium.py
```
#### Kvantisointi (INT4)
Suuret mallit (kuten Whisper Large, NLLB ja BLOOM) ladataan automaattisesti 4-bittisesti (`utils/model_loader.py` kautta) VRAM-muistin säästämiseksi.

### 2. Arviointi (Evaluation)
Suorita automaattinen arviointi skriptillä `evaluation.py` malliajon jälkeen.
```bash
python evaluation.py
```
Tämä skriptii laskee projektissa valitut metriikat (WER, WIL, METEOR, TER, COMET, MEANT) ja tallentaa tulokset.

---
## Automaattiset arviointimetriikat
**ASR-metriikat**
* **WER** (Word Error Rate)
* **WIL** (Word Information Lost, WWER sijaan)

**MT-metriikat**
* **METEOR** (Metric of Evaluation of Translation with Explicit Ordering)
* **TER** (Translation Edit Rate)
* **COMET** (Cross-Lingual Optimised Metric for Evaluation of Translation)
* **MEANT**