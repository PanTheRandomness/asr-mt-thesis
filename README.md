# Avoimen lähdekoodin kielimallien tarkkuus ja luotettavuus STT-käännöksissä eri olosuhteissa.
Opionnäytetyö: Pan Tuononen, Tietotekniikan Insinööti (AMK), Savonia AMK, 2025.
---
## Yleiskatsaus
Tämä arkisto sisältää koodin ja konfiguraatiot opinnäytetyön käytännön osuuteen, jossa mitataan ennalta koulutettujen avoimen lähdekoodin kielimallien suorituskykyä **puheentunnistuksen (ASR)** ja **konekäännöksen (MT)** tehtävissä.

Tutkimuksessa ajetaan valitut mallit suomen, englannin ja ranskan kielisillä audio- ja tekstiaineistoilla, ja tuloksia arvioidaan automaattisilla metriikoilla.

Erityishuomio on kiinnitetty mallien **kvantisointiin** rajoitetun VRAM-muistin (ASUS RTX 2060 6BG) ympäristössä.

## Kielimallit ja tehtävät

### Puheentunnistus (ASR)
ASR-mallit suorittavat **Speech-to-Text (STT)** -transkription.

| Malli                              | Arkkitehtuuri    | Koko            | Huomioitavaa                                                                                                                                       |
|:-----------------------------------|:-----------------|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Whisper large v2**               | Transformer      | Suuri           | Käytetään ensisijaisesti **transkriptioon (STT)**.                                                                                                 |
| **Wav2Vec 2.0 XLRS-53 (3 mallia)** | Wav2Vec2         | Suuri (n. 300M) | **Kielikohtainen hienosäätö** parhaan suorituskyvyn saavuttamiseksi.                                                                               |
| **Parakeet-tdt-0.6b-v3**           | Transducer (TDT) | 600M            | **NeMo-malli**, optimoitu korkeaan suorituskykyyn.                                                                                                 |

#### Wav2Vec 2.0 XLSR-53 -mallit (Yksityiskohtainen erittely)
| Mallinimi | Kieli | Hugging Face ID | Huomioitavaa |
|:---|:---|:---|:---|
| **XLSR-53 Finnish** | Suomi | `jonatasgrosman/wav2vec2-large-xlsr-53-finnish` | Suomelle optimoitu malli. |
| **Wav2Vec2 960h** | Englanti | `facebook/wav2vec2-large-960h` | Englanille optimoitu malli. |
| **XLSR-53 French** | Ranska | `facebook/wav2vec2-large-xlsr-53-french` | Ranskalle optimoitu malli. |

### Konekäännös (MT)
MT-mallit suorittavat **Text-to-Text** -käännöksen.

| Malli                       | Koko            | Kielituki  | Huomioitavaa                                    |
|:----------------------------|:----------------|:-----------|:------------------------------------------------|
| **NLLB-200-distilled-600M** | 600M            | 196 kieltä | Destilloitu versio, soveltuu 6GB VRAM-muistiin. |
| **BLOOM**                   | (Useita kokoja) | 46 kieltä  | Käytetään MT-tehtävässä.                        |
| **Helsinki-NLP Opus-MT**    | Useita          | fi, en, fr | Useita kapeamman domainin malleja (6 kpl).      |

#### Helsinki-NLP Opus-MT-mallit (Yksityiskohtainen erittely)
| Mallinimi                                         | Käännössuunta            | Huomioitavaa                             | Hugging Face URL                                                                   |
|:--------------------------------------------------|:-------------------------|:-----------------------------------------|:-----------------------------------------------------------------------------------|
| **opus-mt-fr-en**                                 | Ranska → Englanti        |                                          | `https://huggingface.co/Helsinki-NLP/opus-mt-fr-en`                                |
| **opus-mt-en-fr**                                 | Englanti → Ranska        |                                          | `https://huggingface.co/Helsinki-NLP/opus-mt-en-fr`                                |
| **opus-mt-en-fi**                                 | Englanti → Suomi         |                                          | `https://huggingface.co/Helsinki-NLP/opus-mt-en-fi`                                |
| **opus-mt-fi-en**                                 | Suomi → Englanti         |                                          | `https://huggingface.co/Helsinki-NLP/opus-mt-fi-en`                                |
| **opus-mt-fi-fr**                                 | Suomi → Ranska           |                                          | `https://huggingface.co/Helsinki-NLP/opus-mt-fi-fr`                                |
| **opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu**  | Ranska, Englanti → Suomi | Ainoa malli Ranska → Suomi-käännöksiin.  | `https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-fiu` |
---

## Laitteisto ja vaatimukset

### Laitteisto (Testiympäristö)

Laitteisto koostuu kierrätetyistä ja varastosta löytyvistä komponenteista, eli yhtään uutta komponenttia ei ole tätä työtä varten hankittu. Kielimallien suorituskyky paranisi uudemmilla ja tehokkaammilla komponenteilla.

| Komponentti           | Tiedot                            | Huomioitavaa                                                  |
|:----------------------|:----------------------------------|:--------------------------------------------------------------|
| **GPU**               | ASUS RTX 2060 DUAL OC EVO **6GB** | Vaatii suurempien mallien **4-bittisen (INT4) kvantisoinnin** |
| **CPU**               | Intel Core-i5-6600                |                                                               |
| **RAM**               | Kingston 8 GB Fury Beast DDR4     |                                                               |
| **Käyttöjärjestelmä** | Linux (Ubuntu/LXQt etäkäyttöön)   | `bitsandbytes` vaatii Linux-ympäristön                        |

### Ohjelmistovaatimukset

Projekti käyttää **Python 3.12**-versiota. Kaikki tarvittavat kirjastot on lueteltu tiedostossa `requirements.txt`.

**Kvantisoinnin ja GPU-kiihdytyksen avainkirjastot:**
* `accelerate`
* `bitsandbytes`
* `torch` (CUDA-versio)
* `nemo_toolkit[asr]`

---

### Projektin rakenne
```
/Pääprojekti
├── .venv/                   # Virtuaaliympäristö
├── data/                    # Koulutus- ja testiaineistot (.wav, .txt, jne.)
│   ├── en                   # Hakemisto englanninkieliselle aineistolle. Sis. .wav-tiedostot sekä en_SOURCE.txt.
│   ├── fi                   # Hakemisto suomenkieliselle aineistolle. Sis. .wav-tiedostot sekä fi_SOURCE.txt.
│   ├── fr                   # Hakemisto ranskankieliselle aineistolle. Sis. .wav-tiedostot sekä fr_SOURCE.txt.
│   └── results              # Tulokset
│       ├── asr              # ASR-tulokset malleittain
│       ├── mt               # MT-tulokset malleittain
│       └── evaluation       # Automaattisten metriikoiden arviointitulokset
├── asr/                     # ASR-mallien skriptit
│   ├── whisper_medium.py    # Koodi Whisper Large v2 -mallille
│   ├── wav2vec2_xlsr.py     # Koodi Wav2Vec 2.0 XLSR-53 -mallille
│   └── parakeet.py          # Koodi Parakeet-tdt-0.6b-v3 -mallille (NeMo)
├── mt/                      # MT-mallien skriptit
│   ├── nllb.py              # Koodi NLLB.200-distilled-600M -mallille
│   ├── bloom.py             # Koodi BLOOM-mallille
│   └── opus_models.py       # Koodi Helsinki NLP Opus-MT -malleille
├── evaluation.py            # Metriikoiden laskemisskripti (WER, COMET, TER, jne.)
└── requirements.txt         # Lista riippuvuuksista
```

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
Suorita skrptit projektin juurikansiossa.

#### Puheentunnistus (ASR)
Kaikkien äänitiedostojen putkittainen transkriptio:
```bash
python -m asr.whisper_large
```

```bash
python -m asr.parakeet
```

```bash
python -m asr.wav2vec_xlsr
```

Jos on tarpeen ajaa transkriptio vain yhdelle mallille, lisätään komennon perään ajettavan äänitiedoston polku komentoriviargumentissa:
```bash
python -m asr.malli data/kielilyhenne/tiedoston_nimi.wav
```

#### Konekääntäminen (MT)
MT-skriptejä (NNLB, BLOOM, Opus-MT) ajetaan komentoriviargumenteilla.

##### Yksittäinen käännös (Single Translation)
**Argumentit:**
- --src_lang [fi/en/fr]: Määrittää **lähdekielen** lyhenteen.
- --tgt_lang [fi/en/fr]: Määrittää **kohdekielen** lyhenteen.
- --src_file [POLKU]: Määrittää **lähdetekstin** tiedostopolun.
- --batch_size [int size, default 4]: Määrittää ajettavan erän koon. Vapaaehtoinen argumentti BLOOM-käännöksissä. Oletusarvo 4.

**Esimerkki (MT - NLLB):**
```bash
python -m mt.nllb --src_lang fi --tgt_lang en --src_file data/fi/source_texts.txt
```

##### Eräajo (Batch Translation)
Käytä mt.run_all.py-ajuria suorittaaksesi käännöksen automaattisesti kaikkiin muihin kieliin yhden lähtökielen pohjalta.

**Argumentit**:
- --model [opus/nllb/bloom]: Määrittää käytettävän malliperheen.
- --src_lang [fi/en/fr]: Määrittää lähdekielen lyhenteen.
- --src_file [POLKU]: Määrittää lähdetekstin tiedostopolun.

**Esimerkki (Batch MT - Opus):**
```bash
python -m mt.run_all --model opus --src_lang fi --src_file data/fi/source_texts.txt
```

#### Kvantisointi (INT4)
Suuret mallit (kuten Whisper Large, NLLB ja BLOOM) ladataan automaattisesti 4-bittisesti (`utils/model_loader.py` kautta) VRAM-muistin säästämiseksi.

### 2. Arviointi (Evaluation)
Suorita automaattinen arviointi skriptillä `evaluation.py` malliajon jälkeen.
```bash
python evaluation.py
```
**Argumentit:**
- --src_file [POLKU]: Määrittää **arvioitavan tekstin** tiedostopolun.
- --ref_file [POLKU]: Määrittää referenssitekstin tiedostopolun.
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