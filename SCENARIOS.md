# EV Şarj Simülasyonu — Senaryo Rehberi

## Kullanım

```bash
# Hazır senaryo
python simulation_v3.py --scenario <isim> --generate-new

# Kendi JSON senaryonu
python simulation_v3.py --scenario-file benim_senaryom.json --generate-new

# Mevcut dataset ile çalıştır (generate-new olmadan)
python simulation_v3.py --scenario avm_medium
```

---

## Hazır Senaryolar

### `avm_medium` — Orta Ölçek AVM *(varsayılan)*

Türkiye'de tipik bir orta büyüklükte alışveriş merkezi. AVM 07:00'de açılır, 22:00'de kapanır.

| Parametre | Değer |
|---|---|
| Trafo | 1600 kVA → **1.267 kW** kullanılabilir |
| Pik limiti | **950 kW** (17:00–22:00) |
| Baz yük | 150–800 kW (akşam piki ~18:30) |
| Günlük araç | 35 |
| Geliş dağılımı | %25 sabah (09:30), %25 öğle (13:00), %50 akşam (18:30) |
| İstasyonlar | S1–S2: 200 kW Ultra Hızlı, S3–S4: 180 kW Hızlı, S5: 120 kW Standart |
| Toplam kapasite | 880 kW |

---

### `office_large` — Büyük Ofis Binası

Kurumsal ofis kompleksi. Çalışanlar sabah gelir, akşama kadar araç park halinde şarj olur.

| Parametre | Değer |
|---|---|
| Trafo | 1000 kVA → **792 kW** kullanılabilir |
| Pik limiti | **550 kW** (08:00–19:00, gündüz çalışma saatleri) |
| Baz yük | 40–500 kW (sabah piki ~09:00) |
| Günlük araç | 40 |
| Geliş dağılımı | %70 sabah (08:30), %30 öğle (13:00) |
| İstasyonlar | S1–S2: 150 kW Hızlı, S3–S5: 22 kW Standart AC |
| Toplam kapasite | 366 kW |

> **Not:** Ofis senaryosunda araçlar 08:00–09:00 arası yoğun geldiği için kuyruk oluşur; AC istasyonlar yavaş şarj ettiğinden araçlar öğlene kadar biter.

---

### `hotel` — Otel

Şehir oteli. Misafirler akşam check-in yapar, gece şarj olur.

| Parametre | Değer |
|---|---|
| Trafo | 1200 kVA → **950 kW** kullanılabilir |
| Pik limiti | **700 kW** (18:00–23:00) |
| Baz yük | 100–600 kW (7/24 operasyon, akşam piki ~20:00) |
| Günlük araç | 30 |
| Geliş dağılımı | %40 öğleden sonra (15:00), %60 akşam (20:00) |
| İstasyonlar | S1–S2: 150 kW Hızlı, S3–S5: 22 kW Standart AC |
| Toplam kapasite | 366 kW |

---

### `hospital` — Hastane

7/24 çalışan hastane. Personel 3 vardiyada gelir; baz yük gece de yüksek kalır.

| Parametre | Değer |
|---|---|
| Trafo | 2000 kVA → **1.584 kW** kullanılabilir |
| Pik limiti | **1.100 kW** (07:00–20:00) |
| Baz yük | 380–750 kW (düz profil, vardiya piklerinde artış) |
| Günlük araç | 25 |
| Geliş dağılımı | %34 sabah vardiyası (07:30), %33 öğleden sonra (15:30), %33 gece (23:30) |
| İstasyonlar | S1–S2: 150 kW Hızlı, S3–S5: 22 kW Standart AC |
| Toplam kapasite | 366 kW |

---

### `airport` — Havalimanı

Yoğun trafikli havalimanı. Sabah ve akşam uçuş dalgalarında yüksek araç yoğunluğu.

| Parametre | Değer |
|---|---|
| Trafo | 4000 kVA → **3.168 kW** kullanılabilir |
| Pik limiti | **2.200 kW** (06:00–22:00) |
| Baz yük | 450–1800 kW (sabah ve akşam çift pik) |
| Günlük araç | 80 |
| Geliş dağılımı | %50 sabah (06:30), %50 akşam (17:30) |
| İstasyonlar | S1–S2: 200 kW Ultra Hızlı, S3–S4: 180 kW Hızlı, S5: 120 kW Standart |
| Toplam kapasite | 880 kW |

---

## Kendi Senaryonu Oluştur

`senaryo_sablonu.json` dosyasını kopyalayıp düzenleyebilirsin:

```json
{
  "name": "Fabrika_Izmir",
  "environment": {
    "name": "Fabrika_Izmir",
    "base_min_kw": 300,
    "base_max_kw": 700,
    "operation_start_hour": 6.0,
    "operation_duration_hours": 16,
    "morning_peak_hour": 7.5,
    "morning_peak_kw": 100,
    "morning_peak_width": 1.0,
    "evening_peak_hour": 15.5,
    "evening_peak_kw": 80,
    "evening_peak_width": 1.0,
    "noise_kw": 30,
    "load_min_kw": 250,
    "load_max_kw": 900
  },
  "grid": {
    "trafo_kva": 2500,
    "power_factor": 0.90,
    "safety_margin": 0.88,
    "evening_peak_kw": 1400,
    "peak_start_hour": 16.0,
    "peak_end_hour": 20.0
  },
  "fleet": {
    "daily_ev_count": 60,
    "initial_soc_min": 0.10,
    "initial_soc_max": 0.40,
    "target_soc": 0.80,
    "arrival_patterns": [
      {"mean_hour": 7.5, "std_minutes": 30, "fraction": 0.50},
      {"mean_hour": 15.5, "std_minutes": 30, "fraction": 0.50}
    ],
    "ev_models": [
      {"model_name": "Togg T10X",         "battery_capacity_kwh": 88.5, "max_dc_power_kw": 150, "probability": 0.25},
      {"model_name": "Tesla Model Y",     "battery_capacity_kwh": 75.0, "max_dc_power_kw": 250, "probability": 0.20},
      {"model_name": "Tesla Model Y RWD", "battery_capacity_kwh": 60.0, "max_dc_power_kw": 170, "probability": 0.15},
      {"model_name": "BYD Atto 3",        "battery_capacity_kwh": 60.4, "max_dc_power_kw": 88,  "probability": 0.15},
      {"model_name": "MG4 Standard",      "battery_capacity_kwh": 51.0, "max_dc_power_kw": 117, "probability": 0.10},
      {"model_name": "Renault Megane",    "battery_capacity_kwh": 60.0, "max_dc_power_kw": 130, "probability": 0.10},
      {"model_name": "Porsche Taycan",    "battery_capacity_kwh": 93.4, "max_dc_power_kw": 270, "probability": 0.05}
    ]
  },
  "layout": {
    "stations": [
      {"station_id": "S1", "station_type": "ultra_fast", "max_power_kw": 200},
      {"station_id": "S2", "station_type": "ultra_fast", "max_power_kw": 200},
      {"station_id": "S3", "station_type": "fast",       "max_power_kw": 180},
      {"station_id": "S4", "station_type": "fast",       "max_power_kw": 180},
      {"station_id": "S5", "station_type": "standard",   "max_power_kw": 120}
    ]
  }
}
```

### Alan Açıklamaları

| Alan | Açıklama |
|---|---|
| `base_min_kw` | Gece/boş saatlerde minimum baz yük |
| `base_max_kw` | Tam operasyonda zirve baz yük |
| `operation_start_hour` | Baz yükün yükselmeye başladığı saat (0–24) |
| `operation_duration_hours` | Yüksek yük penceresi süresi (saat) |
| `morning_peak_hour` | Sabah pik saati |
| `morning_peak_kw` | Sabah pik yüksekliği (baz yüke eklenir) |
| `morning_peak_width` | Sabah pikin genişliği σ (saat) |
| `evening_peak_hour` | Akşam pik saati |
| `evening_peak_kw` | Akşam pik yüksekliği |
| `evening_peak_width` | Akşam pikin genişliği σ (saat) |
| `noise_kw` | Rastgele gürültü standart sapması |
| `load_min_kw` / `load_max_kw` | Baz yük alt/üst sınırı (clip) |
| `trafo_kva` | Trafo nominal gücü (kVA) |
| `power_factor` | Güç faktörü (tipik: 0.90) |
| `safety_margin` | Kullanılabilir kapasite oranı (tipik: 0.88) |
| `evening_peak_kw` | Pik saatlerde toplam trafo limiti |
| `peak_start_hour` / `peak_end_hour` | Pik saat penceresi |
| `daily_ev_count` | Günlük araç sayısı |
| `fraction` | Her geliş dalgasının payı (toplam = 1.0 olmalı) |
| `station_type` | `ultra_fast`, `fast`, `standard` |

> **Trafo kullanılabilir güç** = `trafo_kva × power_factor × safety_margin`  
> Örnek: 1600 kVA × 0.90 × 0.88 = **1.267 kW**
