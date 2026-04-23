---
name: Generic Senaryo Mimarisi Planı
description: simulation_v3.py'yi herhangi bir ortam/araç sayısı/baz yük için parametrik hale getirme planı
type: project
originSessionId: c25b14b7-1333-4999-a4ad-4fb04f5c13d0
---
# Generic Senaryo Mimarisi Planı

**Durum:** Planlandı, henüz implement edilmedi (2026-04-23)

**Why:** Şu an tüm parametreler hardcoded (AVM baz yük, 5 istasyon, 35/75 araç, geliş saatleri). Farklı ortamlar (ofis, otel, hastane, havalimanı) test etmek için her seferinde kodu elle değiştirmek gerekiyor.

**How to apply:** Bu planı implement ederken aşağıdaki yapıyı takip et. Simülasyon motoru (controller, algoritma, excel, grafik) hiç değişmeyecek — sadece input katmanı parametrik hale gelecek.

---

## Mimari

```
ScenarioConfig
├── EnvironmentProfile   ← baz yük eğrisi parametreleri
├── GridSpec             ← trafo kVA, güç faktörü, safety margin, pik saatleri
├── StationLayout        ← istasyon listesi (id, tip, güç)
└── FleetProfile         ← araç modelleri, günlük sayı, geliş dalgaları
```

## Yeni Dataclass'lar (simulation_v3.py'e eklenecek)

### EnvironmentProfile
```python
@dataclass
class EnvironmentProfile:
    name: str
    base_min_kw: float           # Gece minimum
    base_max_kw: float           # Gündüz tepe
    operation_start_hour: float  # Açılış saati
    operation_duration_hours: float
    morning_peak_hour: float
    morning_peak_kw: float
    morning_peak_width: float    # σ saat cinsinden
    evening_peak_hour: float
    evening_peak_kw: float
    evening_peak_width: float
    noise_kw: float
    load_min_kw: float
    load_max_kw: float
```

### GridSpec
```python
@dataclass
class GridSpec:
    trafo_kva: float
    power_factor: float = 0.90
    safety_margin: float = 0.88
    evening_peak_kw: float = 950.0
    peak_start_hour: float = 17.0
    peak_end_hour: float = 22.0

    @property
    def trafo_max_kw(self) -> float:
        return round(self.trafo_kva * self.power_factor * self.safety_margin, 1)
    @property
    def peak_start_min(self) -> int: return int(self.peak_start_hour * 60)
    @property
    def peak_end_min(self) -> int: return int(self.peak_end_hour * 60)
```

### ArrivalPattern
```python
@dataclass
class ArrivalPattern:
    mean_hour: float     # Ortalama geliş saati
    std_minutes: float   # Standart sapma (dakika)
    fraction: float      # Toplam araçların bu dalgaya düşen payı
```

### FleetProfile
```python
@dataclass
class FleetProfile:
    daily_ev_count: int
    ev_models: List[EVModel]
    arrival_patterns: List[ArrivalPattern]
    initial_soc_min: float = 0.10
    initial_soc_max: float = 0.40
    target_soc: float = 0.80
```

### ScenarioConfig
```python
@dataclass
class ScenarioConfig:
    name: str
    environment: EnvironmentProfile
    grid: GridSpec
    fleet: FleetProfile
    layout: StationLayout   # List[ChargingStation]

    def to_grid_limit_policy(self) -> GridLimitPolicy: ...
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, path: str) -> ScenarioConfig: ...
```

## Güncellenecek Sınıflar

- **BackgroundLoadGenerator.generate()**: `EnvironmentProfile` parametresi alır, hardcoded katsayılar silinir
- **ArrivalGenerator.__init__()**: `FleetProfile` alır, hardcoded 570/780/1110 dakika silinir
- **GridLimitPolicy**: `GridSpec.to_grid_limit_policy()` ile oluşturulur
- **main()**: `ScenarioConfig` alır

## Senaryo Fabrikası

```python
class Scenarios:
    @staticmethod
    def avm_medium() -> ScenarioConfig: ...   # Mevcut AVM — referans senaryo
    @staticmethod
    def office_large() -> ScenarioConfig: ... # Ofis: 08-19, sabah ağırlıklı geliş
    @staticmethod
    def hotel() -> ScenarioConfig: ...        # Otel: gece yük var, akşam geliş
    @staticmethod
    def hospital() -> ScenarioConfig: ...     # Hastane: 7/24 düz yük, vardiya geliş
    @staticmethod
    def airport() -> ScenarioConfig: ...      # Havalimanı: yüksek baz, sabah+akşam dalga
```

## CLI

```bash
python simulation_v3.py --scenario avm_medium
python simulation_v3.py --scenario office_large --generate-new
python simulation_v3.py --scenario-file fabrika.json --generate-new
python simulation_v3.py  # varsayılan: avm_medium
```

## JSON Config Örneği

```json
{
  "name": "Fabrika_Izmir",
  "environment": { "base_min_kw": 300, "base_max_kw": 700, ... },
  "grid": { "trafo_kva": 2500, "safety_margin": 0.88, "evening_peak_kw": 1400 },
  "fleet": { "daily_ev_count": 60, "arrival_patterns": [...] },
  "stations": [{"id": "S1", "type": "ultra_fast", "max_kw": 300}]
}
```

## Değişmeyen Kısımlar

`EV`, `ChargingStation`, `UnmanagedController`, `ManagedController`, `Simulation`, `ExecutiveDashboard`, Excel export — **hiçbiri değişmez**.

## Mevcut AVM Değerleri (avm_medium referansı)

- EnvironmentProfile: base_min=180, base_max=500, op_start=7, op_dur=15, morning_peak=(10, 80kW, σ1.2), evening_peak=(18.5, 140kW, σ1.5), noise=25, clip(150,800)
- GridSpec: trafo_kva=1600, pf=0.90, safety=0.88 → 1267.2 kW, evening_peak=950, peak=17-22
- Fleet: 35 araç, 3 dalga (09:30 σ40 %25, 13:00 σ35 %25, 18:30 σ75 %50)
- Stations: S1(UF,200), S2(UF,200), S3(F,180), S4(F,180), S5(STD,120)
