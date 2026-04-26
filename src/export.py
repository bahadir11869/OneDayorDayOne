#!/usr/bin/env python3
"""Excel ve Grafik İhracat - EV Yük Dengeleme Simülasyonu"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models import SimulationResult, MetricsSummary, VehicleSession


def build_station_matrix(timeline_log):
    """Zaman çizelgesini 1440 satırlık Matris formuna dönüştürür"""
    rows = {m: {"Dakika": m, "Saat": f"{m//60:02d}:{m%60:02d}", "S1": "-", "S2": "-", "S3": "-", "S4": "-", "S5": "-", "Kuyruk": "-"} for m in range(1440)}
    queues = {m: [] for m in range(1440)}
    
    for t in timeline_log:
        m = t["Dakika"]
        if t["Durum"] == "Şarjda":
            rows[m][t["İstasyon"]] = f"{t['Araç ID']} | {t['Güç (kW)']}kW | %{t['SoC (%)']}  | {t.get('BazGüç (kW)', '-')}"
        elif t["Durum"] == "Kuyrukta":
            queues[m].append(t['Araç ID'])
            
    for m in range(1440):
        if queues[m]:
            rows[m]["Kuyruk"] = ", ".join(queues[m])
            
    return list(rows.values())


def export_comparative_excel(ctrl_u, ctrl_m, fn: str = "../OutputExcel/ev_sarj_karsilastirma_raporu.xlsx"):
    """İki kontrolcünün karşılaştırmasını Excel'e yazdırır."""
    ud = {e.session_id: e for e in ctrl_u.completed}
    
    # 1. ÖZET SAYFASI
    sum_data = []
    for em in ctrl_m.completed:
        eu = ud.get(em.session_id)
        if not eu: continue
        sum_data.append({
            "Araç ID": em.session_id,
            "Model": em.model_name,
            "Geliş Dakikası": em.arrival_minute,
            "Geliş Saati": f"{em.arrival_minute//60:02d}:{em.arrival_minute%60:02d}",
            "Algoritmasız Şarj Süresi (dk)": eu.charge_minutes,
            "Algoritmalı Şarj Süresi (dk)": em.charge_minutes,
            "Şarj Süresi Farkı (dk)": em.charge_minutes - eu.charge_minutes,
            "Algoritmasız Bekleme (dk)": eu.wait_time_minutes,
            "Algoritmalı Bekleme (dk)": em.wait_time_minutes,
            "Bitiş SoC (%)": round(em.current_soc * 100, 1)
        })

    try:
        with pd.ExcelWriter(fn, engine='openpyxl') as w:
            pd.DataFrame(sum_data).to_excel(w, sheet_name='Özet Karşılaştırma', index=False)
            
            # MATRİSLERİ YAZ
            if ctrl_u.timeline_log:
                pd.DataFrame(build_station_matrix(ctrl_u.timeline_log)).to_excel(w, sheet_name='Algoritmasiz_Istasyon_Logu', index=False)
            if ctrl_m.timeline_log:
                pd.DataFrame(build_station_matrix(ctrl_m.timeline_log)).to_excel(w, sheet_name='Algoritmali_Istasyon_Logu', index=False)
                
        print(f"\n✓ ✓ ✓ İSTASYON BAZLI (MATRİS) EXCEL RAPORU OLUŞTURULDU: {fn} ✓ ✓ ✓\n")
    except Exception as e:
        print(f"Excel aktarımı başarısız: {e}")


def export_multi_controller_excel(all_controllers: list, fn: str = "../OutputExcel/ev_coklu_kontrolcu_raporu.xlsx"):
    """Tüm kontrolcülerin metriklerini ve istasyon loglarını tek Excel dosyasına yazar."""
    metrics_data = []
    for name, ctrl in all_controllers:
        p = np.array(ctrl.power_log)
        l = np.array(ctrl.limit_log)
        over = p > l
        completed = ctrl.completed
        metrics_data.append({
            "Kontrolcü": name,
            "Tamamlanan Araç": len(completed),
            "Maks Güç (kW)": round(float(p.max()), 1),
            "Aşım Dakikası": int(over.sum()),
            "Toplam Aşım (kWh)": round(float(np.where(over, p - l, 0).sum() / 60), 2),
            "Ort. Bekleme (dk)": round(float(np.mean([e.wait_time_minutes for e in completed])) if completed else 0.0, 1),
            "Ort. Şarj Süresi (dk)": round(float(np.mean([e.charge_minutes for e in completed])) if completed else 0.0, 1),
        })

    try:
        with pd.ExcelWriter(fn, engine='openpyxl') as w:
            pd.DataFrame(metrics_data).to_excel(w, sheet_name='Metrik_Karsilastirma', index=False)
            for name, ctrl in all_controllers:
                if ctrl.timeline_log:
                    sheet_name = (name + "_Log")[:31]
                    pd.DataFrame(build_station_matrix(ctrl.timeline_log)).to_excel(w, sheet_name=sheet_name, index=False)
        print(f"✓ Çoklu kontrolcü Excel raporu: {fn}")
    except Exception as e:
        print(f"Çoklu Excel aktarımı başarısız: {e}")


class ExecutiveDashboard:
    @staticmethod
    def create(r_u: SimulationResult, r_m: SimulationResult,
               ctrl_label: str = "Kontrol Sonrası",
               filename: str = "executive_dashboard_v4.png",
               bg_load: Optional[np.ndarray] = None):
        fig = plt.figure(figsize=(16, 30))
        fig.suptitle(f"EV Şarj Yük Dengeleme — {ctrl_label}", fontsize=18, fontweight="bold", y=0.995)
        gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.25)
        axs = [fig.add_subplot(gs[i]) for i in [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3, slice(None)), (4, slice(None))]]

        mods = sorted(list({s.model_name.split()[0] for s in r_u.vehicle_sessions}))
        x = np.arange(len(mods)); wd = 0.35
        ud = {s.session_id: s for s in r_u.vehicle_sessions}
        d_c, d_w = {m: [] for m in mods}, {m: [] for m in mods}
        for s in r_m.vehicle_sessions:
            if s.session_id in ud:
                m = s.model_name.split()[0]
                d_c[m].append(s.charge_time_minutes - ud[s.session_id].charge_time_minutes)
                d_w[m].append(max(0, s.wait_time_minutes - ud[s.session_id].wait_time_minutes))


        w7 = 0.2
        axs[0].bar(x-1.5*w7, [max(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#e74c3c", ec="black", label="Maks Şarj Artışı")
        axs[0].bar(x-0.5*w7, [min(d_c[m]) if d_c[m] else 0 for m in mods], w7, color="#f1948a", ec="black", label="Min Şarj Artışı")
        axs[0].bar(x+0.5*w7, [max(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#2980b9", ec="black", label="Maks Bekleme Artışı")
        axs[0].bar(x+1.5*w7, [min(d_w[m]) if d_w[m] else 0 for m in mods], w7, color="#7fb3d5", ec="black", label="Min Bekleme Artışı")
        axs[0].set(xticks=x, xticklabels=mods, ylabel="Delta (dk)", title="Panel 1: Maks/Min Gecikme")
        axs[0].legend(loc="upper left"); axs[0].grid(True, alpha=0.3, axis="y")


        hrs = np.arange(1440) / 60
        axs[6].fill_between(hrs, 0, r_u.power_timeseries, alpha=0.3, color="red", label="Kontrol Öncesi")
        axs[6].plot(hrs, r_m.power_timeseries, lw=2.5, color="darkgreen", label=ctrl_label)
        axs[6].plot(hrs, r_m.grid_limit_timeseries, color="red", ls="--", lw=2, label="Trafo Limiti")
        axs[6].axvspan(17, 22, alpha=0.08, color="orange", label="Şebeke Pik (17-22)")
        axs[6].set(xlabel="Gün Saati", ylabel="Yük (kW)", title="Panel 7: Toplam Yük Profili", xlim=(0,24))
        axs[6].legend(loc="upper left"); axs[6].grid(True, alpha=0.3)


        w = [s.wait_time_minutes for s in r_m.vehicle_sessions]
        bars = axs[1].bar(["0 dk", "1-15 dk", "15+ dk"], [sum(1 for x in w if x==0), sum(1 for x in w if 0<x<=15), sum(1 for x in w if x>15)], color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black")
        axs[1].set(ylabel="Araç", title="Panel 2: Bekleme Dağılımı")
        for b in bars: axs[1].text(b.get_x() + b.get_width()/2, b.get_height()+0.3, str(int(b.get_height())), ha="center", va="bottom", fontweight="bold")


        g = lambda r: {m: [] for m in mods}
        c_u, c_m, w_u, w_m = g(r_u), g(r_m), g(r_u), g(r_m)
        for s in r_u.vehicle_sessions: m=s.model_name.split()[0]; c_u[m].append(s.charge_time_minutes); w_u[m].append(s.wait_time_minutes)
        for s in r_m.vehicle_sessions: m=s.model_name.split()[0]; c_m[m].append(s.charge_time_minutes); w_m[m].append(s.wait_time_minutes)
        

        axs[2].bar(x-wd/2, [np.mean(c_u[m]) for m in mods], wd, color="lightcoral", ec="black", label="Öncesi")
        axs[2].bar(x+wd/2, [np.mean(c_m[m]) for m in mods], wd, color="lightgreen", ec="black", label=ctrl_label)
        axs[2].set(ylabel="Şarj (dk)", title="Panel 3: Şarj Süresi", xticks=x, xticklabels=mods); axs[2].legend()

        axs[3].bar(x-wd/2, [np.mean(w_u[m]) for m in mods], wd, color="#f9c784", ec="black", label="Öncesi")
        axs[3].bar(x+wd/2, [np.mean(w_m[m]) for m in mods], wd, color="#74b9ff", ec="black", label=ctrl_label)
        axs[3].set(ylabel="Bekleme (dk)", title="Panel 4: Bekleme Süresi", xticks=x, xticklabels=mods); axs[3].legend()

        axs[4].axis("off")
        axs[4].text(0.05, 0.95,
                    f"Algoritma: {ctrl_label}\n"
                    f"Kapasite: {r_m.metrics_summary.protected_capacity_percent:.1f}%\n"
                    f"Servis: {r_m.metrics_summary.evs_completed} araç\n"
                    f"Ort. Bekleme: {r_m.metrics_summary.avg_delay_minutes:.1f} dk\n"
                    f"Aşım: {r_m.metrics_summary.overload_minutes} dk  |  {r_m.metrics_summary.total_overload_kwh:.1f} kWh",
                    transform=axs[4].transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round", fc="lightgreen", alpha=0.4))

        axs[5].bar(x-wd/2, [np.mean(c_u[m])+np.mean(w_u[m]) for m in mods], wd, color="lightcoral", ec="black", label="Öncesi")
        axs[5].bar(x+wd/2, [np.mean(c_m[m])+np.mean(w_m[m]) for m in mods], wd, color="lightgreen", ec="black", label=ctrl_label)
        axs[5].set(ylabel="Sistem (dk)", title="Panel 6: Toplam Süre", xticks=x, xticklabels=mods); axs[5].legend()



        # ── Panel 8: Baz Yük Profili (İstasyon Yükü Hariç) ───────────────────
        if bg_load is not None:
            axs[7].fill_between(hrs, 0, bg_load, alpha=0.35, color="steelblue", label="Baz Yük")
            axs[7].plot(hrs, bg_load, lw=2, color="steelblue")
            axs[7].plot(hrs, r_m.grid_limit_timeseries, color="red", ls="--", lw=2, label="Trafo Limiti")
            axs[7].axvspan(17, 22, alpha=0.08, color="orange", label="Şebeke Pik (17-22)")
            axs[7].set(xlabel="Gün Saati", ylabel="Yük (kW)",
                       title="Panel 8: Baz Yük Profili (İstasyon Yükü Hariç)", xlim=(0, 24))
            axs[7].legend(loc="upper left"); axs[7].grid(True, alpha=0.3)
        else:
            axs[7].axis("off")
            axs[7].text(0.5, 0.5, "Baz yük verisi mevcut değil",
                        ha="center", va="center", transform=axs[7].transAxes, fontsize=12)

        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"✓ Grafik: {filename}")
        plt.show(block=False)  # Grafiği ekranda açar ama kodun akışını durdurmaz
