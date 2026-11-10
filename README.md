# Açıklanabilir Yapay Zeka ve Güvenilir Modeller — Dönem Ödevleri

Veri kümesi: UCI *Default of Credit Card Clients* (Yeh & Lien, 2009), 30.000 kredi kartı müşterisi, Tayvan 2005.

## Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
Veri dosyasını `data/default of credit card clients.xls` konumuna koyun
(indirme: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip).

## Yapı
```
src/data.py         yükleme, temizleme kararları, sabit %20 holdout (tüm ödevler ortak)
src/models.py       LightGBM / EBM / LASSO kurucuları ve arama uzayları
src/evaluation.py   metrikler, Nadeau–Bengio düzeltilmiş GA
src/trees.py        GOSDT (eşik tahmini / çeyreklik ayrıklaştırma) ve budanmış CART
src/scorecard.py    EBM → elle hesaplanabilir risk puan kartı
odev2_rashomon.py   Ödev 2: Rashomon açığı
odev3_stability.py  Ödev 3: yorumlanabilir modeller ve kararlılık analizi
src/global_methods.py  permütasyon önemi, korelasyon blokları, PDP/ICE, ALE, vekil ağaç
odev4_global.py     Ödev 4: küresel açıklama yöntemlerinin karşılaştırmalı denetimi
src/shapley.py      kesin müdahaleci Shapley (kütüphanesiz, 2^d numaralandırma) + aksiyom testleri
odev5_shapley.py    Ödev 5: Shapley değerleri, arka plan seçimi, TreeSHAP türleri
odev5_figures.py    Ödev 5 rapor şekilleri (yalnızca results/odev5 CSV'lerinden)
src/explainers.py   Ödev 6: SHAP-müd/yol, LIME (kendi uygulamamız), Örtme, kontroller, uyuşma ölçütleri,
                    yaprak rastgeleleştirici, bölünme eşiği geçişi
odev6_evaluate.py   Ödev 6: açıklamaların değerlendirilmesi (akıl sağlığı, sadakat, kararlılık, uyuşma)
results/odev2/      tablolar (csv/json) ve şekiller
models/             eğitilmiş modeller (Ödev 3–5 bunları kullanır)
```

## Ödev 2
```bash
python odev2_rashomon.py --fast   # ~1-2 dk duman testi (results/odev2_fast)
python odev2_rashomon.py          # tam deney
```
Arama ve tekrarlı ÇD sonuçları önbelleğe alınır; yeniden aramak için `--retune`.

Tekrarlanabilirlik: tüm tohumlar `src/data.py` içindeki `SEED = 42` ile sabittir; holdout
indeksleri `data/split_indices.json` dosyasına yazılır.

## Ödev 3
Önce Ödev 2 çalışmış olmalı (EBM ve LASSO ayarları `results/odev2/budget.json` dosyasından okunur).
```bash
python odev3_stability.py --fast  # duman testi (results/odev3_fast)
python odev3_stability.py         # tam deney: 50 önyükleme
```
Çıktılar `results/odev3/`: kararlılık tabloları (`stability_*.csv`), ayrıklaştırma etkisi
(`discretization.csv`, `gosdt_trees.txt`), çelişki araması (`contradictions.csv`, `diagnostic_*.csv`),
tek sayfalık puan kartı (`figures/scorecard_one_page.pdf`).

## Ödev 4
Ödev 2'deki LightGBM'i (`models/odev2_LightGBM.joblib`) yükler; 7. adım için Ödev 3 sonuçlarını
(`results/odev3/`) kullanır (yoksa o adımı atlar).
```bash
python odev4_global.py --fast     # duman testi (results/odev4_fast)
python odev4_global.py            # tam deney: 30 tekrar
```
Çıktılar `results/odev4/`: `pi_train_vs_test.csv`, `grouped_pi.csv`, `blocks.json`, `ice_parallelism.csv`,
`ale_vs_pdp.csv`, `surrogate_fidelity.csv`, `surrogate_tree.txt`, `rank_table.csv`,
`consistency_with_odev3.csv`, `summary.json` ve `figures/`.

## Ödev 5
Ödev 2'deki LightGBM'i kullanır; ilk 12 öznitelik Ödev 4'ün test permütasyon önemine göre seçilir
(yoksa yeniden hesaplanır), diğer 9 öznitelik dev kümesi ortalamasında sabitlenir.
```bash
python odev5_shapley.py --fast    # duman testi (results/odev5_fast)
python odev5_shapley.py           # tam deney
python odev5_shapley.py --ref own # Adım 3 referansını da kendi kodumuzla hesapla (yavaş)
python odev5_figures.py           # rapor şekilleri → results/odev5/report_figures
```
Açıklanan çıktı log-odds'tur. Her φ tablosunda explainer, değer fonksiyonu, arka plan, arka plan
boyutu ve taban sütunları vardır. Çıktılar: `axiom_tests.csv`, `own_vs_shap.csv`, `background_size*.csv`,
`reference_groups*.csv`, `treeshap_*.csv`, `persons.csv`, `summary.json` ve `figures/`.

## Ödev 6
Ödev 2'deki LightGBM'i (`models/odev2_LightGBM.joblib`) ve ayarlarını (`results/odev2/budget.json`) kullanır.
Dört yöntem: SHAP-müd (müdahaleci TreeSHAP), SHAP-yol (yol-bağımlı TreeSHAP), LIME (kendi uygulamamız,
ek paket gerekmez) ve Örtme (tek öznitelik silme). İki kontrol: Rastgele açıklama ve Ort-uzaklık
(modelden bağımsız, ortalamadan z-uzaklığı). SHAP-müd, LIME ve Örtme aynı arka planı (dev'den 200 satır) kullanır.
```bash
python odev6_evaluate.py --fast   # duman testi (results/odev6_fast), ~1 dk
python odev6_evaluate.py          # tam deney (results/odev6)
```
| Eksen | Test | Çıktı |
|---|---|---|
| Akıl sağlığı | kademeli yaprak rastgeleleştirme (%25/50/75/100, 3 tohum), karıştırılmış etiketle yeniden eğitim (3 tohum) | `sanity.csv`, `sanity_auc.csv` |
| Sadakat | en önemli k özniteliği dolguyla değiştirme; 3 dolgu (ortalama, rastgele, yakın-komşu), MoRF alanı ve PGI@3; Kendall τ | `fidelity_*.csv` |
| Kararlılık | kararı değiştirmeyen gürültü (ε = 0.01, 0.05 sd); açıklama değişimi + tahmin değişimi + geçilen bölünme eşiği | `stability_*.csv` |
| Uyuşma | ilk-1, ilk-3, Spearman, işaret uyumu; en çok uyuşmayan çift için E1–E3 deneyleri | `agreement.csv`, `experiments.csv` |
| Özet | dört eksen, yöntem başına | `summary_table.csv`, `summary.json` |

Şekiller `results/odev6/figures/`; özgün açıklamalar `phi_original.npz`. Tüm tohumlar `SEED = 42`'den türetilir.
