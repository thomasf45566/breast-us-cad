# BreastUS-CAD 正式文件(整合版 v2.0)
## 乳房超音波良惡性分類:凍結式外部驗證與部署導向之後續研究

**王表元(Thomas Wang),PGY2,國立台灣大學醫學院附設醫院外科組**
**2026-09|研究原型,非醫療器材**
**線上展示:https://huggingface.co/spaces/happytommy/breast-us-cad**

---

## 摘要

**背景:** 台灣女性乳房緻密比例高,超音波為第一線工具,但判讀者間變異顯著。深度學習模型之跨機構可攜性——尤其 operating point 層級——鮮少被嚴謹評估。

**方法:** 以 BUS-BRA(1,875 張 / 1,064 位病人,病理確診)依官方 patient-level 五折訓練 ViT-B/16,經 hflip TTA 與 temperature scaling,於 pooled out-of-fold 預測上以 sensitivity ≥ 0.90 規則預定 operating point 後凍結,再依 pre-registered protocol 於四個外部世代(BrEaST、去重 BUSI、GDPH、SYSUCC;共 2,454 張,經 970 萬對感知雜湊稽核確認與訓練集零重疊)單次驗證。其後以四個 protocol amendments 進行部署導向後續研究:site-specific recalibration 學習曲線、跨場域閾值轉移、ensemble disagreement 之 abstention 分析、以及領域預訓練骨幹(BiomedCLIP)與多來源訓練(leave-one-cohort-out, LOCO)之 pre-registered 比較。

**結果:** 內部五折 AUC 0.931 ± 0.016;operating point sens 0.903 / spec 0.771。外部 AUC 0.838–0.934,sensitivity 皆 ≥ 0.918,specificity 降至 0.409–0.630;機制為良性校準後機率中位數由 0.06 右移至 0.17–0.31。模型表現落於兩位放射科醫師 BI-RADS 判讀之分佈內(κ 0.22–0.52)。後續研究顯示:10–20 例本地標註即可於中位數上恢復幾乎全部 specificity(可靠性需約 100–200 例);任何單調機率校準接本地閾值重選,其決策與直接重定閾值完全相同;外部世代彼此借用閾值皆優於訓練集閾值;BiomedCLIP 預訓練縮小良性漂移(比值 0.69–0.96)但於四世代 AUC 皆較低;LOCO 多來源訓練於四世代 ΔAUC +0.013 至 +0.056(判準成立)、Δspec +0.27 至 +0.30(paired CI 皆排除零),代價為 held-out sensitivity 降至 0.80–0.96。

**結論:** 判別力可跨洲際遷移而校準不可;失效偏向多呼叫惡性之安全方向。訓練端(多來源、領域預訓練)與部署端(本地校準)之修正殊途同歸:**部署之最後一哩為以少量本地資料設定 operating point**。本框架可直接延伸至 NTUH 本土研究。

---

## 1. 背景與動機

### 1.1 臨床脈絡
台灣 55 歲以下女性逾八成乳房攝影屬不均質或極度緻密(Chang et al.),緻密乳房中超音波敏感度顯著優於攝影,故超音波為台灣乳房病灶評估之一線工具。然其判讀一致性有限:本研究於兩個中國世代觀察到兩位放射科醫師以 BI-RADS ≥ 4a 為陽性之 κ 僅 0.215–0.515,與文獻報告相符。

### 1.2 研究缺口
既有文獻多以單一資料集之影像層級隨機切分報告 AUC 0.93–0.98,存在 patient-level leakage 與資料品質問題(BUSI 之重複影像已見諸文獻);外部驗證研究普遍顯示 AUC 降至 0.85–0.88,而 operating point(閾值層級)之可攜性、及其部署端與訓練端修正策略之系統性比較,均鮮少報告。

### 1.3 研究目標
(1) 建立 patient-level、具校準與臨床導向 operating point 之分類器;(2) 凍結後單次跨洲際外部驗證,分別評估判別力與 operating point 可攜性;(3) 以機率分佈、可解釋性與判讀者比較理解機制;(4) 系統性評估部署端(本地校準)與訓練端(領域預訓練、多來源訓練)之修正策略;(5) 為 NTUH 本土研究建立方法學基礎。

---

## 2. 材料與方法

### 2.1 訓練資料
BUS-BRA(Gómez-Flores et al., *Med Phys* 2024):巴西國家癌症研究所、四種儀器、1,875 張 B-mode 影像、1,064 位病人、病理確診 722 良性 / 342 惡性病例,附 BI-RADS、分割遮罩與官方 patient-level 五折切分。全程採官方切分並以自動化測試守門。影像層級盛行率 32.4%。

### 2.2 模型開發(v1)
前處理:灰階複製三通道、224×224、ImageNet 標準化;訓練增強:水平翻轉、≤10° 平移旋轉、亮度對比。骨幹以 EfficientNet-B0 為 baseline,fold-5 篩選(門檻 +0.01)後對 ConvNeXt-Small 與 ViT-B/16 行完整五折 CV;訓練 AdamW + cosine(ViT 含 3-epoch warmup)、BCE 加類別權重、30 epochs、batch 32,於 Apple M4(MPS)執行。開發過程含 Grad-CAM 品質檢查、燒錄標註稽核、外圈 15% occlusion probe,及以 pre-registered AND 規則(pooled OOF AUC 降幅 < 0.01 且 caliper 鄰近熱區可見改善)測試之 CoarseDropout 實驗(未達標,棄用)。

TTA 為原圖與水平翻轉之機率平均;temperature 於 pooled OOF logits 以 LBFGS 擬合;operating point 於校準後 OOF 上取 sensitivity ≥ 0.90 之最高閾值,patient-level bootstrap(2,000 次)估 CI。最終管線(五模型 ensemble + TTA + T + 閾值)以 `frozen-v1` 凍結。

### 2.3 外部驗證 protocol(pre-registered)
外部推論前以版本控制固定:標籤對映、排除規則、與內部一致之前處理、每世代獨立報告、bootstrap 型式(BrEaST 為 patient-level,餘為 image-level 並明示)。世代清單於 Amendment 1 凍結為四個(表 1)。BUSI_WHU 因磁碟檔案家族(756/171)與發表類別數(560/367)矛盾且三路驗證失敗而排除;BUS-CoT 因聚合公開來源與訓練集重疊而不適用。

**表 1|外部世代**

| 世代 | 來源 | n(去重後)| 標籤來源 | 盛行率 |
|---|---|---|---|---|
| BrEaST | 波蘭(Pawłowska 2024)| 252 | 切片/追蹤 | 0.389 |
| BUSI(cleaned)| 埃及(Al-Dhabyani 2020)| 379 | 資料集標籤+去重 | 0.430 |
| GDPH | 廣東省人民醫院 | 810 | 病理報告 | 0.463 |
| SYSUCC | 中山大學腫瘤中心 | 1,013 | 病理報告 | 0.715 |

### 2.4 資料稽核
pHash 對五個資料來源行約 970 萬對比對(d ≤ 8,邊界人工裁決):SYSUCC 移除 507 重複與 39 標籤衝突影像(含同影像雙標籤);GDPH 846→810;跨集零重複(訓練污染排除)。Keep-list 於推論前凍結。

### 2.5 單次驗證與品質保證
推論前六項唯讀 audit(git 狀態、protocol、keep-lists、registry、凍結產物、無外部預測檔)全數通過並留書面 GO;推論碼以 self-check 驗證(fold-5 逐位重現 AUC 0.9234);四世代單一 session 一次評分,結果直接入帳(tag `external-v1`)。

### 2.6 次分析
機率分佈分析(post-hoc 描述性);放射科醫師 BI-RADS 比較(pre-registered,≥ 4a 為陽性);Grad-CAM 內部四象限圖庫與外部假陽性圖庫。

### 2.7 部署導向後續研究(Amendments 2–4)
**Amendment 2 — Recalibration 學習曲線:** 每世代抽 k ∈ {10…200} 例模擬本地標註(自然盛行率;BrEaST patient-level、餘 image-level),於其上重定 operating point,於餘 n−k 例評估;R = 500;方法 M1(閾值重選)、M2a(重估 T + 內部閾值)、M2b(重估 T + 本地閾值)、M3(Platt + 本地閾值);單類別抽樣退回凍結閾值並報告。Pre-registered k* = 恢復比例中位數 ≥ 0.80 且 sens 中位數 ≥ 0.85 之最小 k;post-hoc k_reliable = 恢復比例 2.5 百分位 ≥ 0.5 之最小 k。跨場域轉移矩陣為 post-hoc 描述性延伸。

**Amendment 3 — Disagreement/abstention:** 授權一次受約束重跑以保存 10 個成員機率(重算平均逐位重現已存值,float32);信號 U_std、U_range 與基線 margin = |p − 0.2683|;分析為誤判預測 AUROC 與 q ∈ {5–30}% abstention 曲線;判準:AUROC ≥ 0.65 且 q=10% 保留集 spec +0.05 而 sens 不降、且優於 margin。

**Amendment 4 — 訓練端修正:** Q2 以 v1 protocol 換骨幹(USFM;載入不潔時以 BiomedCLIP 替代,pre-registered)比較外部漂移;Q1 為 LOCO 四輪(訓練 BUS-BRA + 三外部世代、留一世代單發評估;訓練側 validation = BUS-BRA fold 5 + 各外部世代 15%;每輪獨立擬合 T 與 sens ≥ 0.90 閾值;對照為 v1 fold-5 單模型 + TTA,自已存成員機率計算)。判準:Q2 = AUC ≥ v1+0.01 於 ≥3/4 或良性漂移比 ≤ 0.70 於 ≥3/4;Q1 = ΔAUC ≥ +0.01 於 ≥3/4 或 Δspec ≥ +0.10 且 sens ≥ 0.85 於 ≥3/4。骨幹決定規則(Q2 過則 USFM 系、否則 v1 ViT)先行固定。

### 2.8 分割與展示系統
U-Net(EfficientNet-B0 encoder)於 folds 1–4 訓練、fold 5 評估,為展示輔助功能。Gradio 系統與驗證管線共用推論模組;部署於 Hugging Face Spaces;線上/本機一致性經逐位驗證(BUS-BRA 尺寸範圍內 bitwise;其他尺寸受 cv2 resize kernel 分派差異影響 ≤ 1 灰階,實測機率影響中位 8×10⁻⁴、最大 7×10⁻³,1,421 張中 2 例邊界翻轉)。

### 2.9 可重現性
Git tags:`baseline-v1`→`cv-v1`→`frozen-v1`→`external-v1`→`v2-biomedclip`→`v2-loco`;每一數值對應一 commit;推論碼每次變更重跑 self-check;四個 amendments 之時序以 commit 歷史為證。

---

## 3. 結果

### 3.1 內部驗證與骨幹比較
EfficientNet-B0 五折 0.891 ± 0.027;ConvNeXt-Small 0.9304 ± 0.0173 與 ViT-B/16 0.9307 ± 0.0161 統計平手(Δ 0.0002),ViT 因 CPU 推論快 9 倍(40 vs 376 ms)與 Grad-CAM 可用性獲選。可解釋性稽核:ViT 末層 CAM 空白為飽和 logit 之方法 artifact(blocks[-2].norm1 可用);所有檢視影像含燒錄標註;occlusion 中位降幅 < 0.01;模型高度 overconfident(121 張惡性僅 3 張機率落 0.6–0.95)。CoarseDropout:CV 0.9329 ± 0.0199(平手)、pooled OOF 以 0.0008 壓線過 gate 1、gate 2 未過(1/3 改善、1 不變、1 熱圖改善但預測 0.75→0.28)→ 依規則棄用。

### 3.2 凍結管線
TTA:pooled OOF 0.9231→0.9254(採納)。T = 2.364;ECE 0.0716→0.0401;AUC 不變(assert)。Operating point 0.2683:sens 0.903(0.874–0.928)/ spec 0.771(0.745–0.798)/ PPV 0.654 / NPV 0.943;混淆 548/290/59/978。註:ensemble 無無偏內部估計,外部驗證為其首次考試。

### 3.3 單次外部驗證(表 2)

**表 2|External-v1(凍結閾值 0.2683)**

| 世代 | AUC(95% CI)| Sens | Spec |
|---|---|---|---|
| BrEaST | 0.854(0.802–0.902)| 0.918 | 0.409 |
| BUSI | 0.934(0.906–0.958)| 0.957 | 0.630 |
| GDPH | 0.915(0.895–0.934)| 0.971 | 0.453 |
| SYSUCC | 0.838(0.810–0.866)| 0.931 | 0.474 |
| 內部(OOF)| 0.925 | 0.903 | 0.771 |

### 3.4 機率漂移
良性中位校準機率:內部 0.063 → BrEaST 0.306 / BUSI 0.174 / GDPH 0.289 / SYSUCC 0.287;惡性維持 0.680–0.831。Specificity 崩落完全歸因於良性分佈右移;漂移幅度與資料集風格距離同向。

### 3.5 判讀者比較
GDPH(κ 0.515):模型 0.971/0.453;Reader1 0.976/0.894;Reader2 0.979/0.513。SYSUCC(κ 0.215):模型 0.931/0.474;Reader1 0.913/0.651;Reader2 0.993/0.132。GDPH 之 Reader1 顯著高於模型 ROC;Reader2 落於曲線上;SYSUCC 模型居兩判讀者之間。

### 3.6 Grad-CAM
內部 TP 熱區於病灶本體與邊緣;內部 FP 3/4 聚焦真實可疑結構、1/4 顯示殘餘標註敏感;外部 FP 7/8 熱區正落病灶本體(SYSUCC 四張皆低回音分葉狀良性)——外部 specificity 崩落為 appearance-driven。

### 3.7 Recalibration 學習曲線
Sanity:k=0 逐位重現 external-v1;oracle 閾值 BrEaST 0.376 / BUSI 0.385 / GDPH 0.448 / SYSUCC 0.331,對應 frozen→oracle specificity:0.409→0.623 / 0.630→0.819 / 0.453→0.775 / 0.474→0.568。Pre-registered k*:GDPH、SYSUCC = 10;BrEaST、BUSI = 20;中位恢復比例近 1.0(GDPH spec 0.453 → k=10 中位 0.83,oracle 0.775)。單類別抽樣僅見於 k=10(BrEaST 0.8% / BUSI 0.6% / SYSUCC 2.4% / GDPH 0)。惟 k=10–30 之 95% 帶涵蓋 0–0.97,收斂需 k ≈ 100–200;本地規則以 sens 換 spec(中位 sens 0.83–0.90);post-hoc k_reliable 僅 GDPH 於 200 達標。方法比較:M2b/M3 與 M1 決策完全相同(單調變換 + 排序空間閾值之結構性等價);M2a sens 最高(0.92–0.95)、帶最窄,但恢復不足(GDPH 0.657 vs oracle 0.775);T 擬合失敗率 k=10 為 7–20%、k ≥ 100 為 0。

### 3.8 跨場域閾值轉移
內部參考列逐位重現(閾值 0.2683、T 2.3644)。任一外部閾值(0.331–0.448)於所有其他世代之 specificity 皆優於內部閾值;GDPH 之 0.448 換得最高 spec(0.71–0.89)但跨場域 sens 降至 0.80–0.87。M2 temperature 轉移效果甚微(BrEaST T 2.27 / SYSUCC T 2.23,與內部差 0.03–0.04)。

### 3.9 Disagreement 與 abstention
重算平均於全部 2,454 張逐位重現(float32)。誤判 AUROC(U_std / margin):BrEaST 0.774/0.664、BUSI 0.856/0.754、GDPH 0.802/0.717、SYSUCC 0.626/0.747(前三世代 CI 不重疊)。q=10% abstention(U_std,凍結→保留):BrEaST spec 0.409→0.453(sens 0.918→0.909,富集 ×1.53);BUSI 0.630→0.705(0.957→0.953,×2.64);GDPH 0.453→0.504(0.971→0.968,×1.77);SYSUCC 0.474→0.515(0.931→0.923,×1.14)。以 margin abstain 則方向相反(保留 sens 上升、spec 持平或降)——兩信號 abstain 不同影像。判準 0/4:條款 2 於四世代皆因 sens 條款未過(BUSI/GDPH 之 spec 增益達標);SYSUCC 之條款 1、3 亦未過(margin 反優)。依規定不放寬;判準之概念缺陷(將轉介計為漏診)留作後續 pre-registration。

### 3.10 領域預訓練骨幹(Q2)
USFM 為 BEiT 式架構(無 absolute pos_embed;相對位置偏置表 + LayerScale 共 27 tensors 無槽位),無法潔淨載入 vanilla ViT → pre-registered BiomedCLIP 替代(150/150 tensors,訓練前聲明)。內部:五折 0.9170 ± 0.0220(五折皆低於 v1)、pooled OOF 0.9109;T 2.8645、閾值 0.2007(sens 0.901/spec 0.718)。外部單發(各用自身凍結閾值):AUC 0.844/0.910/0.882/0.831(ΔAUC −0.010/−0.024/−0.033/−0.007);良性漂移比 0.892/0.689/0.962/0.700(163)。判準:分支 A 0/4、分支 B 1/4(SYSUCC 以 0.0002 之差計 fail,即計 pass 亦不改 verdict)→ **NOT MET**;LOCO 骨幹依規則 (v) 定為 v1 ImageNet ViT。註:此結果屬 BiomedCLIP 替代品,不代表超音波 MIM 預訓練;BiomedCLIP 於四世代皆縮小良性漂移卻同時皆降 AUC,暗示穩定性—判別力之權衡。

### 3.11 多來源訓練(Q1,LOCO)

**表 3|LOCO vs v1-single(各自凍結閾值,單發)**

| Held-out | LOCO AUC(CI)| v1-single AUC | ΔAUC(paired CI)| LOCO sens/spec | v1-single sens/spec | 良性中位 v1→LOCO |
|---|---|---|---|---|---|---|
| BrEaST | 0.866(0.817–0.910)| 0.847 | +0.019(−0.016–+0.055)| 0.857/0.753 | 0.939/0.474 | 0.279→0.223 |
| BUSI | 0.937(0.912–0.959)| 0.907 | +0.030(+0.007–+0.055)| 0.847/0.880 | 0.939/0.607 | 0.150→0.051 |
| GDPH | 0.945(0.930–0.960)| 0.890(0.867–0.913)| +0.056(+0.037–+0.074)| 0.960/0.713 | 0.952/0.411 | 0.339→0.278 |
| SYSUCC | 0.844(0.817–0.870)| 0.832 | +0.013(−0.010–+0.035)| 0.800/0.716 | 0.934/0.450 | 0.318→0.171 |

(LOCO 閾值/T:breast 0.474/1.222;busi 0.412/2.480;gdph 0.541/0.9545;sysucc 0.497/0.876。訓練側 val sens 均 ≈ 0.90。Δspec:+0.279(0.196–0.362)/+0.273(0.209–0.338)/+0.301(0.248–0.357)/+0.266(0.212–0.321)。)

**判準:分支 A 4/4 → MET**;分支 B 2/4(BUSI 0.847、SYSUCC 0.800 低於 0.85)。Paired ΔAUC CI 僅 BUSI、GDPH 排除零;Δspec(+0.27 至 +0.30)四世代 CI 皆排除零。機制光譜:BrEaST 以閾值定位為主;BUSI 為真實分佈收斂(良性中位 0.051,全案最低);GDPH 為最乾淨之判別力提升(AUC CI 不重疊、唯一 LOCO sens 高於 v1-single 之世代);SYSUCC 良性下移同時拖累惡性(sens 0.800 之由來)。Sens 可攜性:val ≈ 0.90 → held-out 0.80–0.96,對照 v1 凍結閾值之全域 ≥ 0.92。

### 3.12 分割與展示
U-Net fold-5 Dice 0.902 / IoU 0.833(280/383 > 0.9;失效於陰影/低對比病灶)。展示系統本機 0.53 s;Space 冷啟 9.7 s / warm 6.8 s;重現性註記見 §2.8。

---

## 4. 討論

### 4.1 判別力可攜,校準不可攜——及其兩端修正之合流
外部驗證確立 AUC 與 operating point 可攜性之分離,機制為良性分佈之位置偏移(logit 空間平移,temperature 無從修正)。其後四線後續研究從不同方向逼近同一結論:(1) 部署端——10–20 例本地標註即可於中位數恢復 specificity,惟保證性需約 100–200 例;本地標籤之全部價值在閾值定位(單調校準 + 本地閾值 ≡ 閾值重定之結構性等價);(2) 跨場域——外部世代彼此相近而與訓練來源疏離,BUS-BRA 方為離群者,其良性影像「過於乾淨」致閾值系統性偏低;(3) 訓練端——生醫圖文預訓練(BiomedCLIP)縮小漂移但犧牲判別力;多來源訓練(LOCO)同時改善 AUC(+0.01–0.06)與 specificity(+0.27–0.30),惟增益之大宗仍為「多場域 validation 產出更佳之起始閾值」,且 sensitivity 可攜性由 v1 之全域 ≥ 0.92 退為 0.80–0.96。**四線殊途同歸:無論訓練端如何強化,部署之最後一哩仍為以本地資料設定 operating point。**

### 4.2 失效方向與臨床定位
所有外部世代 sensitivity ≥ 0.918(v1 凍結閾值):模型於域外多開切片而不多漏癌,屬可接受之失效方向。與判讀者比較,模型落於 inter-reader 分佈內而未達最佳專家;κ 0.22–0.52 本身即一致性工具之需求論證。外部誤判為 appearance-driven——模型與判讀者被同一批非典型良性以同一方式誤導(SYSUCC 之 disagreement 反轉為第三方印證:非典型良性使五模型一致地錯)。惟此比較存在資訊不對等(判讀者見完整檢查,模型僅見單張截圖)。

### 4.3 兩個 pre-registered 失敗的方法學價值
CoarseDropout(gate 2 未過即棄、不迭代)與 abstention 判準(信號真實——AUROC 0.77–0.86 且優於 margin——而判準將成功轉介計為 sensitivity 損失;不放寬、留作重新 pre-registration)。兩者示範:判準先於數據,且判準本身可以錯——修正之途徑為新的 pre-registration,而非事後放寬。

### 4.4 公開資料集品質
SYSUCC 35% 重複與同影像雙標籤、BUSI_WHU 標籤不可驗證、BUS-CoT 聚合污染:未經稽核之公開資料切分可同時引入 leakage 與 label noise。pHash 稽核成本低廉,宜為標準前置步驟。

### 4.5 限制
單一國家單一機構之訓練資料;無台灣/NTUH 資料、無 IRB 臨床驗證、非醫療器材;燒錄標註為潛在捷徑(無標註影像之表現預期較低);三世代無病人 ID(image-level bootstrap 與 LOCO val 切分之潛在樂觀);PPV 不可轉移;ensemble 無無偏內部估計;判讀者比較之資訊不對等與評分脈絡未知;中段機率殘餘 overconfidence;USFM 未於原生骨架受測;LOCO 為單模型(對照亦為單模型以維公平);展示系統於非 BUS-BRA 尺寸存在 ≤ 10⁻³ 之 resize 分派差異(2/1,421 邊界翻轉)。

### 4.6 後續工作
(1) NTUH 回溯性研究(IRB 規劃中),第一階段約 100 例:不動模型、僅設 operating point(起始採外部共識保守閾值 0.33–0.38)並驗證台灣族群判別力;第二階段 BI-RADS 3/4A 追蹤 upgrade 預測;第三階段超音波腋下淋巴結轉移預測以支持腋下手術降階。(2) Abstention 判準之重新 pre-registration(workflow 層級 sensitivity 與轉介率)。(3) USFM 於原生 BEiT 骨架之比較;LOCO × 領域預訓練之組合。(4) BUSI_WHU 內容層級標籤復原。(5) 多視角/多模態(影像+報告)以縮小與判讀者之資訊差。

---

## 5. 結論
在 patient-level 切分、pre-registration 與凍結式單發驗證之紀律下,單一機構訓練之乳房超音波分類器展現跨洲際判別力,其 operating point 則需本地設定;訓練端強化有效但不取代本地校準。失效偏向安全方向。本研究同時提供公開資料集稽核之實務範式,及可直接落地 NTUH 之分階段研究路徑。

---

## 附錄 A|圖表清單(依引用順序)

| 編號 | 檔案(reports/)| 用途 |
|---|---|---|
| 圖 1 | roc_oof_operating_point.png | 內部 ROC + 凍結工作點 |
| 圖 2 | prob_shift_external.png | 1×5 機率漂移 |
| 圖 3 | birads_comparison_gdph.png / _sysucc.png | 模型 vs 判讀者 |
| 圖 4 | gradcam_gallery_internal.png / gradcam_external_fp.png | 可解釋性 |
| 圖 5 | v2_recalib_M1_curves.png | 學習曲線 |
| 圖 6 | v2_recalib_methods.png | 方法比較 + 帶寬 |
| 圖 7 | v2_cross_site_matrix.png | 跨場域矩陣 |
| 圖 8 | v2_disagreement_auroc.png / v2_abstention_curves.png | Abstention |
| 圖 9 | v2_loco_summary.png | LOCO ROC + 分佈 |
| 圖 10 | seg_examples.png | 分割 |
| 補充 | reliability_oof_vit_tta.png、phash 證據圖、v2_resize_dispatch_impact | Backup |
