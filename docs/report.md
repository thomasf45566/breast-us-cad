# BreastUS-CAD 正式文件(整合版 v2.1 — 依 2026-09-06 獨立稽核修訂)
## 乳房超音波良惡性分類:凍結式外部驗證與部署導向之後續研究

**王表元(Thomas Wang),PGY2,國立台灣大學醫學院附設醫院外科組**
**2026-09|研究原型,非醫療器材**
**線上展示:https://huggingface.co/spaces/happytommy/breast-us-cad**

> **修訂說明(2026-09-07):** 本版依獨立對抗式稽核(docs/AUDIT_2026-09-06.md)
> 逐句修正(P1),並納入 P2 之 POST-HOC 量化(RESULTS.md「Post-hoc analyses
> responding to the 2026-09-06 audit」)。每一事實句皆對應 RESULTS.md、data/external_protocol.md 或 reports/、
> models/ 下已提交之 CSV/JSON;無來源之句已刪除,或明確標示為「未記錄之
> session 內觀察」。逐項對照表見 docs/AUDIT_RESPONSE_2026-09-06.md。

---

## 摘要

**背景:** 台灣女性乳房緻密比例高,超音波為第一線工具,但判讀者間變異顯著。深度學習模型之跨機構可攜性——尤其 operating point 層級——鮮少被嚴謹評估。

**方法:** 以 BUS-BRA(1,875 張 / 1,064 位病人,病理確診)依官方 patient-level 五折訓練 ViT-B/16,經 hflip TTA 與 temperature scaling,於 pooled out-of-fold 預測上以 sensitivity ≥ 0.90 規則預定 operating point 後凍結,再依 pre-registered protocol 於四個外部世代(BrEaST、去重 BUSI、GDPH、SYSUCC;共 2,454 張)單次驗證。外部世代與訓練集之污染檢查為全對感知雜湊比對(五個集合 1,875/252/379/846/1,559 張,集合內與集合間全部無序配對共 12,056,505 對;d ≤ 8 之候選經目視裁決),結果為「pHash d ≤ 8 無近重複」,而非「零重疊」。其後以四個 protocol amendments 進行部署導向後續研究:site-specific recalibration 學習曲線、跨場域閾值轉移(post-hoc)、ensemble disagreement 之 abstention 分析、以及領域預訓練骨幹(BiomedCLIP 替代)與多來源訓練(leave-one-cohort-out, LOCO)之 pre-registered 比較。

**結果:** 內部指標分為兩個不同的預測器:(a) 單模型、無 TTA 之五折 CV AUC 0.9307 ± 0.0161,各折取該折 validation AUC 最佳之 epoch 之檢查點(best-epoch on the reported fold;POST-HOC 固定 epoch 下 0.9195 ± 0.0164,樂觀量 0.011 ± 0.007);(b) 五模型 ensemble + TTA + T 之 pooled OOF AUC 0.9254。凍結 operating point(閾值 0.2683)之 sens 0.903 / spec 0.771 係於同一批 OOF 預測上擬合閾值後之 in-sample 值(POST-HOC 嵌套 out-of-sample 估計 0.900 ± 0.057 / 0.780 ± 0.132)。外部 AUC 0.838–0.934,sensitivity 皆 ≥ 0.918,specificity 降至 0.409–0.630;伴隨良性校準後機率中位數由 0.06 右移至 0.17–0.31(描述性)。判讀者比較:GDPH 模型於 sensitivity 與 specificity 兩軸皆低於兩位判讀者;SYSUCC 模型居兩位判讀者之間;κ(0.22 / 0.51)為判讀者間一致性,不涉及模型。後續研究:10–20 例本地標註即可於中位數上恢復接近 oracle 之 specificity(pre-registered k*),但 k=10 之中位 sensitivity 為 0.83–0.91;post-hoc 之 draw-level 可靠性指標 k_reliable 僅 GDPH 於 k=200 達標,其餘三世代於各自 pre-registered 網格內未達;單調機率校準接本地閾值重選與直接重定閾值為結構性等價(實跑曲線於 k ≤ 30 因擬合失敗回退與同分而有小差異);外部世代彼此借用閾值之 specificity 皆高於訓練集閾值(post-hoc、in-sample 閾值、描述性;GDPH 閾值使跨場域 sens 降至 0.80–0.87);BiomedCLIP 替代骨幹依 pre-registered 判準 **NOT CLAIMED**(分支 A 0/4、分支 B 1/4;外部 AUC 點估計於四世代皆較低);LOCO 多來源訓練達成 pre-registered 判準(分支 A:ΔAUC ≥ +0.01 vs v1 fold-5 單模型於 4/4),但 paired ΔAUC CI 於 BrEaST、SYSUCC 含零,分支 B 2/4 未達,對照為單模型(相對已部署之 ensemble 僅 2/4 世代 ≥ +0.01),held-out sensitivity 0.80–0.96。

**結論:** 判別力可跨洲際遷移而 operating point 不可(四世代單發驗證)。外部誤判方向為假陽性增加而非漏診;此方向是否「安全」未經危害分析。訓練端修正中,多來源訓練達成其 pre-registered 判準(附上述保留),領域預訓練替代品未達;部署端之本地閾值重選於中位數有效但小樣本下 draw-level 不可靠。**部署之最後一哩為以本地資料設定 operating point**,此為本研究最有支撐之結論。

---

## 1. 背景與動機

### 1.1 臨床脈絡
台灣 55 歲以下女性逾八成乳房攝影屬不均質或極度緻密(Chang et al.),緻密乳房中超音波敏感度顯著優於攝影,故超音波為台灣乳房病灶評估之一線工具。然其判讀一致性有限:本研究於兩個中國世代觀察到兩位放射科醫師以 BI-RADS ≥ 4a 為陽性之 Cohen's κ 為 0.215(SYSUCC)與 0.515(GDPH),與文獻報告相符。

### 1.2 研究缺口
既有文獻多以單一資料集之影像層級隨機切分報告 AUC 0.93–0.98,存在 patient-level leakage 與資料品質問題(BUSI 之重複影像已見諸文獻);外部驗證研究普遍顯示 AUC 降至 0.85–0.88,而 operating point(閾值層級)之可攜性、及其部署端與訓練端修正策略之系統性比較,均鮮少報告。

### 1.3 研究目標
(1) 建立 patient-level、具校準與臨床導向 operating point 之分類器;(2) 凍結後單次跨洲際外部驗證,分別評估判別力與 operating point 可攜性;(3) 以機率分佈、可解釋性與判讀者比較理解機制;(4) 系統性評估部署端(本地校準)與訓練端(領域預訓練、多來源訓練)之修正策略;(5) 為 NTUH 本土研究建立方法學基礎。

---

## 2. 材料與方法

### 2.1 訓練資料
BUS-BRA(Gómez-Flores et al., *Med Phys* 2024;以下資料集描述取自該發表論文):巴西國家癌症研究所(INCA)、四種超音波儀器、1,875 張 B-mode 影像、1,064 位病人、病理確診 722 良性 / 342 惡性病例,附 BI-RADS、分割遮罩與官方 patient-level 五折切分(資料集發布之 `5-fold-cv.csv`,存於 git-ignored 之 data/raw/,**不在版本庫內**;每折影像數 376/385/366/365/383)。全程採官方切分並以自動化測試守門。影像層級盛行率 607/1,875 = 32.4%(models/operating_point.json 混淆矩陣 tp+fn)。

### 2.2 模型開發(v1)
前處理:灰階複製三通道、224×224、ImageNet 標準化;訓練增強:水平翻轉、≤10° 平移旋轉、亮度對比。骨幹以 EfficientNet-B0 為 baseline(fold-5 AUC 0.8823;五折 0.891 ± 0.027),於 fold 5 篩選 ConvNeXt-Small(0.9257)與 ViT-B/16(0.9293)後對兩者行完整五折 CV;訓練 AdamW + cosine(ViT 含 3-epoch warmup)、BCE 加類別權重、30 epochs、batch 32,於 Apple M4(MPS)執行。**Epoch 選擇:** 每折保存該折 validation AUC 最高之 epoch 之檢查點(reports/cv_vit_summary.csv `best_epoch` 欄:29/10/18/16/19),而該折之報告 AUC 即該 epoch 於同一折之值——各折 CV AUC 因此為 best-epoch-on-the-reported-fold,屬樂觀偏差;由此五個檢查點產生之 pooled OOF 預測(供 T 與閾值擬合)承襲此偏差。開發過程含 Grad-CAM 品質檢查、燒錄標註稽核(raw-image audit sheet)、外圈 15% occlusion probe,及以 pre-registered AND 規則(pooled OOF AUC ≥ plain − 0.01 且 caliper 鄰近熱區可見改善)測試之 CoarseDropout 實驗(gate 2 未過,棄用)。

TTA 為原圖與水平翻轉之機率平均;**TTA 之採用未經 pre-registration**——係於看到 pooled OOF AUC +0.0023 後決定(凍結前之研究者自由度,作用於其後用以擬合 T 與閾值之同一批 OOF 資料)。Temperature 於 pooled OOF logits 以 LBFGS 擬合;operating point 於校準後 OOF 上取 sensitivity ≥ 0.90 之最高閾值,patient-level bootstrap(2,000 次)估 CI。最終管線(五模型 ensemble + TTA + T + 閾值)以 `frozen-v1` 凍結。

### 2.3 外部驗證 protocol(pre-registered,內部版本控制)
外部推論前以版本控制固定:標籤對映、排除規則、與內部一致之前處理、每世代獨立報告、bootstrap 型式(BrEaST 為 case-level ≡ image-level,餘為 image-level 並明示)。世代清單於 Amendment 1 凍結為四個(表 1)。BUSI_WHU 排除:磁碟檔案家族(756/171)與發表類別數(560/367)矛盾,且 Mendeley 記錄無逐檔標籤、DSATNet 載入器僅供分割、作者 HF 再發布經重新編號無法對映(protocol (f))。Pre-registration 之來源與限制見 §2.10。

**表 1|外部世代**

| 世代 | 來源 | n(去重後)| 標籤來源 | 盛行率 |
|---|---|---|---|---|
| BrEaST | 波蘭(Pawłowska 2024)| 252 | 切片/追蹤 | 0.389 |
| BUSI(cleaned)| 埃及(Al-Dhabyani 2020)| 379 | 資料集標籤+去重 | 0.430 |
| GDPH | 廣東省人民醫院 | 810 | 病理報告 | 0.463 |
| SYSUCC | 中山大學腫瘤中心 | 1,013 | 病理報告 | 0.715 |

### 2.4 資料稽核
64-bit pHash、d ≤ 8 為候選;集合內候選以連通分量去重(保留字典序首張,標籤衝突群整組剔除),集合間候選以目視裁決(protocol (e)(g))。比對範圍為五個集合(BUS-BRA 1,875、BrEaST 252、BUSI keep-list 379、GDPH 846、SYSUCC 1,559)之集合內與集合間全部無序配對,由集合大小計 C(n,2) 之和加上集合間乘積之和 = 12,056,505 對(先前文件所載「約 970 萬對」無來源,已更正)。結果:SYSUCC 1,559→1,013(移除 507 重複與 39 標籤衝突影像,含同影像雙標籤);GDPH 846→810(34 重複、2 標籤衝突);BUSI 386→379。集合間 d ≤ 8 僅 2 個候選(bus_0999-l ↔ SYSUCC malignant(81);GDPH benign(784) ↔ SYSUCC malignant(23)),目視裁決均為不同掃描(裁決紀錄:reports/phash_cross_pairs_table.csv;含 SYSUCC/GDPH 影像之原始並列圖不隨版本庫散布);其餘集合對最小距離 ≥ 10。**結論為「pHash d ≤ 8 無近重複」**:此方法無法排除 d > 8 之同病人再掃、裁切或不同幀。Keep-list 於推論前凍結。

### 2.5 單次驗證與品質保證
Amendment 1 撰寫時已核對 reports/ 內無任何外部預測檔(protocol:81–83);推論碼以 self-check 驗證(fold-5 TTA AUC 0.9234433158791243 逐位重現,reports/external_selfcheck.txt);四世代以單一指令一次評分(`external_val.py --dataset all --confirm`),結果直接入帳(tag `external-v1`),四個預測 CSV 各僅有一個新增 commit、從未修改。*未記錄之 session 內觀察:推論前另行執行之多項唯讀檢查(git 狀態、registry 等)與其「GO」決定未留存於任何已提交檔案,故不作為事實陳述。*

### 2.6 次分析
機率分佈分析(post-hoc 描述性);放射科醫師 BI-RADS 比較(pre-registered,≥ 4a 為陽性,自已存預測計算);Grad-CAM 內部四象限圖庫與外部假陽性圖庫(定性、單模型)。

### 2.7 部署導向後續研究(Amendments 2–4)
**Amendment 2 — Recalibration 學習曲線:** 每世代抽 k ∈ {10, 20, 30, 50, 100, 200}(k ≥ n/2 者剔除)例模擬本地標註(自然盛行率;BrEaST case-level、餘 image-level),於其上重定 operating point,於餘 n−k 例評估;R = 500,seed 42;方法 M1(閾值重選)、M2a(重估 T + 內部閾值)、M2b(重估 T + 本地閾值)、M3(Platt + 本地閾值);單類別抽樣退回凍結閾值並報告。Pre-registered k* = 恢復比例中位數 ≥ 0.80 且 sens 中位數 ≥ 0.85 之最小 k。**Post-hoc:** k_reliable(恢復比例 2.5 百分位 ≥ 0.5 之最小 k,於看到 M1 結果後定義);M2 之 T 擬合失敗(非正 T)回退凍結管線之規則為執行時新增、非 amendment 所載;跨場域轉移矩陣為 post-hoc 描述性延伸。

**Amendment 3 — Disagreement/abstention:** 授權一次受約束重跑以保存 10 個成員機率(重算平均須逐位重現已存值,否則中止;實測 float32 逐位重現);信號 U_std、U_range 與基線 margin = |p − 0.2683|;分析為誤判預測 AUROC 與 q ∈ {5, 10, 20, 30}% abstention 曲線;判準:AUROC ≥ 0.65 且 q=10% 保留集 spec +0.05 而 sens 不降、且優於 margin。註冊之單一圖檔拆為兩張 PNG(呈現層面偏離,已聲明)。

**Amendment 4 — 訓練端修正:** Q2 以 v1 protocol 換骨幹(USFM;載入不潔時以 BiomedCLIP 替代,pre-registered)比較外部漂移;Q1 為 LOCO 四輪(訓練 BUS-BRA folds 1–4 + 三外部世代 85%、留一世代單發評估;訓練側 validation = BUS-BRA fold 5 + 各外部世代 15%;每輪獨立擬合 T 與 sens ≥ 0.90 閾值;early stopping 之 patience 7 未於 amendment 中預先指定;對照為 v1 fold-5 單模型 + TTA,自已存成員機率計算)。判準:Q2 = AUC ≥ v1+0.01 於 ≥3/4 或良性漂移比 ≤ 0.70 於 ≥3/4;Q1 = ΔAUC ≥ +0.01 於 ≥3/4 或 Δspec ≥ +0.10 且 held-out sens ≥ 0.85 於 ≥3/4。註冊之 Q1 問題為「多來源訓練是否減少 domain-shift 之 **specificity** 崩落」(protocol:390–392),而判準允許僅以 AUC 分支通過。骨幹決定規則(Q2 過則 USFM 系、否則 v1 ViT)先行固定。

### 2.8 分割與展示系統
U-Net(EfficientNet-B0 encoder)於 folds 1–4 訓練、fold 5 評估,為展示輔助功能。Gradio 系統與驗證管線共用**同一模型、校準與推論模組**(src/inference.py);**前處理不完全相同**:驗證路徑為 albumentations/cv2 INTER_LINEAR,展示路徑為整數 numpy 移植之 KleidiCV bilinear kernel,兩者於 BUS-BRA 尺寸範圍(1,875 張,713 種尺寸)逐位相同,但 cv2 僅對部分尺寸分派 KleidiCV kernel,故於 1,421/2,454 張外部 keep-list 影像(BrEaST 99/252、BUSI 0/379、GDPH 309/810、SYSUCC 1,013/1,013)差 ≤ 1 灰階,經完整 ensemble 之校準機率差最大 7×10⁻³、中位 8×10⁻⁴,1,421 張中 2 例邊界翻轉(BrEaST case038、case200,距閾值 < 0.002)。external-v1 之數字產生於 inference.py 建立之前(commit 2cbe1da 08-29 vs 2859043 08-30),走典型 albumentations 路徑。線上 Space 與本機四張範例機率最大差 1.07×10⁻⁷(跨架構 BLAS 底線)。

### 2.9 可重現性
Git tags:`baseline-v1`→`cv-v1`→`frozen-v1`→`external-v1`→`v2-biomedclip`→`v2-loco`。RESULTS.md 每一節列出產生腳本與 reports/、models/ 下之 CSV/JSON 產物,所有表列數值可由該等已提交檔案重算。**不在版本庫內者:** 所有模型檢查點(models/*.pt,.gitignore)與 BUS-BRA fold 檔(data/raw/busbra/5-fold-cv.csv)。v1 之五個 cv_vit 檢查點、分割模型與兩個 JSON 公開於 HF 權重庫 happytommy/breast-us-cad-weights;v2 之九個檢查點(v2_biomedclip_fold{1-5}、v2_loco_{breast,busi,gdph,sysucc})與 BiomedCLIP 匯出權重尚未公開,Q1/Q2 目前僅能自已存預測 CSV 層級重現。MPS 訓練無 bitwise 決定性(未啟用 deterministic algorithms、DataLoader workers=4),檢查點本身無法逐位重現;推論碼每次變更重跑 self-check。

### 2.10 Pre-registration 之來源與限制(provenance)
- 所有 protocol 與 amendments 為**內部、版本控制之 pre-registration**(data/external_protocol.md,append-only;Amendments 2/3/4 各自單獨 commit 於其所規範之計算之前),**未於任何外部登錄處(OSF 等)留存時間戳**。
- 2026-08-31 曾以 `git filter-branch` 重寫全部先前 30 個 commit 之作者/提交者身分(hostname 衍生之 email → 正式 email);author/committer 日期保留、tags 帶入、`frozen-v1` 附註 tag 以相同日期重建(commit 2c24500 訊息)。重寫前之 commit hash 因此無法直接對映至現行歷史。
- 稽核當時(2026-09-06)GitHub 版本庫為私有;唯一公開產物為 HF 權重庫(2026-08-30 建立),其時間晚於 external-v1 commit(2026-08-29)。凍結先於外部評分之時序,目前僅有本機、已重寫、自行指定之 commit 時間戳與檔案 mtime 為據,無第三方紀錄。
- Amendment 1 於登錄時已含其所登錄之去重計數與 pHash 結果(對模型指標而言仍在先)。
- 公開時間戳(Zenodo/OSF)列於後續工作。

---

## 3. 結果

### 3.1 內部驗證與骨幹比較
五折 CV(單模型、無 TTA、各折 best-epoch 檢查點,見 §2.2):EfficientNet-B0 0.891 ± 0.027;ConvNeXt-Small 0.9304 ± 0.0173 與 ViT-B/16 0.9307 ± 0.0161 統計平手(Δ 0.0002),ViT 因 CPU 推論快 9 倍(40 vs 376 ms)與 Grad-CAM 可用性獲選。可解釋性稽核:ViT 末層 CAM 空白為飽和 logit 之方法 artifact(blocks[-2].norm1 可用);所有檢視影像含燒錄標註;occlusion 中位降幅 < 0.01(伴重尾,~16% 影像降幅 > 0.2,受病灶延伸至邊界混淆)。校準前 ensemble 明顯 overconfident(T = 2.36);校準後中段機率仍殘餘輕度 overconfidence(reliability diagram)。CoarseDropout:CV 0.9329 ± 0.0199、pooled OOF 0.9139 以 0.0008 壓線過 gate 1、gate 2 未過(1/3 改善、1 不變、1 熱圖改善但預測 0.75→0.28)→ 依規則棄用;gate 2 為三張影像之主觀目視判斷。**POST-HOC 量化(2026-09-07,RESULTS.md「Post-hoc analyses」§1、§3):** 五折 best-epoch AUC 0.9307 ± 0.0161 對固定 epoch 18(五個最佳 epoch 之中位數)之 0.9195 ± 0.0164,樂觀量 0.0111 ± 0.0073(對最終 epoch 30 為 0.0071 ± 0.0059),折間排序不變;fold-5 121 張惡性中,校準前 TTA 機率落於 [0.6, 0.95] 者 32 張(26.4%),> 0.95 者 61 張——先前「僅 3 張」之數字無來源且不被重現。

### 3.2 凍結管線
TTA:pooled OOF 0.9231→0.9254(採納,未 pre-registered)。T = 2.3644;ECE(15 bins)0.0716→0.0401;NLL 0.4411→0.3237;AUC 不變(assert)。Operating point 0.2683:sens 0.9028(0.874–0.928)/ spec 0.7713(0.745–0.798)/ PPV 0.654 / NPV 0.943;混淆 548/290/59/978。**此 sens/spec 為 in-sample:閾值於同一批 pooled OOF 預測上擬合**(Amendment 2 (l) 對同一運算之稱謂為「in-sample oracle」)。**POST-HOC 嵌套估計(2026-09-07,RESULTS.md「Post-hoc analyses」§2):** 以其餘四折選閾值、於第 k 折評估,五個閾值 0.236–0.311(0.2753 ± 0.0354),out-of-sample sens 0.9003 ± 0.0569 / spec 0.7802 ± 0.1315(pooled 決策 0.9012 / 0.7784)——平均值與 in-sample 相近,但 held-out sens 於 3/5 折低於 0.90 設計下限(0.849–0.976),spec 介於 0.550–0.878。註:ensemble 無無偏內部估計(每張影像對 5 成員中 4 者為 in-fold),外部驗證為其首次考試。

### 3.3 單次外部驗證(表 2)

**表 2|External-v1(凍結閾值 0.2683)**

| 世代 | AUC(95% CI)| Sens | Spec |
|---|---|---|---|
| BrEaST | 0.854(0.802–0.902)| 0.918 | 0.409 |
| BUSI | 0.934(0.906–0.958)| 0.957 | 0.630 |
| GDPH | 0.915(0.895–0.934)| 0.971 | 0.453 |
| SYSUCC | 0.838(0.810–0.866)| 0.931 | 0.474 |
| 內部(pooled OOF,ensemble+TTA+T;sens/spec 為 in-sample)| 0.925 | 0.903 | 0.771 |

判別力:BUSI、GDPH 之 AUC 落於或接近內部值;BrEaST(0.802–0.902)與 SYSUCC(0.810–0.866)之 CI 上界皆低於內部 0.9254。混淆(tp/fp/fn/tn):BrEaST 90/91/8/63 · BUSI 156/80/7/136 · GDPH 364/238/11/197 · SYSUCC 674/152/50/137。

### 3.4 機率漂移(描述性)
良性中位校準機率:內部 0.063 → BrEaST 0.306 / BUSI 0.174 / GDPH 0.289 / SYSUCC 0.287;惡性維持 0.680–0.831;良性 ≥ 閾值之比例 0.37–0.59。Specificity 崩落與良性分佈右移同時出現(描述性觀察;無因果檢定)。*未檢定之詮釋:漂移幅度可能與資料集風格差異有關,但本研究未測量任何風格距離。*

### 3.5 判讀者比較
GDPH(κ 0.515):模型 0.971/0.453;Reader1 0.976/0.894;Reader2 0.979/0.513——**模型於兩軸皆低於兩位判讀者**;Reader1 之 specificity 遠高於模型(0.89 vs 0.45,點估計;未做任何檢定),Reader2 接近模型之 ROC 曲線。SYSUCC(κ 0.215):模型 0.931/0.474;Reader1 0.913/0.651;Reader2 0.993/0.132——模型 operating point 居兩判讀者之間。κ 為兩位判讀者之間之一致性,不涉及模型。**POST-HOC(2026-09-07,RESULTS.md「Post-hoc analyses」§4,描述性):** 模型於良性影像之假陽性中,被判讀者亦評為 ≥ 4a 之比例:GDPH reader1 34/238 = 0.143(其真陰性中為 0.061)、reader2 152/238 = 0.639(0.305);SYSUCC reader1 81/152 = 0.533(0.146)、reader2 143/152 = 0.941(0.788)——模型假陽性相對真陰性富集判讀者 ≥ 4a 呼叫,但 GDPH 高 specificity 判讀者對模型 86% 之假陽性評為 ≤ 3。

### 3.6 Grad-CAM(定性、單模型)
內部 TP 熱區於病灶本體與邊緣;內部 FP 3/4 聚焦真實可疑結構、1/4 顯示殘餘標註敏感;外部 FP 7/8 熱區落於病灶本體(SYSUCC 四張皆低回音分葉狀良性)——與 appearance-driven 之解讀相符,但 CAM 不能量化之。註(2026-09-07):該 GDPH/SYSUCC 外部 FP 圖庫因資料集無明確授權,不隨版本庫散布(僅作者本機保留);版本庫內改附 BrEaST 良性假陽性前 8 例之同設計圖庫(reports/gradcam_external_fp_breast.png,CC BY 4.0),其閱讀未納入上述 7/8 之陳述。

### 3.7 Recalibration 學習曲線
Sanity:k=0 逐位重現 external-v1;oracle 閾值 BrEaST 0.376 / BUSI 0.385 / GDPH 0.448 / SYSUCC 0.331,對應 frozen→oracle specificity:0.409→0.623 / 0.630→0.819 / 0.453→0.775 / 0.474→0.568。Pre-registered k*:GDPH、SYSUCC = 10;BrEaST、BUSI = 20;中位恢復比例近 1.0(GDPH spec 0.453 → k=10 中位 0.827,oracle 0.775)。**Sensitivity 代價:** k=10 之中位 sens 為 BrEaST 0.830 / BUSI 0.846 / GDPH 0.866 / SYSUCC 0.909,較凍結管線之外部 sens(≥ 0.918)為低;判準容許至 0.85。單類別抽樣僅見於 k=10(BrEaST 0.8% / BUSI 0.6% / SYSUCC 2.4% / GDPH 0)。k=10–30 之 95% specificity 帶涵蓋約 0–0.97;**post-hoc k_reliable(2.5 百分位恢復 ≥ 0.5)僅 GDPH 於 k=200 達標,BrEaST、BUSI(網格至 k=100)與 SYSUCC(至 k=200)皆未達**——於 pre-registered 網格內,無任何方法使小樣本重校準達到 draw-level 可靠。方法比較:M2b/M3 與 M1 為結構性等價(單調變換 + 排序空間閾值),實跑曲線於 k ≤ 30 僅因 M2 擬合失敗回退與 Platt 同分而異,k* 亦有差異(BrEaST M1 20 / M2b 10;GDPH M1 10 / M2b 20;SYSUCC M1 10 / M2b 20);M2a sens 最高(0.92–0.95)、帶最窄,但恢復不足(GDPH 0.657 vs oracle 0.775),k* 未達;T 擬合失敗率 k=10 為 7–20%、k ≥ 100 為 0。

### 3.8 跨場域閾值轉移(post-hoc、描述性)
內部參考列逐位重現(閾值 0.2683、T 2.3644)。四個外部閾值 0.3755(BrEaST)/ 0.3851(BUSI)/ 0.4484(GDPH)/ 0.3314(SYSUCC)均為各世代全資料之 in-sample 擬合。任一外部閾值於所有其他世代之 specificity 皆高於內部閾值(無 CI);GDPH 之 0.448 換得最高 spec(0.71–0.89)但跨場域 sens 降至 0.80–0.87。M2 temperature 轉移效果甚微(BrEaST T 2.27 / SYSUCC T 2.23,與內部差 0.03–0.04)。

### 3.9 Disagreement 與 abstention
重算平均於全部 2,454 張逐位重現(float32)。誤判 AUROC(U_std / margin):BrEaST 0.774/0.664、BUSI 0.856/0.754、GDPH 0.802/0.717、SYSUCC 0.626/0.747;**U_std 與 margin 之 CI 於 BUSI、GDPH 不重疊,於 BrEaST 重疊(0.710–0.831 vs 0.595–0.729)**。q=10% abstention(U_std,凍結→保留):BrEaST spec 0.409→0.453(sens 0.918→0.909,富集 ×1.53);BUSI 0.630→0.705(0.957→0.953,×2.64);GDPH 0.453→0.504(0.971→0.968,×1.77);SYSUCC 0.474→0.515(0.931→0.923,×1.14)。以 margin abstain 則方向相反(保留 sens 上升、spec 持平或降)——兩信號 abstain 不同影像。判準 0/4:條款 2 於四世代皆因 sens 條款未過(BUSI/GDPH 之 spec 增益達標);SYSUCC 之條款 1、3 亦未過(margin 反優)。依規定不放寬。*作者事後詮釋(非判準之一部分):失敗源於 sens 保留條款而非信號品質(SYSUCC 除外);任何放寬皆須新的 pre-registration。*

### 3.10 領域預訓練骨幹(Q2)
USFM 為 BEiT 式架構(無 absolute pos_embed;相對位置偏置表 + LayerScale 共 27 tensors 無槽位),無法潔淨載入 vanilla ViT → pre-registered BiomedCLIP 替代(150/150 tensors,訓練前聲明)。內部:五折 0.9170 ± 0.0220(五折皆低於 v1)、pooled OOF 0.9109;T 2.8645、閾值 0.2007(sens 0.901/spec 0.718,in-sample)。外部單發(各用自身凍結閾值):AUC 0.844/0.910/0.882/0.831(ΔAUC −0.010/−0.024/−0.033/−0.007;點估計皆較低,但各世代 v1/v2 之未配對 CI 皆重疊,未做配對檢定);判準所定義之良性漂移比 0.892/0.689/0.962/0.700163。**判準:分支 A 0/4、分支 B 1/4(SYSUCC 以 0.0002 之差計 fail)→ 「領域預訓練減少漂移」NOT CLAIMED**;LOCO 骨幹依規則 (v) 定為 v1 ImageNet ViT。註:此結果屬 BiomedCLIP 替代品,不代表超音波 MIM 預訓練。

### 3.11 多來源訓練(Q1,LOCO)

**表 3|LOCO vs v1-single(各自凍結閾值,單發)**

| Held-out | LOCO AUC(CI)| v1-single AUC | ΔAUC(paired CI)| LOCO sens/spec | v1-single sens/spec | 良性中位 v1→LOCO |
|---|---|---|---|---|---|---|
| BrEaST | 0.866(0.817–0.910)| 0.847 | +0.019(−0.016–+0.055)| 0.857/0.753 | 0.939/0.474 | 0.279→0.223 |
| BUSI | 0.937(0.912–0.959)| 0.907 | +0.030(+0.007–+0.055)| 0.847/0.880 | 0.939/0.607 | 0.150→0.051 |
| GDPH | 0.945(0.930–0.960)| 0.890(0.867–0.913)| +0.056(+0.037–+0.074)| 0.960/0.713 | 0.952/0.411 | 0.339→0.278 |
| SYSUCC | 0.844(0.817–0.870)| 0.832 | +0.013(−0.010–+0.035)| 0.800/0.716 | 0.934/0.450 | 0.318→0.171 |

(LOCO 閾值/T:breast 0.474/1.222;busi 0.412/2.480;gdph 0.541/0.9545;sysucc 0.497/0.876。訓練側 val sens 均 ≈ 0.90。Δspec:+0.279(0.196–0.362)/+0.273(0.209–0.338)/+0.301(0.248–0.357)/+0.266(0.212–0.321)。)

**判準:分支 A 4/4 → MET(via branch A)。** 同段保留事項:(i) paired ΔAUC CI 僅 BUSI、GDPH 排除零,BrEaST、SYSUCC 含零;(ii) 分支 B 2/4 未達(BUSI 0.847、SYSUCC 0.800 低於 0.85);(iii) 判準之對照為 v1 fold-5 **單模型**;相對已部署之 v1 **ensemble**(external-v1),LOCO ΔAUC 為 +0.0115 / +0.0033 / +0.0301 / +0.0064,僅 GDPH、BrEaST 兩世代 ≥ +0.01;(iv) 註冊之問題為 specificity 崩落,而通過之分支為 AUC;Δspec(+0.27 至 +0.30)四世代 paired CI 皆排除零,但係於各模型不同閾值下比較(閾值定位混淆)。機制(描述性):BrEaST 以閾值定位為主;BUSI 良性中位降至 0.051(全案最低);GDPH 判別力提升最明確(AUC CI 不重疊、唯一 LOCO sens 高於 v1-single 之世代);SYSUCC 良性下移同時拖累惡性(sens 0.800)。Sens 可攜性:val ≈ 0.90 → held-out 0.80–0.96,對照 v1 凍結閾值之全域 ≥ 0.918。

### 3.12 分割與展示
U-Net fold-5 Dice 0.902 / IoU 0.833(280/383 > 0.9;失效於陰影/低對比病灶)。展示系統本機 0.53 s;Space 冷啟 9.7 s / warm 6.8 s;重現性註記見 §2.8。

---

## 4. 討論

### 4.1 判別力可攜,校準不可攜——及其兩端修正
外部驗證確立 AUC 與 operating point 可攜性之分離,伴隨良性分佈之位置偏移(M2a 顯示僅重估 T 無法恢復,漂移為 logit 空間之位置而非尺度)。後續研究:(1) 部署端——10–20 例本地標註即可於中位數恢復 specificity(以 k=10 中位 sens 0.83–0.91 為代價),但 draw-level 可靠性於 pre-registered 網格內僅 GDPH k=200 達標;本地標籤之價值在閾值定位(單調校準 + 本地閾值 ≡ 閾值重定之結構性等價);(2) 跨場域(post-hoc、描述性)——任一外部閾值轉移至其他世代之 specificity 皆高於內部閾值;*未檢定之詮釋:BUS-BRA 之良性影像相對「乾淨」致閾值系統性偏低*;(3) 訓練端——BiomedCLIP 替代骨幹依判準 NOT CLAIMED;多來源訓練(LOCO)達成分支 A(ΔAUC +0.013 至 +0.056 vs 單模型,2/4 CI 含零)並於各自閾值下 Δspec +0.27–0.30,惟增益之大宗仍為「多場域 validation 產出更佳之起始閾值」,且 sensitivity 可攜性由 v1 之全域 ≥ 0.918 退為 0.80–0.96。**部署之最後一哩仍為以本地資料設定 operating point。**

### 4.2 失效方向與臨床定位
所有外部世代 sensitivity ≥ 0.918(v1 凍結閾值),specificity 0.41–0.63:外部誤判為假陽性(37–59% 良性被呼叫)而非漏診。此方向是否「可接受」未經任何危害分析;且 LOCO 模型於其自身閾值下 held-out sens 降至 0.80。與判讀者比較:GDPH 模型於兩軸皆低於兩位判讀者,SYSUCC 居兩者之間;κ 0.22–0.52 為判讀者間一致性,說明一致性工具之需求,不涉及模型表現。*未檢定之詮釋:外部 FP 之 Grad-CAM 熱區落於病灶本體,與「非典型良性外觀」相符;POST-HOC 逐影像交叉表(§3.5)顯示 GDPH reader1 僅對模型 14% 之假陽性評為 ≥ 4a,故「模型與判讀者被同一批影像誤導」對該判讀者不成立,對 reader2 僅部分成立;未做任何檢定。* 此比較存在資訊不對等(判讀者見完整檢查,模型僅見單張截圖)。

### 4.3 三個 pre-registered 負面結果
(1) CoarseDropout:gate 2 未過即棄、不迭代(gate 2 為三張影像之主觀判斷,未測任何 robustness 指標);(2) abstention 判準:0/4,依規定不放寬,*事後詮釋*為信號真實(AUROC 0.77–0.86,3/4 世代優於 margin)而判準將轉介計為 sensitivity 損失——修正之途徑為新的 pre-registration;(3) BiomedCLIP 替代骨幹:0/4 與 1/4,NOT CLAIMED。三者皆依註冊判準機械式套用、未於 RESULTS.md 放寬。

### 4.4 公開資料集品質
SYSUCC 35% 重複與同影像雙標籤、BUSI_WHU 標籤不可驗證:未經稽核之公開資料切分可同時引入 leakage 與 label noise。pHash 稽核成本低廉,宜為標準前置步驟(其限制見 §2.4)。

### 4.5 限制
單一國家單一機構之訓練資料;無台灣/NTUH 資料、無 IRB 臨床驗證、非醫療器材;**內部 CV AUC 為各折 best-epoch 值(POST-HOC 量化:固定 epoch 下 0.9195 ± 0.0164,樂觀量 ≈ 0.011),內部 sens/spec 為 in-sample(POST-HOC 嵌套估計 0.900 ± 0.057 / 0.780 ± 0.132,折間變異大)**;TTA 採用未 pre-registered;pre-registration 為內部版本控制、無外部時間戳、歷史曾重寫身分(§2.10);燒錄標註為潛在捷徑(無標註影像之表現預期較低);三世代無病人 ID(image-level bootstrap 與 LOCO val 切分之潛在樂觀);PPV 不可轉移;ensemble 無無偏內部估計;判讀者比較之資訊不對等與評分脈絡未知;中段機率殘餘 overconfidence;USFM 未於原生骨架受測;LOCO 為單模型(對照亦為單模型),early-stop patience 未預先指定;v2 檢查點與 fold 檔未公開;展示系統於非 BUS-BRA 尺寸存在 resize 分派差異(校準機率最大 7×10⁻³、中位 8×10⁻⁴,2/1,421 邊界翻轉)。

### 4.6 後續工作
(1) NTUH 回溯性研究(IRB 規劃中),第一階段約 100 例:不動模型、僅設 operating point(起始候選為四個外部 in-sample 閾值 0.331 / 0.376 / 0.385 / 0.448 之任一,須於本地資料上重選)並驗證台灣族群判別力;第二階段 BI-RADS 3/4A 追蹤 upgrade 預測;第三階段超音波腋下淋巴結轉移預測以支持腋下手術降階。(2) Abstention 判準之重新 pre-registration(workflow 層級 sensitivity 與轉介率)。(3) USFM 於原生 BEiT 骨架之比較;LOCO × 領域預訓練之組合。(4) BUSI_WHU 內容層級標籤復原。(5) 多視角/多模態(影像+報告)以縮小與判讀者之資訊差。(6) Provenance:v2 檢查點與 fold 檔公開、Zenodo/OSF 外部時間戳、(plan.md P3–P4;epoch 選擇偏差與 in-sample operating point 之 post-hoc 量化已於 P2 完成,見 §3.1、§3.2)。

---

## 5. 結論
在 patient-level 切分、內部 pre-registration 與凍結式單發驗證之紀律下,單一機構訓練之乳房超音波分類器於四個外部世代展現 AUC 0.84–0.93 之判別力,其 operating point 則不可攜(specificity 0.77 → 0.41–0.63),需本地設定。訓練端修正中,多來源訓練達成其 pre-registered 判準(僅 AUC 分支,2/4 CI 含零),領域預訓練替代品未達;部署端之本地閾值重選於中位數有效但小樣本下不可靠。外部誤判方向為假陽性;其臨床可接受性未經檢驗。本研究同時提供公開資料集稽核之實務範式,及可直接落地 NTUH 之分階段研究路徑。

---

## 附錄 A|圖表清單(依引用順序)

| 編號 | 檔案(reports/)| 用途 |
|---|---|---|
| 圖 1 | roc_oof_operating_point.png | 內部 ROC + 凍結工作點 |
| 圖 2 | prob_shift_external.png | 1×5 機率漂移 |
| 圖 3 | birads_comparison_gdph.png / _sysucc.png | 模型 vs 判讀者 |
| 圖 4 | gradcam_gallery_internal.png / gradcam_external_fp_breast.png | 可解釋性(外部 FP 圖庫以 BrEaST(CC BY 4.0)呈現;原 GDPH/SYSUCC 版本 gradcam_external_fp.png 因該資料集無明確授權,僅保留於本機、不隨版本庫散布,見 RESULTS.md Errata 2026-09-07) |
| 圖 5 | v2_recalib_M1_curves.png | 學習曲線 |
| 圖 6 | v2_recalib_methods.png | 方法比較 + 帶寬 |
| 圖 7 | v2_cross_site_matrix.png | 跨場域矩陣 |
| 圖 8 | v2_disagreement_auroc.png / v2_abstention_curves.png | Abstention |
| 圖 9 | v2_loco_summary.png | LOCO ROC + 分佈 |
| 圖 10 | seg_examples.png | 分割 |
| 補充 | reliability_oof_vit_tta.png、phash 證據表(phash_cross_pairs_table.csv + phash_cross_pairs_busbra_thumb.png、phash_within_d8_table.csv;含 GDPH/SYSUCC 影像之原始配對圖不隨版本庫散布)、v2_resize_dispatch_impact.csv | Backup |
