# BreastUS-CAD — 專案摘要(一頁,與修正版 report v2.2 同步)

> 建立於 2026-09-07(plan.md P5);2026-09-12 依兩份再稽核同步(P8')。版本庫、Drive 與先前 session 中均無此檔的舊版,故為新建;
> 每一數字皆取自 docs/report.md(依 2026-09-06 稽核與 2026-09-11 再稽核修正)與 RESULTS.md,不引入任何新數字。
> 研究原型,非醫療器材。DOI 10.5281/zenodo.22630912;https://github.com/thomasf45566/breast-us-cad

## 一句話
單一機構(BUS-BRA)訓練的乳房超音波良惡性分類器,凍結後於四個外部世代單次驗證:**判別力可攜(AUC 0.84–0.93),operating point 不可攜(specificity 0.77 → 0.41–0.63;內部值為 in-sample、且屬單一 held-out 檢查點預測器)**;部署前須以本地資料設定 operating point(小樣本重選之可靠性僅於中位數層面獲支持)。

## 方法(一段)
BUS-BRA 1,875 張 / 1,064 位病人,官方 patient-level 五折;ViT-B/16 + hflip TTA + temperature scaling;operating point 以 sens ≥ 0.90 規則於 pooled OOF 上預定後凍結(tag `frozen-v1`)。外部驗證依內部版本控制之 pre-registered protocol(+4 amendments)於 BrEaST、去重 BUSI、GDPH、SYSUCC(共 2,454 張)**單次**執行(tag `external-v1`),預測 CSV 各僅一個新增 commit、從未修改。

## 內部指標(兩個不同預測器,標示清楚)
- 單模型、無 TTA、各折取 best epoch 之五折 CV AUC **0.9307 ± 0.0161**;此為 best-epoch-on-the-reported-fold,POST-HOC 對固定 epoch 之敏感度 **0.011**(非無偏樂觀估計)。
- pooled OOF AUC **0.9254**——每張影像取其唯一 held-out 檢查點 + TTA + T;**已部署之五模型 ensemble 無無偏內部估計**;T 與閾值於此單檢查點分佈上擬合、外部套用於 ensemble,故內部→外部比較同時換了預測器與世代(RESULTS.md POST-HOC 5:世代效應遠大於預測器效應)。
- Operating point(0.2683;規則實作為 sklearn roc_curve 工作點中之最高合格者,窮舉規則相差一張影像):in-sample sens 0.903 / spec 0.771;POST-HOC 留折閾值敏感度分析(非嵌套模型評估)**0.900 ± 0.057 / 0.780 ± 0.132**,held-out sensitivity 於 **3/5 折低於 0.90**。

## 資料稽核
pHash 全對比對(五集合 1,875/252/379/846/1,559 張,集合內+集合間共 **12,056,505 對**,d ≤ 8 為候選):SYSUCC 1,559→1,013(507 重複、39 標籤衝突,含同影像雙標籤);GDPH 846→810;BUSI 386→379;跨集僅 2 個 d=8 候選,目視裁決均為不同掃描。結論是 **「pHash d ≤ 8 無經目視確認之跨集合近重複」,而非「零重疊」**(有限篩檢;此法不能排除 d > 8 之同病人再掃或裁切)。BUSI_WHU 因標籤對映無法驗證而排除。

## 外部驗證(external-v1,凍結閾值)
| 世代 | AUC(95% CI) | Sens | Spec |
|---|---|---|---|
| BrEaST(n=252) | 0.854(0.802–0.902) | 0.918 | 0.409 |
| BUSI(n=379) | 0.934(0.906–0.958) | 0.957 | 0.630 |
| GDPH(n=810) | 0.915(0.895–0.934) | 0.971 | 0.453 |
| SYSUCC(n=1,013) | 0.838(0.810–0.866) | 0.931 | 0.474 |

Sensitivity 於四世代為 **0.918–0.971**(合計仍有 76 例假陰性);specificity 崩落(561 例假陽性)伴隨良性校準後機率中位數 0.06 → 0.17–0.31(描述性)。誤判以假陽性為主而非全無漏診;此錯誤型態是否「安全」未經危害分析。

## 判讀者比較(GDPH κ 0.51;SYSUCC κ 0.22,皆為判讀者間一致性,不涉及模型)
On SYSUCC the model lies between the two readers; on GDPH it is below both. Its false positives overlap heavily with the more conservative reader (reader2 called 64% of them ≥4a) and barely with the best reader (reader1: 14%).(GDPH 良性影像:模型 238 個假陽性中,reader2 評 ≥ 4a 者 152/238 = 0.639,reader1 僅 34/238 = 0.143;描述性,未檢定。)

## 部署導向後續研究(Amendments 2–4,結果依註冊判準機械式套用)
- **本地重校準(M1):** pre-registered k* 10–20; the post-hoc reliability criterion was reached only on GDPH at k=200; k=10 sensitivity median 0.83–0.85(BrEaST/BUSI;GDPH/SYSUCC 為 0.87–0.91)。正斜率單調校準 + 本地閾值 ≡ 直接重定閾值(結構性等價;負斜率 Platt 擬合為例外:k=10 時 12/4/3/16 個 draw)。閾值規則之窮舉反事實改變 2,432/11,000 個 M1 draw 之閾值,k* 不變。跨場域閾值互借之 specificity 皆高於內部閾值(post-hoc、in-sample、描述性;GDPH 閾值使 sens 降至 0.80–0.87)。
- **Ensemble disagreement 作為 abstention 信號:** 判準 0/4 NOT USEFUL(sens 保留條款皆未過);U_std 誤判 AUROC 於 BrEaST/BUSI/GDPH 為 0.77–0.86,SYSUCC 0.63 反低於 margin 基線 0.75。
- **領域預訓練骨幹(BiomedCLIP 替代 USFM):** 判準 **NOT CLAIMED**(分支 A 0/4、分支 B 1/4;外部 AUC 點估計於四世代皆較低)。
- **多來源 LOCO 訓練:** criterion MET via the AUC branch (4/4), but 2/4 paired ΔAUC CIs include zero and the specificity branch failed 2/4; Δspec is compared at different thresholds per model, and how much of the gain is threshold placement is an untested hypothesis (descriptive decomposition on BrEaST only).(ΔAUC +0.013 至 +0.056 vs v1 fold-5 單模型;相對已部署 ensemble 僅 2/4 ≥ +0.01;Δspec +0.27–0.30 於各自閾值下;held-out sens 0.80–0.96;外部 15% 訓練側切片為影像層級。)

## 展示與部署
Gradio demo(HF Space happytommy/breast-us-cad):與驗證管線共用同一模型/校準/推論模組,前處理於非 BUS-BRA 尺寸有最大 0.0072 之校準機率差(已記錄);本機 0.53 s,Space warm 約 5–9 s(六次量測之中位數 4.8–8.6 s,RESULTS.md Errata 2026-09-07 / 2026-09-12);線上 vs 本機四範例最大差 1.07×10⁻⁷。權重(v1 + v2)與 SHA-256 於 HF;BUS-BRA 官方 fold 檔已提交並於載入時驗證雜湊。

## Provenance 與限制(誠實版)
Pre-registration 為內部版本控制、無外部時間戳;2026-08-31 曾重寫作者身分(日期與內容保留,對映表見 docs/PROVENANCE.md);HF 權重庫與 Space(08-30)晚於 external-v1(08-29),故凍結先於外部評分之時序只有自我認證;`--confirm` 當時僅為意圖旗標,單發之證據為 git 歷史(2026-09-12 起腳本拒絕覆寫既有預測檔)。TTA 採用未 pre-registered。三個外部 FP/pHash 圖因 GDPH/SYSUCC 無明確授權不隨版本庫散布。獨立稽核(2026-09-06)與逐項回應見 docs/AUDIT_2026-09-06.md、docs/AUDIT_RESPONSE_2026-09-06.md;兩份再稽核(2026-09-11)與回應見 docs/AUDIT2_claude_2026-09-11.md、docs/AUDIT2_astra_2026-09-11.md、docs/AUDIT2_RESPONSE.md;首個外部時間戳為 Zenodo DOI(2026-09-07),其封存檔內嵌之 provenance 文字仍描述 re-point 前之 commit(已記錄之缺陷;v1.1-audited 封存檔以 .git-commit 自述 commit)。

## 三個 pre-registered 負面結果
CoarseDropout(gate 2 未過即棄)、abstention 判準 0/4、BiomedCLIP NOT CLAIMED——三者皆依註冊判準機械式套用、未放寬。
