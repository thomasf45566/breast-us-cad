# BreastUS-CAD — 面試口述稿(與修正版 report v2.1 同步)

> 建立於 2026-09-07(plan.md P5)。版本庫、Drive 與先前 session 中均無此檔的舊版,故為新建;
> 每一數字皆取自 docs/report.md / RESULTS.md,不引入任何新數字。研究原型,非醫療器材。
> 結構:A 開場 60 秒 → 依對方興趣展開 B1–B6(各約 45 秒)→ C 常見追問。

---

## A. 開場(約 60 秒)

我做的是乳房超音波良惡性分類器,重點不在模型,而在**驗證紀律**。用公開的 BUS-BRA(1,875 張、1,064 位病人)按官方 patient-level 五折訓練 ViT,凍結模型、校準與 operating point 之後,依事先寫進版本控制的 protocol,在四個外部世代——波蘭 BrEaST、埃及 BUSI、廣東 GDPH、中山 SYSUCC,共 2,454 張——**只跑一次**。結果:AUC 0.84 到 0.93,判別力跨洲際可攜;但凍結閾值下 sensitivity 雖然都 ≥ 0.918,specificity 從內部 0.77 掉到 0.41–0.63——operating point 不可攜。後續四個 amendments 從部署端與訓練端分別逼近,結論一致:**部署的最後一哩是用本地資料設定 operating point**。整個專案剛經過一次獨立對抗式稽核,文件已逐句修正,程式碼、權重、fold 檔與 DOI 都公開。

---

## B. 展開模組(每段約 45 秒)

### B1 — 內部指標怎麼讀(不誇大)
內部有兩個不同的預測器,我分開講。單模型、無 TTA 的五折 CV AUC 是 0.9307 ± 0.0161,但那是每折取 validation 最佳 epoch 的值——稽核後我用 wandb 歷史做了 post-hoc 量化,對固定 epoch 的樂觀量約 0.011。五模型 ensemble 加 TTA 加 temperature 的 pooled OOF AUC 是 0.9254。凍結 operating point 的 sens 0.903 / spec 0.771 是 in-sample,閾值就在同一批 OOF 上選的;嵌套 out-of-sample 估計是 0.900 ± 0.057 / 0.780 ± 0.132,而且 held-out sensitivity 在 5 折裡有 3 折低於 0.90。所以「sens ≥ 0.90」是擬合集的性質,不是對新病人的保證——這和外部驗證看到的是同一件事。

### B2 — 資料稽核與外部驗證
公開資料不能直接信。我對五個集合做全對感知雜湊比對,12,056,505 對,d ≤ 8 為候選:SYSUCC 有 35% 重複、39 個標籤衝突,同一張圖同時被標良性和惡性;BUSI_WHU 標籤對映驗不出來,排除。結論要講精確:是「pHash d ≤ 8 無近重複」,不是「零重疊」,因為這方法排不掉 d > 8 的同病人再掃。外部驗證單次執行、結果直接入帳:四世代 AUC 0.854 / 0.934 / 0.915 / 0.838,specificity 崩落伴隨良性校準機率中位數從 0.06 右移到 0.17–0.31——模型把外部的良性看得「更可疑」。誤判方向是假陽性不是漏診,但這方向算不算安全,我沒做危害分析,不敢說。

### B3 — 判讀者比較
GDPH 和 SYSUCC 各有兩位放射科醫師的 BI-RADS,以 ≥ 4a 為陽性。On SYSUCC the model lies between the two readers; on GDPH it is below both. Its false positives overlap heavily with the more conservative reader (reader2 called 64% of them ≥4a) and barely with the best reader (reader1: 14%). κ 0.22 和 0.51 是兩位醫師之間的一致性,跟模型無關,只說明一致性工具有需求。我原本寫「模型和醫師被同一批影像用同一種方式騙」,稽核指出沒有逐影像分析支持;後來做了交叉表,對 GDPH 最好的那位醫師來說根本不成立——模型 86% 的假陽性她評 ≤ 3。

### B4 — 部署端:本地重校準與 abstention
Amendment 2:模擬新場域抽 k 張本地標註重選閾值。Pre-registered k* 10–20; the post-hoc reliability criterion was reached only on GDPH at k=200; k=10 sensitivity median 0.83–0.85(BrEaST/BUSI)。也就是中位數上 10–20 張就接近 oracle,但單一場域抽 10 張可能落在任何地方。任何單調校準接本地閾值,結構上等價於直接重定閾值——本地標籤的價值全在閾值定位。Amendment 3 問 ensemble 分歧能不能當轉介信號:判準 0/4,因為 sens 保留條款全部沒過;信號本身在三個世代 AUROC 0.77–0.86,SYSUCC 反而輸給 margin 基線。依規定不放寬;要改判準就是新的 pre-registration。

### B5 — 訓練端:BiomedCLIP 與 LOCO
Amendment 4 兩個問題。Q2 換領域預訓練骨幹:USFM 載不乾淨,依預先寫好的 fallback 用 BiomedCLIP;結果依判準是 **NOT CLAIMED**——分支 A 0/4、分支 B 1/4,外部 AUC 點估計四世代都較低。Q1 多來源 LOCO 訓練:criterion MET via the AUC branch (4/4), but 2/4 paired ΔAUC CIs include zero, the specificity branch failed 2/4, and the gain is largely threshold placement。對照是 v1 fold-5 單模型;相對已部署的 ensemble 只有 2/4 世代 ≥ +0.01;held-out sensitivity 0.80–0.96。所以訓練端強化不取代本地校準——四條線殊途同歸。

### B6 — 獨立稽核(45 秒內)
稽核在乾淨 clone 裡重算了所有東西。**確認的**:外部驗證四世代的每個數字、patient-level 切分、資料稽核的 keep-list,全部逐位重現。**抓到的**:文件在八個地方漂離資料——BiomedCLIP 寫成有效、判讀者比較寫錯方向、CI 不重疊多算一個世代、pHash 對數無來源、還有幾句根本沒有出處。**我做的**:逐句修正文件、一個數字都沒改;把兩個 disclosure 事後量化——epoch 選擇偏差和 in-sample operating point;公開 v2 權重、fold 檔、DOI;repo 轉公開。這比模型本身更能說明我怎麼做研究。

---

## C. 常見追問(短答)

- **為什麼 pre-registration 還被質疑?** 因為是內部版本控制、沒有外部時間戳,而且 08-31 重寫過作者身分(日期與內容保留,對映表在 PROVENANCE.md)。第一個外部時間戳是 2026-09-07 的 Zenodo DOI,對 external-v1 而言是事後的。以後每個 protocol 先上 OSF 再算。
- **TTA 是 pre-registered 嗎?** 不是。凍結前看到 +0.0023 才採用,作用在後來拿去擬合 T 和閾值的同一批 OOF 上。已揭露。
- **Demo 跟驗證管線一樣嗎?** 同一模型、校準、推論模組;前處理在非 BUS-BRA 尺寸上有 ≤ 7×10⁻³ 的校準機率差,已記錄。線上 vs 本機四範例最大差 1.07×10⁻⁷;Space warm 約 6–9 秒。
- **三個負面結果?** CoarseDropout(gate 2 沒過就棄,不迭代)、abstention 0/4、BiomedCLIP NOT CLAIMED。都依註冊判準機械式套用。
- **接下來?** NTUH 回溯性研究第一階段約 100 例:不動模型、只設 operating point(起始候選為四個外部 in-sample 閾值 0.331 / 0.376 / 0.385 / 0.448 之一,須本地重選),protocol 先登 OSF。
