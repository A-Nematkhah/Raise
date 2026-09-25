# What’s new?!

لاگ تغییرات پروژه — هر اصلاح مهم را به ترتیب زمانی، با «قبل / بعد» می‌نویسیم.

---

## Score1 (فیلتر ارزان Stage I) — ۲۰۲۶-۰۹-۱۹

Score1 امتیاز تحلیلی Stage I است: روی trajectoryهای ذخیره‌شده، رتبهٔ قوانین ترجیح
(موفقیت ≫ سایر ≫ شکست؛ داخل موفقیت مسیر کوتاه‌تر بهتر) با رتبهٔ پاداش تجمعیِ
پیشنهادی LLM مقایسه می‌شود و میانگین Spearman فیلتر ارزان قبل از RL گران است.
هدف این بسته جایگزینی Stage II/III نبود؛ سخت‌تر و راهنماتر کردن همین فیلتر بود تا
کاندیدهای بی‌فایده زود حذف شوند و mutation کور نباشد.

### Holdout و گیت‌های سخت‌گیر

**قبل:** کل دیتاست یکجا وارد Score1 می‌شد و همان عدد هم رتبه‌بندی نسل و هم تصمیم
خوب/بد را می‌ساخت. پاداش ثابت یا کم‌واریانس روی خیلی از فریم‌ها Spearman را NaN
می‌کرد؛ `degenerate_fraction` بیشتر گزارش/هشدار بود و کاندید لزوماً حذف نمی‌شد.
Holdout رسمی هم نبود: حفظه روی همان trajها می‌توانست Score1 بالا بگیرد و هزینه
تازه در Stage II/III معلوم شود.

**بعد:** دیتاست روی شناسهٔ scenario به train/holdout (~۳۰٪، seed ثابت) شکسته
می‌شود؛ با یک scenario، holdout خالی می‌ماند. تکامل فقط با Score1 روی train جلو
می‌رود و holdout جدا ذخیره می‌شود. اگر degenerate_fraction ≥ ۰٫۵ باشد، hard-reject
با `score=-inf` است. اگر `train − holdout` از حاشیهٔ ۰٫۴ بیشتر باشد، به‌عنوان
exploit/حفظه دوباره reject می‌شود. متادیتای `score1_rejected` /
`score1_reject_reason` / `score1_train` / `score1_holdout` / `score1_raw` در
evolver ثبت می‌شود. منطق در `split_stage1_dataset`، `score1_for_dataset`،
`make_score1_fn` و `domains/crowdnav/stage1.py` است.

### Diagnostics سناریو → reflection و mutation

**قبل:** reflection نسل عمدتاً لیست امتیاز و جملات قالبی («هدف قوی‌تر»، «برخورد
کمتر») بود. Mutation روی نیمه‌ی ضعیف همان متن سراسری را می‌دید؛ مشخص نبود ضعف
روی کدام scenario است، پس ویرایش‌ها اغلب عمومی و کم‌هدف بودند.

**بعد:** برای هر scenario میانگین Spearman در `scenario_scores` می‌آید.
`format_score1_failure_hints` بدترین سناریوها، degeneracy، reject و در صورت وجود
train/holdout را در یک جمله می‌چیند و در `score1_failure_hints` می‌گذارد. این متن
وارد reflection underperformerها و weakness پرامپت mutation همان parent می‌شود،
بدون عوض کردن sandbox یا امضای `compute_reward(state, memory)`.

### Recollect کامل دیتاست

**قبل:** `data/stage1_dataset/stage1_dataset.npz` قدیمی بود و با schedule متنوع
فعلی collector هم‌تراز نبود؛ holdout و ادعای fidelity روی دادهٔ کهنه ضعیف بود.

**بعد:** پاس کامل M=100 × N_traj=10 با `collect_stage1_dataset.py` در رژیم
`without_random` اجرا شد (~۲۰ دقیقه) و مسیر پیش‌فرض را عوض کرد. بکاپ در
`data/stage1_dataset_backup_20260919_231634` ماند. حدود ۱۰۰۰ traj با
success/collision/timeout و ORCA/SF/noise/random؛ میانگین یکتایی fingerprint داخل
هر scenario حدود ۹ از ۱۰. از این نقطه گیت‌های بالا روی دادهٔ تازه معنا دارند.

### جمع‌بندی

ایدهٔ مقاله (Spearman با قوانین تحلیلی) همان است، ولی دیگر میانگین خام روی همهٔ
داده نیست: داده تازه، holdout برای تعمیم، reject سخت برای degenerate/حفظه، و ضعف
قابل‌مصرف در پرامپت تکامل. کارهای بعدی (مثل ranking چندهدفهٔ Stage II) جدا در
همین فایل ثبت می‌شوند.

---

## Surrogate + Active Learning — اسکلت و پلن فنی (۲۰۲۶-۰۹-۲۰)

**قبل:** فقط در brainstorm بود؛ مسیر کد و قرارداد فیچر/برچسب مشخص نبود.

**بعد:** دو بستهٔ اسکلت با stub و سند فنی اضافه شد:

- `crowd_nav/reward_search/surrogate/` + `PLAN.md`
- `crowd_nav/reward_search/active_learning/` + `PLAN.md`
- `data/surrogate_dataset/`, `data/active_learning/`, `artifacts/surrogate/`
- CLI stub: `scripts/bootstrap_surrogate.py`, `scripts/run_active_learning_step.py`

منطق هنوز `NotImplementedError` است؛ ترتیب کار: اول Surrogate v1، بعد AL.

---

## Surrogate v1 — قفل قرارداد + پیاده‌سازی bootstrap (۲۰۲۶-۰۹-۲۰)

**قبل:** اسکلت با `NotImplementedError`؛ target پیشنهادی پلن اولیه `scalar` بود؛ به pipeline وصل نبود.

**بعد (قفل‌شده):**

- `PLAN.md` با تصمیم‌های L1–L10 قفل شد: هدف آموزش **`SR, CR, TR`** (نه scalar)؛
  `scalar` فقط در لیبل برای لاگ/سازگاری نوشته می‌شود.
- پیاده‌سازی: `dataset_io`, `features`, `model` (ensemble RF + uncertainty)،
  `bootstrap.run_bootstrap`, CLI واقعی `scripts/bootstrap_surrogate.py`.
- تست‌ها: `crowd_nav/reward_search/tests/test_surrogate.py` (شامل `--fast` / stub).
- وابستگی: `scikit-learn` + `joblib` در `requirements_pinned.txt`.
- **هنوز** به `RaisePipeline` وصل نیست؛ Active Learning همچنان بعد از این است.

اجرای سریع wiring:

```bash
cd raise_env
python scripts/bootstrap_surrogate.py --fast --force --out data/surrogate_dataset --model-out artifacts/surrogate
```

---

## Active Learning v1 — روی Surrogate (۲۰۲۶-۰۹-۲۰)

**قبل:** فقط stub + `PLAN.md`؛ بدون صف پایدار یا acquire واقعی.

**بعد:**

- `query.score_queries` با وزن‌های uncertainty / disagreement / borderline / diversity
- صف jsonl پایدار (`queue` / `done` / `steps` / `manifest`)
- `acquire`: `stage2_label` کامل (reuse `label_and_append_candidate`)؛
  `stage1_scenario` امن به‌صورت request در `stage1_extra/` (بدون دست‌زدن به npz مقاله)
- `loop.run_active_learning_step` + CLI `scripts/run_active_learning_step.py`
  (بدون مدل → exit 2 / `Surrogate model required`؛ `--force-refit` مدل را دوباره fit می‌کند)
- تست‌ها: `tests/test_active_learning.py`
- هنوز به `RaisePipeline` وصل نیست (opt-in بعدی)

```bash
python scripts/run_active_learning_step.py --surrogate artifacts/surrogate --fast \
  --candidates results/<run>/stage1_population.json --force-refit
```

---

## اتصال Surrogate / AL به Pipeline (۲۰۲۶-۰۹-۲۰)

**قبل:** Surrogate و AL فقط CLI جدا بودند؛ `run_raise` از آن‌ها استفاده نمی‌کرد.

**بعد:** opt-in در `RaisePipeline` / `run_raise.py`:

- `--surrogate DIR`: بعد از Stage I پیش‌بینی و `surrogate_preds_stage1.json`
- قبل از Stage III: دوباره predict + `gate` (حذف confident-weak؛ `--no-surrogate-gate` خاموشش می‌کند)
- `--active-learning`: یک گام AL بعد از Stage I (نیاز به مدل)
- خلاصه در `manifest.json` → کلید `surrogate`

پیش‌فرض بدون فلگ: رفتار Algorithm 1 قبلی بدون تغییر.

---

## Surrogate bootstrap هم‌تراز Gen0 + پایداری ویندوز (۲۰۲۶-۰۹-۲۰)

**قبل:** `bootstrap_surrogate.py` جمعیت را با پرامپت خام (`generate a reward variant`)
می‌ساخت؛ با ران اصلی یکی نبود و Groq اغلب `import math` می‌داد. فلگ‌های CLI روی
`sys.argv` می‌ماندند و workerهای Stage II روی ویندوز `get_args()` را با همان argv
می‌دیدند → crash / BrokenPipe. `num_processes` هم تا ۱۶ می‌رفت و RAM را خالی می‌کرد.

**بعد:**

- `_build_population` مستقیماً `StageIEvolver.initialize_population` (همان Gen0
  pipeline) را صدا می‌زند — همان D1 batch + regen + validator.
- بعد از parse، argv مثل `run_raise` ایزوله می‌شود.
- پیش‌فرض `--num-processes 1` (قابل افزایش)؛ مناسب RTX کوچک / ویندوز.
- نرمال‌سازی کد: حذف `[0]`/`[1]`/`[-1]` از فیلدهای اسکالر RewardState
  (`state.robot.px[0]` و مشابه) قبل از sandbox؛ پرامپت D1/D2 هم صریح‌تر شد.

```powershell
python scripts/bootstrap_surrogate.py --force --n-candidates 50 `
  --stage2-train-steps 4000 --k2-unit gradient_steps --llm groq `
  --device cuda --num-processes 1
```

---

## اصلاح انتخاب best + snapshot یکتا (۲۰۲۶-۰۹-۲۰)

**قبل:** `pick_candidate_by_ranking` با dict روی `candidate_id` آخرین snapshot همان
id را برمی‌گرداند؛ refine همیشه `*_v2` می‌ساخت و snapshotهای چند round روی یک id
می‌افتادند. نتیجه: `best_stage2.json` / `best_stage3.json` می‌توانست بدتر از رتبهٔ
اول R2/R3 باشد.

**بعد:**

- هر trained snapshot id یکتا دارد: `{id}__r{round}_s{index}`.
- پسوند refine افزایشی است (`_v2` → `_v3` → …).
- اگر هنوز id تکراری در pool باشد، بین آن‌ها بالاترین navigation scalar انتخاب می‌شود.

---

## Collect Stage I — seed بعد از reset (۲۰۲۶-۰۹-۲۰)

**قبل:** `np.random.seed(traj_seed)` قبل از `env.reset()` بود؛ reset دوباره RNG را
از seed سناریو می‌نوشت → دو traj نوع random داخل یک scenario بیت‌به‌بیت یکی بودند.

**بعد:** اول `_reset_scenario` (layout ثابت سناریو)، بعد `np.random.seed(traj_seed)`
برای نویز/random. دیتاستهای قدیمی نیاز به recollect دارند تا این fix اعمال شود.

---

## Proxy consistency با lineage بعد از refine (۲۰۲۶-۰۹-۲۰)

**قبل:** Spearman II↔III فقط روی fingerprint کد مشترک بود؛ بعد از D.3 موفق، overlap
تقریباً خالی می‌شد و ρ اغلب `null` / غیرقابل‌استفاده می‌ماند.

**بعد:** روی refine، `parent_genome_key` ذخیره می‌شود و گزارش
`stage2_vs_stage3_lineage` امتیاز Stage III را با امتیاز Stage II والد هم‌تراز
می‌کند. در `manifest` هم `stage2_vs_stage3_lineage_rho` و
`stage2_vs_stage3_preferred` آمده است.

---

## Stage I دقیقاً ۲/۴/۲ مگر `--elitism` (۲۰۲۶-۰۹-۲۰)

**قبل:** نسل بعد همیشه top-1 را نگه می‌داشت و یک اسلات از random می‌دزدید → عملاً
۲/۴/۱؛ فلگ `--elitism` روی این رفتار Stage I اثر نداشت.

**بعد:** پیش‌فرض `keep_runtime_elite=False` → دقیقاً ۲ crossover / ۴ mutation /
۲ random. با `--elitism` همان keep-top-1 قبلی (و inject/protect در II/III) روشن
می‌شود. `BASELINE_REPORT.md` Part C با این defaults هم‌تراز شد و یادداشت هزینهٔ
K2×num_processes اضافه شد.

---

## Closed-loop multi-fidelity (نوآوری) — ۲۰۲۶-۰۹-۲۰

**قبل:** مسیر خطی Alg.1 (همهٔ Stage I، بعد همهٔ Stage II). Surrogate/AL فقط
opt-in جدا بعد از Stage I یا قبل از Stage III بودند و به Genهای بعدی Stage I
فیدبک نمی‌دادند.

**بعد:** فلگ `--closed-loop` یک حلقهٔ نسل‌به‌نسل روشن می‌کند:

1. Gen0: Score1 → Stage II کوتاه روی همه → fit Surrogate  
2. Gen≥1: Score1 → Surrogate gate → **AL داخل حلقه** (uncertain/disagree) →
   Stage II روی survivors∪AL → refit  
3. اختیاری بعداً polish Stage II + Stage III  

بسته: `crowd_nav/reward_search/raise_loop/` + `PLAN.md`. بدون فلگ رفتار قبلی
دست‌نخورده می‌ماند.

```powershell
python scripts/run_raise.py --fast --closed-loop --allow-seed-llm `
  --output-dir results/closed_loop_fast --surrogate artifacts/surrogate `
  --surrogate-dataset data/surrogate_dataset
```

---

## Rename branding → RAISE (۲۰۲۶-۰۹-۲۲)

برندینگ و مسیرهای پروژه یکدست روی **RAISE** شدند:

- پوشهٔ محیط: `raise_env/`
- CLI: `run_raise.py`, `run_raise_paper_scale.py`, `eval_raise_checkpoint.py`, …
- API: `RaisePipeline` / `RaiseRunConfig` / `RaiseArtifacts`
- CI: `.github/workflows/raise-ci.yml`
- ماژول‌های Stage: `explore` / `refine` / `validate`
- README ریشه و citationها روی RAISE (arXiv:2605.11859)

---

## Stage III crash-safe resume (۲۰۲۶-۰۹-۲۳)

**قبل:** اگر Stage III وسط PPO یا بین کاندیدها قطع می‌شد، باید از صفر
شروع می‌کرد (مگر paper-scale `CheckpointStore`).

**بعد:**
- `{output_dir}/stage3/checkpoint.json` + `RESUME.json` بعد از هر کاندید
- mid-PPO: هر `--stage3-save-interval` آپدیت، وزن `.pt` + `train_progress.json`
- `--resume` / `--no-resume` هم برای closed-loop و هم Stage III
- بقیهٔ کاندیدها / راندها از همان‌جا ادامه می‌یابند؛ پروتکل eval عوض نشده

---

## Highway anti-hack + fitness رسمی + Pareto ranking — ۲۰۲۶-۰۹-۲۵

دامنهٔ `highway` (CrowdNav همچنان frozen). هدف: جلوگیری از reward hacking
(کروز ثابت / خزیدن با SR بالا) و یکی‌کردن معیار انتخاب نسل بعد.

### Holdout eval + soft_success@۲۰ + diversity

**قبل:** Stage II/III فقط روی همان توزیع train (`vehicles_count=20`) ارزیابی
می‌شد؛ `soft_success` با آستانهٔ ۱۵ m/s بود؛ نسل بعد می‌توانست چند کلون با
متریک یکسان را دوباره breed کند.

**بعد:**
- بعد از PPO، eval متراکم‌تر holdout (`vehicles_count=32`)؛ انتخاب از
  `last_metrics.holdout` (train در `train_dist` فقط برای لاگ)
- `soft_success`: زنده ماندن + سرعت ≥۲۰ m/s + پیشرفت ≥۴۰۰ m
- قبل از `_next_generation`، `diversify_ranking` کلون‌های fingerprint متریک
  یکسان را عقب می‌اندازد (`raise_core/raise_loop/diversity.py`)

### Score1 decoys (lag / decoy_crash)

**قبل:** فقط crawl خیلی کند (~۱۲ m/s) به‌عنوان مثال منفی؛ پاداش‌های
collision-loving و lag پشت ترافیک (~۱۵ m/s) ضعیف پوشش داده می‌شدند.

**بعد:** collector رفتارهای `lag` / `decoy_crash`؛ Score1 جریمهٔ crawl/lag و
`collision_decoy_penalty` دارد (`domains/highway/stage1.py`,
`scripts/collect_highway_stage1_dataset.py`).

### یک fitness رسمی (نام‌گذاری)

**قبل:** Score1، sum reward PPO، و «scalar» سه‌معیار جدا بودند و در لاگ قاطی
می‌شدند.

**بعد:** هدف رسمی highway: `fitness = highway_fitness(...)` روی holdout؛
کلید `fitness` (+ alias `selection_scalar`)؛ لاگ‌ها `fitness=` /
`evolve_best_fitness`. Score1 فقط پروکسی Stage I می‌ماند.

فرمول وزن‌دار (با gate سیگموید روی `v_eff` / `v_floor`) هنوز برای surrogate /
لاگ موجود است؛ **ترتیب والدین نسل بعد دیگر از آن استفاده نمی‌کند.**

### Pareto ranking به‌جای weighted sum برای evolve

**قبل:** `evolve_rank=scalar` والدین را با جمع وزن‌دار دستی
(۰٫۳۵/۰٫۲۵/۰٫۱۵، `v_min`، tanh، …) مرتب می‌کرد — همان failure mode پاداش
دستی.

**بعد:** پیش‌فرض highway `evolve_rank=pareto`:
1. آستانه‌های feasibility از rollout مرجع IDLE/IDM (یا percentile نسل؛ با کف
   مطلق `V_FLOOR` تا all-crawler خودش را feasible نکند)
2. مرتب‌سازی NSGA-II روی متریک خام (SR, −CR, −TR, progress, speed,
   soft_success) + crowding distance
3. ماژول: `domains/highway/pareto_rank.py`؛ پروفایل 4h و CLI
   `--closed-loop-evolve-rank pareto`

ترتیب مرجع دمو: balanced → fast_risky → crawler (بدون وزن دستی).

### اصلاحات audit اتصالات (همان روز)

- `stamp_pareto_ranks` دیگر `fitness` را با `n−rank` overwrite نمی‌کند (فقط
  `pareto_rank` / `pareto_score`) — وگرنه `--final-rank scalar` خراب می‌شد
- feasibility با `speed_p10` هم‌تراز anti-hack (نه فقط `mean_speed`)
- پاک‌کردن stamp کهنه بعد از eval جدید؛ `pick_best` فقط وقتی همه stampedاند
  از rank استفاده می‌کند
- resume: بارگذاری `pareto_reference.json` تا آستانه وسط ران جابه‌جا نشود

CrowdNav: `evolve_rank` پیش‌فرض همچنان `score1`؛ رفتار baseline دست‌نخورده.

```powershell
python scripts/run_raise_highway_4h.py
# یا صریح:
python scripts/run_raise.py --domain highway --closed-loop `
  --closed-loop-evolve-rank pareto ...
```

---

## Highway wall-clock speed (same K2/K3) — ۲۰۲۶-۰۹-۲۵

**قبل:** هر کاندید Stage II سریال، یک env، PPO از صفر؛ eval همیشه train+holdout.

**بعد (فقط highway):**
- `DummyVecEnv` با `--highway-n-envs` (پروفایل 4h: ۴)
- لیبل موازی با `--highway-label-workers` (پروفایل: ۲) + قفل append دیتاست
- warm-start: وزن policy والد روی PPO تازه با `n_steps`/`batch` درست
  (نه `PPO.load` + mutate — آن با VecEnv باعث `IndexError` در buffer می‌شد)
- `--highway-eval-mode both|holdout_only`

CrowdNav بدون تغییر رفتار. بودجه env-step همان است.

---

*ادامهٔ تغییرات بعدی از همین‌جا اضافه شود.*
