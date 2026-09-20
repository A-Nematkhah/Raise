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
- **هنوز** به `EvoNavPipeline` وصل نیست؛ Active Learning همچنان بعد از این است.

اجرای سریع wiring:

```bash
cd evonav_env
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
- هنوز به `EvoNavPipeline` وصل نیست (opt-in بعدی)

```bash
python scripts/run_active_learning_step.py --surrogate artifacts/surrogate --fast \
  --candidates results/<run>/stage1_population.json --force-refit
```

---

*ادامهٔ تغییرات بعدی از همین‌جا اضافه شود.*
