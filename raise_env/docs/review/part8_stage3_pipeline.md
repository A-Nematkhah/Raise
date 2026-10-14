<div dir="rtl">

# بررسی بخش ۸: Stage III، پایپ‌لاین و CLI

**دامنه:**

- `raise_core/validate.py` (~۳۵۰ خط: `Stage3Runner`، `Stage3Config`)
- `raise_core/stage3_checkpoint.py`
- `raise_core/pipeline.py` (`RaiseRunConfig`، `RaisePipeline`)
- `raise_core/console.py`
- `raise_core/domains/*`، `domains/highway/pack.py`
- `scripts/run_raise.py`
- مسیر آموزش Stage III در `domains/highway/adapter.py`

نقاط اتصال هم بررسی شد: `selection.pick_best_trained`، seedهای holdout (`HOLDOUT_SEED_OFFSET`)، کارت بازخورد (بخش ۳)، نویز fitness (بخش ۵).

**روش:**

- کد خط به خط خوانده شد.
- seedهای ارزیابی Stage II (هر نسل) با Stage III روی پیکربندی پیش‌فرض (`seed=425`) مقایسه شد.
- اجرای واقعی `results/highway_run` (N=6، G=4، K2=12k، K3=70k، E=30، CUDA، Groq) برای fitness و `fitness_se` و برنده‌ی نهایی بررسی شد.
- شبیه‌سازی Monte Carlo برای winner's curse با `se≈0.20` و N=6 اجرا شد.
- نگاشت CLI ← `RaiseRunConfig` ← `ClosedLoopConfig` / `Stage3Config` بررسی شد.

**سطح‌بندی:** همان بخش‌های قبل (H / M / L).

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

</div>

```text
RaisePipeline.run
  ├─ ClosedLoopRunner (G generations, K2, E2)
  ├─ best_stage1 / best_stage2 artifacts
  ├─ Stage3Runner: for each final candidate
  │     strip Stage II fitness → PPO K3 → eval E3 → checkpoint
  ├─ best_stage3 / final_candidate.json / manifest.json
  └─ write_highway_run_artifacts (plots + GIF) if not --fast
```

<div dir="rtl">

- Stage III همهٔ جمعیت نهایی را **از نو** آموزش می‌دهد (نه فقط elite). نتایج Stage II از متادیتا پاک می‌شوند تا برنده‌ی نهایی روی fitness کوتاه Stage II انتخاب نشود.
- خطای یک کاندید با `try/except` گرفته می‌شود (`status=failed`) و اجرا ادامه می‌یابد (اصلاح H2 بخش ۲).
- Resume از `stage3/checkpoint.json`؛ drift روی چند فیلد کلیدی هشدار می‌دهد.
- پیش‌فرض‌ها: N=6، G=4، K2=12 000، E2=30، K3=70 000، E3=30، `highway_warm_start=False`، `highway_eval_mode=holdout_only`.

---

## ۲. یافته‌ها

### H1. انتخاب برنده‌ی Stage III با E3=30 از نظر آماری قابل اتکا نیست (winner's curse)

**محل:** `validate.py` حلقه‌ی `run` + `candidate_fitness` / به‌روزرسانی `best_trained`؛ ارزیابی در `adapter.py` با `eval_episodes=30` و `fitness_se`.

**شواهد (اجرای `highway_run`):**

| کاندید | fitness Stage III | `fitness_se` |
|---|---|---|
| cro_0014 | −۰٫۵۲۴ | ۰٫۱۹۶ |
| ran_0023 | −۰٫۵۲۴ | ۰٫۱۹۲ |
| cro_0018 | −۰٫۵۹۸ | ۰٫۱۹۰ |
| cro_0019 | −۰٫۵۹۸ | ۰٫۱۹۱ |
| mut_0020 | −۰٫۷۵۳ | ۰٫۱۷۷ |
| mut_0021 | −۱٫۳۹۳ | ۰٫۱۱۳ |

اختلاف نفر اول و سوم (~۰٫۰۷) خیلی کوچک‌تر از SE (~۰٫۱۹) است. دو کاندید اول fitness یکسان دارند.

**شبیه‌سازی:** اگر شش برآورد مستقل با نویز N(0، ۰٫۲۰) گرفته شود، امید ریاضی بیشینه‌ی نویز ≈ **۰٫۲۵** و صدک ۹۵ ≈ **۰٫۴۸** است. یعنی «بهترین» گزارش‌شده به‌طور سیستماتیک خوش‌بینانه‌تر از ارزش واقعی است.

**چرا H:** عدد `best_stage3` در پایان‌نامه و `final_candidate.json` بدون بازه‌ی اطمینان یا ارزیابی کور همان ادعای اصلی روش است. بخش ۵ همین را به بخش ۸ ارجاع داده بود.

**پیشنهاد:**

1. برنده‌ی Stage III را فقط نامزد نهایی بدانید؛ fitness ادعا شده را با **ارزیابی کور جدا** (seed holdout جدید، E≥100 یا چند seed اجرا) گزارش کنید.
2. در انتخاب بین کاندیدها، اگر `|F_i − F_j| < 2·max(se_i, se_j)` آن‌ها را هم‌تراز بدانید یا از `fitness − se` (conservative) استفاده کنید.
3. E3 را دست‌کم به ۶۰–۱۰۰ برسانید **یا** چند ارزیابی مستقل با seedهای مختلف میانگین بگیرید.

### H2. ارزیابی Stage III همان seed holdout نسل صفر Stage II را دوباره استفاده می‌کند

**محل:** `adapter.py` وقتی `highway_eval_seed_base` تهی است: `eval_seed = seed + round_index`؛ Stage III همیشه `round_index=0` می‌فرستد (`validate.py`). holdout با `HOLDOUT_SEED_OFFSET=50003`.

**شواهد:**

</div>

```text
Stage II gen0: holdout base = 425 + 50003 = 50428
Stage II gen3: holdout base = 3425 + 50003 = 53428
Stage III:     holdout base = 425 + 50003 = 50428   ← برابر gen0
```

<div dir="rtl">

روی `highway_run`، همه‌ی شش کاندید Stage III فیلد `eval_seed_base=50428` دارند.

**چرا H:** کارت بازخورد و رتبه‌بندی نسل صفر روی همین سناریوها دیده شده‌اند. ارزیابی «نهایی» کور نیست؛ با بهینه‌سازی نسل‌به‌نسل روی خانواده‌ی seedهای `seed + g·1000` هم‌پوشانی دارد (حداقل با gen0). بخش ۳ همین ریسک را به بخش ۸ ارجاع داده بود.

**پیشنهاد:** برای Stage III (و هر ادعای نهایی) یک `highway_eval_seed_base` جدا بگذارید، مثلاً `seed + 10_000_000`، و آن را در config/manifest ثبت کنید. ارزیابی کور thesis را حتی از Stage III هم جدا کنید.

### M1. Warm-start Stage III از Stage II با بودجه‌ی K3 ناسازگار است

**محل:** `Stage3Config.highway_warm_start` (پیش‌فرض `False`)؛ `pipeline._stage3_config` همان پرچم CLI را عبور می‌دهد؛ `adapter` وزن policy والد را کپی می‌کند.

**وضعیت فعلی:** پیش‌فرض خاموش است — درست. اگر کاربر `--highway-warm-start` بزند، Stage III هم warm می‌شود در حالی که K3=70k با کاندیدهایی که از صفر شروع کرده‌اند قابل مقایسه نیست (همان هشدار بخش ۵ برای Stage II).

**پیشنهاد:** warm-start را فقط به Stage II محدود کنید، یا برای Stage III جداگانه `--stage3-warm-start` با پیش‌فرض خاموش و هشدار صریح در CLI/مستندات.

### M2. نگاشت CLI ناقص برای چند knob مهم

**محل:** `scripts/run_raise.py` → `RaiseRunConfig`.

| فیلد config | CLI | اثر |
|---|---|---|
| `highway_meta_fine_low` / `high` | **ندارد** (فقط `--highway-meta-fine-n`) | تغییر بازه‌ی سرعت meta_fine فقط از کد ممکن است |
| `domain` | **ندارد** | فقط از طریق ساخت دستی `RaiseRunConfig`؛ registry فقط `highway` دارد پس فعلاً کم‌خطر است |
| بقیهٔ فیلدهای اصلی | دارد | کامل به نظر می‌رسد (`llm_*`، feedback، operators، K2/K3، …) |

**پیشنهاد:** یا پرچم‌های `--highway-meta-fine-low/high` اضافه شود، یا در help گفته شود که فقط از config قابل تنظیم‌اند.

### M3. snapshot drift در Stage III فقط چند فیلد را می‌بیند

**محل:** `Stage3Runner._config_snapshot` فقط `train_env_steps`، `eval_episodes`، `seed`، `device`، `output_root`.

تغییر `highway_eval_mode`، `highway_warm_start`، `highway_action_mode` یا `highway_n_envs` روی resume هشدار نمی‌دهد، در حالی که نتایج را غیرقابل‌مقایسه می‌کند. Closed-loop بعد از بخش ۷ پوشش بهتری دارد.

**پیشنهاد:** همان کلیدهای حساس Stage II را به snapshot اضافه کنید.

### L1. اگر همه‌ی کاندیدهای Stage III fail شوند، `final_candidate` همان best Stage II است

**محل:** `pipeline.py`: `best_s3 = s3_runner.best_trained or best_s2`.

رفتار معقول برای resume است، ولی `best_stage3.json` / پیام «Stage III complete» ممکن است fitness Stage II را به‌عنوان نتیجه‌ی Stage III نشان دهد. بهتر است `status` صریح (`stage3_all_failed`) در manifest ثبت شود.

### L2. همه‌ی کاندیدهای Stage III با `round_index=0` و seed آموزش یکسان آموزش می‌بینند

عمدی و برای انصاف مقایسه‌ی نسبی خوب است؛ ولی در artifacts فیلد `train_seed` برای همه یکی است و ممکن است گمراه‌کننده به نظر برسد. یک خط توضیح در `spec.md` کافی است.

### L3. `console.py` فقط UX است

سطح لاگ و tqdm درست از API key و dump پرامپت جلوگیری می‌کنند. یافته‌ی جدیدی ندارد.

---

## ۳. پاسخ به پرسش‌های برنامه

### آیا انتخاب برنده با ۳۰ اپیزود از نظر آماری قابل اتکاست؟

**خیر.** با `fitness_se≈۰٫۱۸–۰٫۲۰` و N=6، اختلاف‌های معمول بین کاندیدها داخل نویز است و بیشینه‌گیری حدود ۰٫۲۵ واحد fitness را باد می‌کند (H1).

### آیا warm start در Stage III از مدل Stage II منطقی است؟

با پیش‌فرض فعلی (**خاموش**) بله. روشن کردنش بودجه‌ی مؤثر را خراب می‌کند (M1).

### آیا نگاشت CLI کامل است؟

تقریباً؛ exceptionهای مهم `meta_fine_low/high` و نبود `--domain` است (M2). مسیر pipeline→trainer برای بقیهٔ پرچم‌های highway درست است.

### آیا پیش‌فرض‌ها با بودجه‌ی محاسباتی می‌خوانند؟

روی GPU یک اجرای کامل با این پیش‌فرض‌ها در عمل حدود **۴ تا ۷ ساعت** طول می‌کشد (نام پوشه‌های `highway_4h_*` / `highway_7h_*` و ~۲۱ برچسب Stage II + ۶×۷۰k Stage III). از نظر پژوهشی قابل‌اجراست، ولی برای جاروی seed (مثلاً ۵ seed) به چند روز GPU نیاز است — جزئیات در گزارش موضوعات کلی.

---

## ۴. موارد خارج از دامنه

- فرمول fitness و `fitness_se` (بخش ۵؛ اینجا فقط اثر روی برنده‌ی نهایی)
- کارت بازخورد و بهینه‌سازی روی holdout (بخش ۳؛ اینجا فقط هم‌پوشانی seed)
- نمودارها و CI (بخش ۹)

---

## ۵. جدول اولویت

| شناسه | سطح | خلاصه | هزینه‌ی اصلاح |
|---|---|---|---|
| H1 | بالا | winner's curse با E3=30 | متوسط (پروتکل ارزیابی + اختیاری افزایش E3) |
| H2 | بالا | Stage III = holdout gen0 | کم |
| M1 | متوسط | warm-start Stage III در صورت روشن شدن | کم |
| M2 | متوسط | CLI ناقص برای meta_fine low/high | کم |
| M3 | متوسط | drift ناقص Stage III | کم |
| L1–L3 | پایین | fallback بدون برچسب، توضیح seed، console | کم |

**ترتیب پیشنهادی اصلاح:** H2 ← H1 ← M3 ← M1 ← M2 ← Lها.

---

## ۶. وضعیت اصلاحات

| شناسه | وضعیت | چه شد |
|---|---|---|
| H2 | انجام شد | Stage III روی `seed + 10_000_000` ارزیابی می‌شود (`STAGE3_EVAL_SEED_OFFSET`)، نه روی سناریوهای نسل صفر. holdout مؤثر با seed=425 از ۵۰۴۲۸ به ۱۰۰۵۰۴۲۸ رسید (با آموزش واقعی ۶۴ قدمی بررسی شد). |
| H1 | انجام شد | (۱) ارزیابی کور نهایی: برنده به‌تنهایی و بدون آموزش روی `seed + 20_000_000` با `--final-eval-episodes` (پیش‌فرض ۱۰۰) دوباره ارزیابی می‌شود. نتیجه در `manifest.json` → `final_eval` و متریک کامل در `final_candidate.json` ذخیره می‌شود. fitness انتخاب دست نمی‌خورد. (۲) `manifest.json` → `stage3_selection.tied_ids` کاندیدهایی را نشان می‌دهد که اختلافشان با برنده کمتر از `2·max(SE)` است، و یک هشدار هم در کنسول چاپ می‌شود. قاعده‌ی انتخاب (argmax) عوض نشد. |
| M3 | انجام شد | snapshot در Stage III حالا همه‌ی knobهای highway را پوشش می‌دهد (action mode، meta_fine، eval mode، warm start، n_envs، n_steps، deterministic، eval seed). |
| M1 | انجام شد | `highway_warm_start` فقط روی Stage II اثر دارد. برای Stage III پرچم جدای `--stage3-warm-start` با پیش‌فرض خاموش اضافه شد. |
| M2 | انجام شد | پرچم‌های `--domain`، `--highway-meta-fine-low` و `--highway-meta-fine-high` اضافه شدند. |
| L1 | انجام شد | وقتی هیچ کاندیدی در Stage III آموزش نبیند، `stage3_selection.status = all_failed` ثبت می‌شود، یک هشدار چاپ می‌شود و ارزیابی نهایی انجام نمی‌شود. |
| L2 | انجام شد | در `spec.md` توضیح داده شد که `train_seed` یکسان عمدی است. |
| L3 | — | یافته‌ای نداشت. |

**تست‌ها:** `raise_core/tests/test_stage3_part8_fixes.py`، به‌روزرسانی `test_pipeline.py`.

**یادداشت پایان‌نامه:** عددی که باید گزارش شود `final_eval.fitness` است، نه `best_stage3`. هزینه‌ی ارزیابی کور با ۱۰۰ اپیزود چند دقیقه است.

</div>
