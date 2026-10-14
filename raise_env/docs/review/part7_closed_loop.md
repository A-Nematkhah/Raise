<div dir="rtl">

# بررسی بخش ۷: حلقه‌ی بسته، checkpoint و ادامه بعد از قطعی

**دامنه:**

- `raise_core/raise_loop/runner.py` (~۶۳۰ خط)
- `raise_core/raise_loop/checkpoint.py` (~۸۰ خط)
- `raise_core/raise_loop/parallel_label.py` (~۱۲۰ خط)
- `raise_core/raise_loop/logging_io.py` (~۵۰ خط)
- `raise_core/raise_loop/report.py` (~۱۰۵ خط)
- `raise_core/raise_loop/config.py`
- `raise_core/population_io.py`، `raise_core/json_io.py`

نقاط اتصال هم بررسی شد: `explore.restore_id_counter`، `feedback.CandidateArchive.from_epoch_files`، و تست‌های `test_raise_loop_resume.py` / `test_resume_robustness.py`.

**روش:**

- کد خط به خط خوانده شد.
- مسیرهای resume با stub trainer و دستکاری عمدی checkpoint آزمایش شدند (crash بعد از نوشتن `epochs.jsonl`، کاندید ناموفق در `labeled_ids`، resume از `epoch_done`، تغییر `generations`).
- سرنوشت کاندید با خطای Stage II و رفتار `_needs_training` برای Score1-reject اندازه‌گیری شد.
- نوشتن اتمیک در برابر نوشتن ساده و اثر خط ناقص در `epochs.jsonl` روی `read_epochs` آزمایش شد.

**سطح‌بندی:** همان بخش‌های قبل (H / M / L).

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

هر نسل در `ClosedLoopRunner.run`:

</div>

```text
score_population (Score1)
  → to_label = untrained ∪ carried-for-reeval
  → checkpoint(phase=labeling)
  → label_candidates_parallel (train یا reevaluate؛ بعد از هر کاندید checkpoint)
  → rank by fitness → archive → reflection
  → append epochs.jsonl + write pop_epoch_XXX.json
  → breed next generation (اگر g+1 < G)
  → checkpoint(phase=epoch_done, next_epoch=g+1)
```

<div dir="rtl">

**ادامه بعد از قطعی:**

- checkpoint در دو جا نوشته می‌شود: `closed_loop/checkpoint.json` و کپی در ریشه‌ی run. بارگذاری اول `closed_loop/` را می‌خواند؛ اگر خراب باشد به کپی ریشه برمی‌گردد؛ اگر هر دو خراب باشند خطا می‌دهد و از صفر شروع نمی‌کند.
- فاز `labeling`: جمعیت امتیاز داده‌شده و `labeled_ids_this_epoch` بازیابی می‌شود؛ فقط کاندیدهای باقی‌مانده آموزش می‌بینند.
- فاز `epoch_done` / `init`: از `next_epoch` ادامه می‌دهد و نسل تمام‌شده را دوباره برچسب نمی‌زند.
- شناسه‌ها با `id_counter` ذخیره‌شده و در صورت نبودن، از همه‌ی idهای دیده‌شده در جمعیت، history و پوشه‌های `stage2_train/rXX_*` جلو می‌افتند.

**آنچه درست کار می‌کند (با تست و آزمایش تأیید شد):**

- resume بعد از اتمام کامل، دوباره آموزش نمی‌دهد.
- resume وسط labeling فقط کاندیدهای باقی‌مانده را آموزش می‌دهد.
- resume از `epoch_done` نسل بعد را درست شروع می‌کند و epoch تکراری نمی‌نویسد.
- شناسه‌ها بعد از resume دوباره صادر نمی‌شوند.
- label موازی نتایج را به ترتیب `to_label` به `on_done` می‌دهد؛ کرش یک کاندید epoch را نمی‌کشد.
- نوشتن checkpoint با `atomic_write_json` (temp + `os.replace` + fsync) است و در صورت خطای serialize فایل قبلی را نگه می‌دارد.

</div>

---

<div dir="rtl">

## ۲. یافته‌ها

### H1. اگر crash بین `append_epoch_record` و checkpointِ `epoch_done` رخ دهد، همان نسل دو بار در `epochs.jsonl` نوشته می‌شود

**محل:** `runner.py` بعد از labeling: اول `append_epoch_record`، بعد `pop_epoch_*.json`، بعد breeding، بعد `_checkpoint(phase="epoch_done")`.

**شواهد:** checkpoint را عمداً به `phase=labeling` و `next_epoch=0` برگرداندیم در حالی که یک ردیف epoch=0 در `epochs.jsonl` بود. بعد از resume، فایل شد `[0, 0]` و ردیف دوم `resumed=True` داشت.

**اثر:**

- نمودارها و تحلیل پایان‌نامه که از `epochs.jsonl` می‌خوانند (از جمله `plot_raise_run` و `evolve_best_fitness`) نسل تکراری می‌بینند.
- `manifest.n_epochs` و طول history با تعداد نسل واقعی یکی نیست.
- این پنجره‌ی خطر کوتاه است (بین پایان labeling و نوشتن `epoch_done`)، ولی دقیقاً بعد از گران‌ترین کار هر نسل است؛ قطعی یا kill در همان لحظه غیرواقعی نیست.

**پیشنهاد:**

- **(الف) پیشنهادی:** قبل از append، اگر آخرین ردیف همان `epoch` را دارد جایگزینش کن (یا اصلاً ننویس). نوشتن epoch را با همان تراکنش ذهنیِ `epoch_done` یکی کن: اول ردیف را به‌صورت اتمیک بنویس/جایگزین کن، بعد checkpoint را با `history` شامل همان ردیف به‌روز کن.
- **(ب):** در resume، اگر `epochs.jsonl` همین `epoch` را دارد، append نکن و از همان ردیف برای history استفاده کن.

</div>

---

<div dir="rtl">

### H2. کاندید Score1-reject شده همچنان آموزش Stage II کامل می‌بیند

**محل:** `_needs_training` فقط می‌پرسد `reward_fn is not None` و `fitness is None`. `score_population` برای hard-reject مقدار `score=-inf` و `score1_rejected=True` می‌گذارد، ولی اجرا را متوقف نمی‌کند.

**شواهد:** `_needs_training` روی کاندید با `score1_rejected=True` برابر `True` است. این همان موردی است که بخش‌های ۴ و ۶ به بخش ۷ ارجاع دادند.

**اثر:** هر پاداش ردشده‌ی Score1 یک بودجه‌ی کامل K2 (پیش‌فرض ۱۲ هزار گام PPO + ۳۰ اپیزود holdout) مصرف می‌کند. در اجرای واقعی این می‌تواند ده‌ها دقیقه تا بیش از یک ساعت برای یک کاندید بی‌ارزش باشد. کاندید بعداً در رتبه‌بندی والدین معمولاً ته صف می‌ماند، پس تقریباً فقط هزینه است.

**پیشنهاد:** اگر `score1_rejected` یا `score == -inf` است، آموزش را رد کن و در epoch به‌عنوان `skipped_score1` ثبت کن (شبیه fail، بدون PPO). اگر بعداً خواستید برای پژوهش همچنان آموزش بدهید، پشت یک پرچم صریح باشد.

</div>

---

<div dir="rtl">

### M1. کاندید ناموفق Stage II بعد از ثبت در `labeled_ids_this_epoch` دیگر retry نمی‌شود

**محل:** `on_done` در هر حال (ok یا failed) id را به `labeled_ids_this_epoch` اضافه می‌کند؛ resume همان مجموعه را `already_done` می‌کند.

**شواهد:** کاندیدی که fitness نداشت و idاش در `labeled_ids_this_epoch` بود، بعد از resume دوباره به trainer داده نشد و fitnessاش `None` ماند.

**تمایز مهم:**

- اگر crash **وسط** آموزش یک کاندید رخ دهد و هنوز در `labeled_ids` نباشد، آن کاندید دوباره آموزش می‌بیند. این درست است.
- اگر آموزش با exception تمام شود، fail ثبت می‌شود، بعداً در همان epoch قطعی رخ دهد و resume شود، آن fail دیگر فرصت دوم ندارد حتی اگر خطا گذرا بوده باشد (OOM، timeout، فایل قفل‌شده).

**اثر:** در عمل کمتر از H1 دیده می‌شود، ولی یک اسلات جمعیت را تا پایان نسل بدون fitness نگه می‌دارد. نام `labeled_ids` هم گمراه‌کننده است چون failها را هم شامل می‌شود.

**پیشنهاد:**

- فقط موفقیت‌ها را در `labeled_ids_this_epoch` بگذار؛ یا
- در resume، هر idای که در لیست است ولی `fitness` ندارد را دوباره در صف بگذار (با سقف یک یا دو retry).

و در `epochs.jsonl` فیلد جدا برای `failed_ids` نگه دار؛ `labeled_ids` فقط موفق‌ها باشد.

</div>

---

<div dir="rtl">

### M2. به‌جز checkpoint، بقیه‌ی خروجی‌های حلقه crash-safe نیستند

**محل:**

- `logging_io.append_epoch_record`: append معمولی؛ بدون fsync و بدون بازیابی خط ناقص.
- `logging_io.write_json` (برای `pop_epoch_*.json` و `manifest.json`): بازنویسی مستقیم، بدون temp/`os.replace`.
- در حالی که `json_io.atomic_write_json` همین کار را برای checkpoint انجام می‌دهد.

**شواهد:**

- یک خط ناقص در `epochs.jsonl` باعث می‌شود `read_epochs` با `JSONDecodeError` بمیرد؛ در نتیجه `REPORT.txt` و هر ابزاری که epochs را می‌خواند از کار می‌افتد.
- `write_json` هیچ اشاره‌ای به atomic ندارد.

**اثر:** قطعی وسط نوشتن epoch یا `pop_epoch` می‌تواند آرشیو بازخورد روی resume را خراب کند (`CandidateArchive.from_epoch_files`) یا کل گزارش پس‌ازاجرا را بشکند، با این‌که خود state حلقه از checkpoint درست بازیابی شود.

**پیشنهاد:** `write_json` را به `atomic_write_json` وصل کن. برای JSONL یا خط را اتمیک جایگزین کن، یا موقع خواندن خطوط خراب را با هشدار رد کن.

</div>

---

<div dir="rtl">

### M3. تغییر پیکربندی هنگام resume خاموش نادیده گرفته می‌شود

**محل:** checkpoint فیلد `config` را ذخیره می‌کند، ولی runner آن را با `ClosedLoopConfig` فعلی مقایسه نمی‌کند. Stage III برعکس `_warn_config_drift` دارد.

**شواهد:** با stub، `apply_fast_profile` سقف `generations` را ۲ می‌گذارد؛ پس آزمایش افزایش G محدود بود. از نظر کد، حلقه فقط به `cfg.generations` و `next_epoch` نگاه می‌کند و به `ckpt["config"]` کاری ندارد.

**اثر:** ادامه با N یا K2 یا seed دیگر می‌تواند اجراهای نیمه‌کاره را با بودجه‌ی ناسازگار قاطی کند، بدون هیچ هشداری در لاگ.

**پیشنهاد:** مثل Stage III، اختلاف‌های مهم (`population_size`، `generations`، `stage2_train_steps`، `seed`، `domain`) را هشدار بده یا در حالت سخت رد کن.

</div>

---

<div dir="rtl">

### L1. گزارش مسیر اشتباه برای `manifest.json` می‌نویسد

**محل:** `report.build_closed_loop_report` در بخش files به `output_dir/manifest.json` اشاره می‌کند. runner مانيفست را در `closed_loop/manifest.json` می‌نویسد. در آزمایش، فایل ریشه وجود نداشت و فایل `closed_loop/` وجود داشت.

**پیشنهاد:** مسیر را درست کن (مانيفست ریشه احتمالاً مال پایپ‌لاین کامل است و به بخش ۸ مربوط می‌شود).

</div>

---

<div dir="rtl">

### L2. `label_metrics` برای کاندید failشده `fitness=-inf` می‌گذارد و او را در `labeled_ids` می‌آورد

**محل:** `_compact_metrics` همیشه `candidate_fitness` را می‌نویسد؛ برای کاندید بدون متریک این مقدار `-inf` است. `labeled_ids` همه‌ی `to_label` غیرcarried را می‌گیرد، نه فقط `n_label_ok`.

**اثر:** در تحلیل بعدی ممکن است `-inf` به‌عنوان fitness واقعی خوانده شود، مگر فیلتر شود. با M1 هم‌مسیر است.

</div>

---

<div dir="rtl">

### L3. شمارنده‌های epoch بعد از mid-resume فقط کار همین پروسس را نشان می‌دهند

این عمدی است و با `resumed=True` علامت خورده. برای پایان‌نامه باید دانست که `n_label_ok` در آن ردیف مجموع کامل نسل نیست. می‌شود در docs/`spec.md` یک جمله نوشت.

</div>

---

<div dir="rtl">

### L4. دو نسخه‌ی checkpoint (ریشه و `closed_loop/`)

طراحی دفاعی و تست‌شده است. فقط در مستندات باید روشن باشد که منبع حقیقت `closed_loop/checkpoint.json` است و کپی ریشه پشتیبان است.

</div>

---

<div dir="rtl">

## ۳. پاسخ به پرسش‌های برنامه

### آیا ادامه بعد از قطعی در همه‌ی فازها درست است؟

- **completed / epoch_done / init:** بله (با تست).
- **labeling وسط کار:** بله برای کاندیدهای باقی‌مانده.
- **حفره:** اگر crash بعد از نوشتن `epochs.jsonl` و قبل از `epoch_done` باشد، labeling دوباره تمام می‌شود و **ردیف epoch تکراری** نوشته می‌شود (H1).
- **حفره:** fail ثبت‌شده دیگر retry نمی‌شود (M1).

### کاندیدی که آموزشش fail شده چه سرنوشتی دارد؟

در جمعیت می‌ماند، fitness ندارد (در compact metrics به‌صورت `-inf` دیده می‌شود)، در رتبه‌بندی والدین ته صف است، و در صورت resume همان نسل دوباره آموزش نمی‌بیند. در نسل بعد فقط اگر به‌عنوان والد mutation/crossover انتخاب شود اثر دارد؛ با elitism حمل نمی‌شود.

### آیا آموزش موازی با thread ایمن است؟

برای مدل فعلی بله: هر کاندید جداست، `on_done` و checkpoint روی thread اصلی و به ترتیب ثابت اجرا می‌شوند، و کرش یک کاندید epoch را abort نمی‌کند. خود کد هشدار می‌دهد که با GIL و `n_envs>1` لزوماً سریع‌تر نمی‌شود. فرض thread-safety برای خود SB3/env بیرون از این بخش است و اینجا تست ادغامی سنگینی برای `workers>1` با PPO واقعی نیست.

### آیا شناسه‌ها هرگز تکراری صادر نمی‌شوند؟

در مسیر resume، با `id_counter` و استخراج از idهای موجود، تست‌ها نشان می‌دهند که تکراری صادر نمی‌شود. شمارنده‌ی سریال سراسری است (`cro_0007` و `mut_0007` هم‌زمان ساخته نمی‌شوند چون یک counter مشترک است).

### آیا رکورد هر نسل داده‌ی لازم برای پایان‌نامه را دارد؟

تقریباً غنی است: ranking، fitness، Score1، origin_counts، diversity، reeval، duplicateها. ضعف‌ها: تکرار احتمالی ردیف (H1)، نام‌گذاری `labeled_ids`، و شکنندگی خواندن JSONL ناقص (M2). برای ادعای همگرایی باید بعد از اصلاح H1 به `epochs.jsonl` اعتماد کرد.

</div>

---

<div dir="rtl">

## ۴. موارد خارج از دامنه

| مورد | بخش مقصد |
|---|---|
| Stage III checkpoint و انتخاب برنده‌ی نهایی | ۸ |
| مانيفست ریشه و CLI resume end-to-end | ۸ |
| نمودارها و تفسیر `epochs.jsonl` در پایان‌نامه | ۹ |
| backoff کلاینت LLM هنگام قطعی طولانی | ۳ (یافته‌های باز) |

</div>

---

<div dir="rtl">

## ۵. جدول اولویت

| شناسه | سطح | خلاصه | هزینه‌ی اصلاح |
|---|---|---|---|
| H1 | بالا | ردیف تکراری epoch بعد از crash بین append و `epoch_done` | کم |
| H2 | بالا | آموزش PPO برای Score1-reject | کم |
| M1 | متوسط | fail ثبت‌شده روی resume retry نمی‌شود؛ `labeled_ids` گمراه‌کننده | کم |
| M2 | متوسط | `epochs.jsonl` / `pop_epoch` / manifest غیراتمی؛ خط ناقص خواندن را می‌شکند | کم |
| M3 | متوسط | drift پیکربندی روی resume بدون هشدار | کم |
| L1–L4 | پایین | مسیر اشتباه manifest در گزارش، `-inf` در metrics، شمارنده‌ی mid-resume، توضیح dual-ckpt | کم |

**ترتیب پیشنهادی اصلاح:** H1 ← H2 ← M1 ← M2 ← M3 ← Lها.

</div>
