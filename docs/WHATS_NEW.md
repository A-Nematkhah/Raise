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

*ادامهٔ تغییرات بعدی از همین‌جا اضافه شود.*
