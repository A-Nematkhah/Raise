<div dir="rtl">

# بررسی بخش ۹: خروجی‌ها، نمودارها و تست‌ها

**دامنه:**

- `domains/highway/post_run.py`، `domains/highway/viz.py`
- `scripts/plot_raise_run.py`، `scripts/print_raise_report.py`
- همه‌ی `**/tests/test_*.py` (~۵۱ فایل)
- `.github/workflows/raise-ci.yml`
- گزارش متنی `raise_core/raise_loop/report.py`

**روش:**

- اسکریپت‌های plot/viz و اتصال‌شان از `pipeline` خوانده شد.
- نمودار closed-loop با فیلدهای واقعی `epochs.jsonl` اجرای `highway_run` مقایسه شد.
- پوشش تست و CI فهرست شد؛ فقط یک تست `slow` برای rollout واقعی محیط پیدا شد.
- هیچ فراخوانی واقعی Groq در تست‌ها وجود ندارد (عمدی و درست برای CI).

**سطح‌بندی:** H / M / L.

</div>

---

<div dir="rtl">

## ۱. خلاصه‌ی معماری

</div>

```text
RaisePipeline (not --fast)
  └─ write_highway_run_artifacts(run_dir)
        ├─ plot_raise_run.run_plots → plots/*.png
        └─ visualize_best_policy → plots/viz/*.gif

print_raise_report.py → closed_loop/REPORT.txt (text only)
CI: Ubuntu, Python 3.10, CPU torch, pytest -m "not slow"
```

<div dir="rtl">

نمودارهای اصلی: Score1/fitness در طول epochها، متریک‌های جمعیت Stage II/III، خلاصه‌ی bestها، جزئیات highway برای best II/III، دسته‌های rejection.

---

## ۲. یافته‌ها

### H1. نمودار «Best fitness» بدون نوار خطا، بهبود کاذب را القا می‌کند

**محل:** `plot_raise_run.plot_closed_loop_epochs` — فقط `evolve_best_fitness` را رسم می‌کند؛ `fitness_se` در `label_metrics` هست ولی استفاده نمی‌شود.

**شواهد:** در `highway_run`، `fitness_se` معمولاً ۰٫۱۸–۰٫۲۲ است. منحنی best از −۰٫۹۱ → −۰٫۲۲ → −۰٫۵۳ → −۰٫۶۰ می‌رود؛ پرش نسل ۱ داخل حدود ۲·SE است و نسل‌های بعد دوباره بدتر می‌شوند. بدون SE، شکل نمودار برای فصل نتایج گمراه‌کننده است.

**پیشنهاد:** برای هر epoch، SE کاندید `evolve_best_id` را از `label_metrics` بخوانید و `fill_between` یا errorbar بکشید؛ در caption بنویسید اختلاف کمتر از ~۰٫۴ معنی‌دار نیست (بخش ۵).

### M1. پنل Score1 در همان نمودار با پنل fitness هم‌تراز نیست

**محل:** همان تابع: محور بالا `best_score1` (بهترین Score1 جمعیت) و محور وسط `evolve_best_fitness` (بهترین از نظر fitness).

روی gen0 اجرای واقعی: `best_score1` مال `ini_0004` است، ولی `evolve_best_id` برابر `ini_0000` است. خواننده فکر می‌کند هر دو مال یک کاندیدند.

**پیشنهاد:** یا `evolve_best_score1` رسم شود، یا هر دو سری با برچسب صریح «max Score1» در برابر «Score1 of fitness-best».

### M2. GIF/viz روی seed آموزش رسم می‌شود، نه holdout نهایی

**محل:** `pipeline` → `write_highway_run_artifacts(..., seed=cfg.seed)`؛ `viz.visualize_best_policy` همان seed را برای rollout می‌گیرد.

Seed پیش‌فرض ۴۲۵ نزدیک فضای آموزش است، نه لزوماً سناریوهای holdout (`+50003`). GIF برای ارائه خوب است، ولی «نمایش عملکرد holdout» نیست.

**پیشنهاد:** seed viz را `seed + HOLDOUT_SEED_OFFSET` (یا seed کور Stage III پیشنهادی بخش ۸) کنید و در نام فایل/متادیتا ثبت کنید.

### M3. شکاف پوشش تست برای خروجی‌ها و مسیر واقعی

| ناحیه | وضعیت تست |
|---|---|
| plot closed-loop epochs | یک تست دودویی (`test_post_run_plots`) با JSON مصنوعی قدیمی |
| `viz.py` / GIF | **بدون تست** |
| `print_raise_report.py` / `run_raise.py` CLI | **بدون تست** |
| `console.py` | **بدون تست** |
| Stage III resume | stub — خوب |
| PPO واقعی | فقط `test_wrapper_step` (+ در عمل `test_speed_knobs` با ۶۴ قدم) |
| LLM واقعی (Groq) | **هیچ** — درست برای CI؛ ولی smoke دستی مستند نشده |
| CI | فقط `not slow`؛ روی Ubuntu؛ torch CPU |

**پیشنهاد:** تست‌های واحد برای: (۱) plot با `evolve_best_*` و `fitness_se`، (۲) resolve checkpoint در viz با مسیر ساختگی، (۳) `run_raise.py --help` / رد `--llm seed` بدون `--allow-seed-llm` (بخشی در `test_seed_llm_gate` هست). یک job اختیاری nightly با `slow` مفید است ولی اجباری نیست.

### L1. `post_run` خطا را می‌بلعد

عمدی است تا pipeline بعد از آموزش طولانی برای plot نشکند. فقط در log می‌آید؛ بهتر است `manifest["post_run"] = art` نوشته شود تا شکست plot در artifacts دیده شود.

### L2. تست plot هنوز فیلدهای پایپ‌لاین قدیمی را fixture می‌کند

`n_labeled_dataset` و `stats.soft_success_mean` مربوط به پایپ‌لاین پیش از ساده‌سازی‌اند. تست پاس می‌شود چون plot فقط `best_score1` را لازم دارد، ولی fixture گمراه‌کننده است.

### L3. CI مسیر `raise_env/**` را محدود کرده

تغییر در ریشه‌ی repo بیرون از `raise_env` (مثلاً README) CI را راه نمی‌اندازد — قابل قبول. نسخه‌ی Python فقط ۳٫۱۰ است؛ با pinهای فعلی منطقی است.

---

## ۳. پاسخ به پرسش‌های برنامه

### آیا نمودارها همان چیزی را نشان می‌دهند که پایان‌نامه ادعا می‌کند؟

تا حدی. متریک‌های SR/CR/TR و مقایسهٔ Stage II/III درست‌اند. ولی منحنی closed-loop بدون SE و با Score1 ناهم‌تراز (H1، M1) برای ادعای «بهبود نسل‌به‌نسل» خطرناک است. GIF هم holdout نهایی نیست (M2).

### چه چیزهایی تست نشده‌اند؟

- آموزش PPO کامل (۷۰k) و LLM واقعی: عمداً خارج از CI.
- `viz`، بخش بزرگی از plotهای Stage II/III، و CLI کامل.
- پوشش واحد برای منطق‌های جدید بخش‌های ۱–۷ عموماً خوب است؛ گلوگاه بخش ۹ خروجی بصری است.

---

## ۴. جدول اولویت

| شناسه | سطح | خلاصه | هزینه |
|---|---|---|---|
| H1 | بالا | plot fitness بدون SE | کم |
| M1 | متوسط | Score1 vs evolve-best ناهم‌تراز | کم |
| M2 | متوسط | viz روی seed غیرholdout | کم |
| M3 | متوسط | شکاف تست viz/CLI/plot | متوسط |
| L1–L3 | پایین | manifest post_run، fixture کهنه، دامنه‌ی CI | کم |

**ترتیب پیشنهادی:** H1 ← M1 ← M2 ← L1 ← M3 ← L2.

---

## ۵. وضعیت اصلاحات

| شناسه | وضعیت | چه شد |
|---|---|---|
| H1 | انجام شد | هر سطر `epochs.jsonl` حالا `evolve_best_fitness_se` هم دارد (برای elite منتقل‌شده هم، که در `label_metrics` نیست). نمودار best fitness نوار خطای ±۲ SE و شناسه‌ی کاندید را نشان می‌دهد. برای اجراهای قدیمی SE از `label_metrics` خوانده می‌شود. |
| M1 | انجام شد | پنل بالا Score1 همان کاندید fitness-best (`evolve_best_score1`) را رسم می‌کند. max Score1 جمعیت به‌صورت خط‌چین کم‌رنگ و با برچسب جدا نشان داده می‌شود. |
| M2 | انجام شد | GIF روی ترافیک holdout و خانواده‌ی seed ارزیابی کور (`seed + 20_000_000`) ساخته می‌شود. `viz_manifest.json` فیلدهای `traffic`، `seed_base` و `deterministic` را ثبت می‌کند. با یک policy واقعی بررسی شد. |
| L1 | انجام شد | `manifest.json` → `post_run` شامل تعداد نمودارها، موفقیت viz و خطاها است. |
| M3 | انجام شد (بخشی) | تست‌های جدید: `closed_loop_series`، تولید نمودار روی خروجی واقعی `--fast`، `post_run_summary`، `_resolve_checkpoint`، rollout با env ساختگی (holdout در برابر train) و پرچم‌های CLI (`--print-config`). `console.py` و `print_raise_report.py` هنوز تست ندارند. |
| L2 | انجام شد | fixture نمودار با فیلدهای فعلی `epochs.jsonl` جایگزین شد. |
| L3 | بدون تغییر | دامنه‌ی CI قابل قبول است. |

**تست‌ها:** `domains/highway/tests/test_post_run_plots.py` (بازنویسی)، `domains/highway/tests/test_viz.py` (جدید).

</div>
